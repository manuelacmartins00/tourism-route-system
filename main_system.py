# main_system.py
# VERSÃO BASELINE:
#    - Groq (llama-3.1-8b-instant)
#    - Sem análise de sentimento
#    - Sem AHP — pesos fixos no RouteEvaluator
#    - Sem login/autenticação
#    - Sem filtro de histórico de visitas
#    - RAG + SHAP + Mapa OSRM + Day Planner mantidos

import os
import sys
import math
import numpy as np
import json
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from src.rag.rag_setup import POI_RAG
from src.llm.llm_orchestrator import LlamaOrchestrator, select_algorithm_deterministic
from src.optimizers.route_evaluator import RouteEvaluator, POI
from src.utils.location_resolver import LocationResolver
from src.optimizers.tourism_aco import TourismACO
from src.optimizers.tourism_ga import TourismGA
from src.optimizers.tourism_psoa import TourismPSOA
from src.optimizers.greedy_planner import GreedyPlanner
from src.utils.data_loader import load_pois_from_json
from src.utils.shap_explainer import RouteExplainer

def _within_radius(poi_lat: float, poi_lon: float,
                   center_lat: float, center_lon: float,
                   radius_km: float) -> bool:
    """Verifica se um POI está dentro do raio a partir do centro."""
    R = 6371
    dlat = math.radians(poi_lat - center_lat)
    dlon = math.radians(poi_lon - center_lon)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(center_lat)) *
         math.cos(math.radians(poi_lat)) *
         math.sin(dlon / 2) ** 2)
    dist_km = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return dist_km <= radius_km

class TourismRouteSystem:
    """
    Sistema baseline: RAG + LLM (Groq) + Otimização + SHAP + Mapa OSRM + Day Planning

    Pipeline: LLM → RAG → Otimização → SHAP → Explicação LLM → Mapa OSRM → Day Planning
    """

    def __init__(self, api_key: str = None):
        print("🚀 Inicializando sistema...\n")

        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            print("\n" + "=" * 70)
            print("❌ ERRO: GROQ_API_KEY não configurada!")
            print("=" * 70)
            print("\n📋 Para configurar:")
            print("\n1️⃣  Criar API Key Groq:")
            print("   • Vai a: https://console.groq.com")
            print("   • Clica 'API Keys' → 'Create API Key'")
            print("   • COPIA a key (gsk_...)")
            print("\n2️⃣  Configurar no Projeto:")
            print("   • Cria/edita ficheiro .env na raiz do projeto")
            print("   • Adiciona: GROQ_API_KEY=gsk_TuaKeyAqui")
            print("   • Guarda o ficheiro")
            print("\n3️⃣  Reiniciar Terminal e executar novamente\n")
            print("=" * 70 + "\n")
            raise ValueError("GROQ_API_KEY não configurada!")

        print(f"✓ Usando Groq")
        print(f"✓ API Key: {self.api_key[:20]}...{self.api_key[-10:]}\n")

        try:
            self.llm = LlamaOrchestrator(api_key=self.api_key)
        except Exception as e:
            print(f"\n❌ Erro ao conectar com Groq: {e}\n")
            raise

        print("📚 Carregando RAG...")
        self.rag = POI_RAG(data_file="data/portugal_todos_pois_final_enriched.json")

        print("📊 Carregando dados...")
        self.all_pois_data = load_pois_from_json("data/portugal_todos_pois_final_enriched.json")

        print("🗺️  Carregando resolver geográfico...")
        self.location_resolver = LocationResolver()

        self.transit_service = None
        try:
            from src.transit.transit_service import TransitService
            ts = TransitService()
            ts.load(use_cache=True)
            self.transit_service = ts
            print("🚌 TransitService carregado!\n")
        except Exception as e:
            print(f"⚠️  TransitService não disponível: {e}\n")

        print("✅ Sistema pronto!\n")

    def plan_route(self,
               user_query: str,
               use_shap: bool = True,
               verbose: bool = True,
               force_algorithm: str = None,
               transit_service=None) -> Dict:
        """
        Pipeline completo: LLM → RAG → Otimização → SHAP → Explicação LLM → Mapa → Day Planning

        Args:
            user_query:      Query em linguagem natural (PT ou EN)
            use_shap:        Se True, gera análise SHAP
            verbose:         Se True, imprime progresso
            force_algorithm: "ACO", "GA", "PSO", "GREEDY", ou None (LLM escolhe)

        Returns:
            Dict com rota, métricas, explicações SHAP, LLM, mapa e planeamento por dias
        """

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"📝 Query: {user_query}")
            print(f"{'=' * 70}\n")

        # ========== PASSO 1: LLM EXTRAI PREFERÊNCIAS ==========
        if verbose:
            print("🤖 [LLM] Extraindo preferências...")

        preferences = self.llm.extract_preferences(user_query)

        mode_labels = {
            "foot": "A pé",
            "car": "Carro",
            "public_transport": "Transportes públicos",
            "fastest": "Mais rápido por segmento"
        }

        if verbose:
            label = mode_labels.get(preferences.transport_mode, preferences.transport_mode)
            print(f"   ✓ Tempo: {preferences.max_time} min")
            print(f"   ✓ Orçamento: €{preferences.max_cost}")
            print(f"   ✓ Categorias: {preferences.preferred_categories}")
            print(f"   ✓ Interesses: {preferences.interests}")
            print(f"   ✓ Transporte: {label}\n")

        # ── Verificar campos em falta ─────────────────────────────────
        if preferences.missing_fields:
            return {
                "status": "needs_clarification",
                "query": user_query,
                "missing_fields": preferences.missing_fields,
                "preferences_so_far": {
                    "max_time": preferences.max_time,
                    "max_cost": preferences.max_cost,
                    "location": preferences.location,
                    "categories": preferences.preferred_categories,
                }
            }

        # ── Resolução geográfica ──────────────────────────────────────
        geo = None
        if preferences.location:
            geo = self.location_resolver.resolve(preferences.location)
            if geo and verbose:
                lat_c, lon_c, radius_c = geo
                print(f"   📍 '{preferences.location}' → "
                      f"({lat_c:.4f}, {lon_c:.4f}), raio {radius_c:.0f}km\n")
        
        # ── Resolução do ponto de partida ───
        start_geo = None
        if hasattr(preferences, 'start_location') and preferences.start_location:
            start_geo = self.location_resolver.resolve(preferences.start_location)
            if start_geo and verbose:
                print(f"   🏨 Ponto de partida: '{preferences.start_location}' → ({start_geo[0]:.4f}, {start_geo[1]:.4f})\n")
        if not start_geo and geo:
            start_geo = geo  # fallback: centro da cidade
                
        # ========== PASSO 2: RAG BUSCA POIs ==========
        if verbose:
            print("🔍 [RAG] Recuperando POIs relevantes...")

        rag_query = self.llm.generate_rag_query(preferences, user_history=None)

        if verbose:
            print(f"   Query texto: '{rag_query}'")
            print(f"   Category filter: {preferences.preferred_categories}")
            print(f"   Max cost: {preferences.max_cost}\n")

        lat_min = lat_max = lon_min = lon_max = None
        if geo:
            center_lat, center_lon, radius_km = geo
            delta = radius_km / 111.0
            lat_min = center_lat - delta
            lat_max = center_lat + delta
            lon_min = center_lon - delta
            lon_max = center_lon + delta

        rag_results = self.rag.query(
            text=rag_query,
            n_results=25,
            category_filter=preferences.preferred_categories,
            max_cost=preferences.max_cost,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        )
        candidate_pois = rag_results['pois']

        # Fallback: se POIs insuficientes para preencher 60% do tempo disponível
        tempo_candidatos = sum(p['duration'] for p in candidate_pois)
        if tempo_candidatos < 0.6 * preferences.max_time:
            if verbose:
                print(f"   ⚠️ Fallback: {tempo_candidatos}min de candidatos para {preferences.max_time}min disponíveis")
                print(f"   🔄 Re-query sem filtro de categorias...")
            rag_results_fallback = self.rag.query(
                text=rag_query,
                n_results=50,
                category_filter=None,
                max_cost=preferences.max_cost,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max
            )
            # Merge: adiciona POIs do fallback que não estão já nos candidatos
            existing_ids = {p['id'] for p in candidate_pois}
            for p in rag_results_fallback['pois']:
                if p['id'] not in existing_ids:
                    candidate_pois.append(p)
                    existing_ids.add(p['id'])
            if verbose:
                print(f"   ✓ Candidatos após fallback: {len(candidate_pois)}")
                
        # Hard filter geográfico pós-RAG — remove POIs fora do raio exacto
        # (o bounding box do RAG é um quadrado; este filtro corta os cantos)
        if geo:
            center_lat, center_lon, radius_km = geo
            before = len(candidate_pois)
            candidate_pois = [
                p for p in candidate_pois
                if _within_radius(p['lat'], p['lon'], center_lat, center_lon, radius_km)
            ]
            if verbose:
                print(f"   📍 Hard filter geográfico: {before} → {len(candidate_pois)} POIs (raio {radius_km:.0f}km)\n")

            if len(candidate_pois) == 0:
                if verbose:
                    print("   ⚠️ Hard filter removeu todos os POIs — a relaxar para bounding box\n")
                rag_results_nogeo = self.rag.query(
                    text=rag_query,
                    n_results=25,
                    category_filter=preferences.preferred_categories,
                    max_cost=preferences.max_cost,
                )
                candidate_pois = rag_results_nogeo['pois']

        if len(candidate_pois) == 0:
            print(f"\n{'=' * 70}")
            print("⚠️  ERRO: RAG RETORNOU 0 POIs!")
            print(f"{'=' * 70}")
            print("Possíveis causas:")
            print("  1. Categorias extraídas não existem no JSON")
            print("  2. ChromaDB precisa ser reconstruído — apaga data/chroma_db")
            print(f"{'=' * 70}\n")
            return {
                "error": "NO_POIS_FOUND",
                "query": user_query,
                "preferences": {
                    "max_time": preferences.max_time,
                    "max_cost": preferences.max_cost,
                    "categories": preferences.preferred_categories,
                    "interests": preferences.interests
                },
                "rag_query": rag_query,
                "n_candidates": 0
            }

        if verbose:
            print(f"   ✓ Query RAG: '{rag_query}'\n")

        # ========== PASSO 3: SELECIONAR ALGORITMO ==========
        if verbose:
            print("🎯 [LLM] Selecionando algoritmo...")

        if force_algorithm:
            selected_algo = force_algorithm.upper()
            if selected_algo not in ["ACO", "GA", "PSO", "GREEDY"]:
                print(f"⚠️ Algoritmo inválido '{selected_algo}', usando ACO")
                selected_algo = "ACO"
            if verbose:
                print(f"   ✓ Algoritmo (FORÇADO): {selected_algo}\n")
        else:
            selected_algo = select_algorithm_deterministic(len(candidate_pois), preferences.max_time)
            if verbose:
                print(f"   ✓ Algoritmo (determinístico): {selected_algo}\n")

        # ========== PASSO 4: PREPARAR OTIMIZAÇÃO ==========
        optimizer_pois = []
        for p in candidate_pois:
            poi_obj = POI(
                id=int(p['id']), name=p['name'],
                lat=p['lat'], lon=p['lon'],
                category=p['category'], score=p['score'],
                duration=p['duration'],
                opening_time=p['opening_time'],
                closing_time=p['closing_time'],
                cost=p['cost']
            )
            optimizer_pois.append(poi_obj)

        if len(optimizer_pois) == 0:
            print("\n⚠️  ERRO: Nenhum POI para otimizar!")
            return {"error": "NO_OPTIMIZER_POIS", "query": user_query, "n_candidates": 0}

        n_pois = len(optimizer_pois)

        if verbose:
            print(f"   ✓ POIs para otimização: {n_pois}\n")

        # Sub-matriz de tempos reais (TransitService) ou Haversine (fallback)
        if transit_service is not None:
            if verbose:
                print(f"   🚌 A construir matriz de tempos reais ({preferences.transport_mode})...\n")
            sub_distance_matrix = transit_service.build_cost_matrix(
                optimizer_pois,
                mode=preferences.transport_mode
            )
        else:
            from src.utils.distance_calculator import haversine
            TRANSPORT_SPEED = {"foot": 5, "car": 50, "public_transport": 20}
            speed_kmh = TRANSPORT_SPEED.get(preferences.transport_mode, 5)
            if verbose:
                print(f"   🚗 Velocidade Haversine: {speed_kmh} km/h ({preferences.transport_mode})\n")
            sub_distance_matrix = np.zeros((n_pois, n_pois))
            for i, poi_i in enumerate(optimizer_pois):
                for j, poi_j in enumerate(optimizer_pois):
                    if i != j:
                        d_km = haversine(poi_i.lat, poi_i.lon, poi_j.lat, poi_j.lon)
                        sub_distance_matrix[i][j] = (d_km / speed_kmh) * 60

        # Preferências — sem sentimento, pesos fixos no RouteEvaluator
        user_prefs_dict = {
            "max_time": preferences.max_time,
            "max_cost": preferences.max_cost,
            "preferred_categories": preferences.preferred_categories,
            "category_weights": preferences.category_weights,
            "start_location": (optimizer_pois[0].lat, optimizer_pois[0].lon),
            "start_time": preferences.start_time,
            "center_lat": geo[0] if geo else None,
            "center_lon": geo[1] if geo else None,
            "max_radius_km": geo[2] if geo else 30.0,
        }

        evaluator = RouteEvaluator(optimizer_pois, sub_distance_matrix, user_prefs_dict)

        # ========== PASSO 5: OTIMIZAÇÃO ==========
        if verbose:
            print(f"⚙️  [OPTIMIZER-{selected_algo}] Otimizando rota...\n")

        if selected_algo == "ACO":
            optimizer = TourismACO(optimizer_pois, sub_distance_matrix, evaluator,
                                   n_ants=30, n_iterations=100)
        elif selected_algo == "GA":
            optimizer = TourismGA(optimizer_pois, sub_distance_matrix, evaluator,
                                  population_size=50, n_generations=30)
        elif selected_algo == "PSO":
            optimizer = TourismPSOA(optimizer_pois, sub_distance_matrix, evaluator,
                                    n_particles=30, n_iterations=50)
        else:  # GREEDY
            optimizer = GreedyPlanner(optimizer_pois, sub_distance_matrix, evaluator)

        optimization_result = optimizer.optimize()

        if verbose:
            print(f"\n   ✓ Fitness: {optimization_result['fitness']:.2f}")
            print(f"   ✓ POIs selecionados: {len(optimization_result['route'])}\n")

        # ========== PASSO 6: ANÁLISE SHAP ==========
        shap_explanation = None
        if use_shap and optimization_result['route']:
            if verbose:
                print("📊 [SHAP] Gerando análise interpretável...\n")
            try:
                explainer = RouteExplainer(optimizer_pois, evaluator)
                shap_explanation = explainer.explain_route(optimization_result['route'])
                if verbose:
                    print(shap_explanation['explanation'])
                    print()
            except Exception as e:
                print(f"⚠️ Erro SHAP: {e}\n")

        # ========== PASSO 7: EXPLICAÇÃO LLM ==========
        if verbose:
            print("📖 [LLM] Gerando explicação em português...\n")

        route_dicts = [
            {'name': p.name, 'category': p.category, 'cost': p.cost, 'duration': p.duration}
            for p in optimization_result['pois']
        ]

        explanation = self.llm.explain_route(
            route=route_dicts,
            preferences=preferences,
            algorithm_used=selected_algo,
            optimization_metadata=optimization_result
        )

        route_pois_list = [
            {"id": p.id, "name": p.name, "category": p.category,
             "lat": p.lat, "lon": p.lon, "duration": p.duration, "cost": p.cost}
            for p in optimization_result['pois']
        ]

        visit_time = sum(p['duration'] for p in route_pois_list)
        total_time_with_travel = evaluator._calculate_time(optimization_result['route'])
        travel_time = total_time_with_travel - visit_time

        # ========== RESULTADO FINAL ==========
        result = {
            "query": user_query,
            "preferences": {
                "max_time": preferences.max_time,
                "max_cost": preferences.max_cost,
                "categories": preferences.preferred_categories,
                "interests": preferences.interests
            },
            "algorithm_used": selected_algo,
            "route": route_pois_list,
            "optimization": {
                "fitness": optimization_result['fitness'],
                "n_candidates": len(candidate_pois),
                "n_selected": len(optimization_result['route']),
                "visit_time_min": visit_time,
                "travel_time_min": travel_time,
                "total_time_min": total_time_with_travel,
                "fitness_history": optimization_result.get('fitness_history', [])
            },
            "shap_explanation": shap_explanation,
            "explanation": explanation
        }

        # ========== PASSO 8: GERAR MAPA OSRM ==========
        if verbose:
            print("🗺️  [MAP] Gerando mapa interativo com OSRM...\n")

        try:
            from src.utils.map_generator import RouteMapGenerator
            map_gen = RouteMapGenerator()
            map_path = map_gen.generate_map(
                result['route'],
                output_file=None,
                algorithm=selected_algo,
                transport_mode=preferences.transport_mode
            )   
            if map_path:
                result['map_file'] = map_path
                if verbose:
                    print(f"✅ Mapa disponível em: {map_path}")
                    print(f"   Abre no browser: file:///{Path(map_path).absolute()}\n")
        except ImportError as e:
            print(f"⚠️ Módulos de mapa não instalados: {e}")
            print(f"   Execute: pip install folium requests polyline\n")
        except Exception as e:
            print(f"⚠️ Erro ao gerar mapa: {e}\n")

        # ========== PASSO 9: PLANEAR DIAS ==========
        if verbose:
            print("📅 [PLANNER] Organizando rota por dias...\n")

        try:
            from src.utils.day_planner import DayPlanner

            total_days = max(1, int(np.ceil(preferences.max_time / 480)))

            planner = DayPlanner(
                hours_per_day=8,
                start_time=preferences.start_time,
                lunch_break=60
            )
            if geo:
                planner.start_lat = start_geo[0]
                planner.start_lon = start_geo[1]

            day_plan = planner.plan_days(
                result['route'],
                distance_matrix=sub_distance_matrix,
                total_days=total_days
            )

            result['day_plan'] = day_plan

            if verbose:
                planner.print_itinerary(day_plan)

        except Exception as e:
            print(f"⚠️ Erro ao planear dias: {e}\n")
            import traceback
            traceback.print_exc()

        if verbose:
            self._print_result(result)

        return result

    def _print_result(self, result: Dict):
        """Imprime resultado formatado no terminal"""

        if "error" in result:
            print(f"\n❌ Erro: {result['error']}")
            return

        print(f"{'=' * 70}")
        print("✅ ROTA FINAL")
        print(f"{'=' * 70}\n")

        total_cost = 0
        total_duration = 0

        for i, poi_dict in enumerate(result['route'], 1):
            total_cost += poi_dict['cost']
            total_duration += poi_dict['duration']
            print(f"{i}. {poi_dict['name']} ({poi_dict['category']})")
            print(f"   └─ {poi_dict['duration']} min | €{poi_dict['cost']:.2f}")

        opt = result.get('optimization', {})
        visit_time = opt.get('visit_time_min', total_duration)
        travel_time = opt.get('travel_time_min', 0)
        total_time = opt.get('total_time_min', visit_time + travel_time)

        print(f"\n💰 Custo Total: €{total_cost:.2f}")
        print(f"⏱️  Tempo de Visitas: {visit_time} min ({visit_time / 60:.1f}h)")
        print(f"🚶 Tempo de Deslocações: {travel_time:.0f} min ({travel_time / 60:.1f}h)")
        print(f"⏰ Tempo Total: {total_time:.0f} min ({total_time / 60:.1f}h)")

        print(f"\n{'=' * 70}")
        print("💬 EXPLICAÇÃO LLM")
        print(f"{'=' * 70}\n")
        print(result['explanation'])
        print(f"\n{'=' * 70}\n")


# ========== EXECUTAR ==========
if __name__ == "__main__":
    try:
        system = TourismRouteSystem()

        result = system.plan_route(
            "quero visitar museus e comer bem, tenho 5 horas e 50 euros",
            use_shap=True,
            force_algorithm=None
        )

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / 'route_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("\n✅ Resultado guardado em: outputs/route_result.json\n")

    except Exception as e:
        print(f"\n❌ ERRO: {e}\n")
        import traceback
        traceback.print_exc()