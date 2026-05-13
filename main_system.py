# main_system.py
# VERSAO BASELINE:
#    - Groq (llama-3.1-8b-instant)
#    - Sem analise de sentimento
#    - Sem AHP - pesos fixos no RouteEvaluator
#    - Sem login/autenticacao
#    - Sem filtro de historico de visitas
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

ACCOMMODATION_BUNDLES = [
    "hotelaria", "alojamento_local", "turismo_habitacao",
    "turismo_espaco_rural", "apartamento_turistico",
    "pousadas_da_juventude", "aldeamento_turistico", "parques_de_campismo",
]

DURATION_RANGES = {
    # Alojamento: apenas check-in, nao conta como visita
    "hotelaria":             (30, 30),
    "alojamento_local":      (30, 30),
    "turismo_habitacao":     (30, 30),
    "turismo_espaco_rural":  (30, 30),
    "apartamento_turistico": (30, 30),
    "pousadas_da_juventude": (30, 30),
    "aldeamento_turistico":  (30, 30),
    "parques_de_campismo":   (30, 30),
    "restaurantes_e_cafes":   (45,  90),
    "monumentos":             (15,  60),
    "turismo_activo":         (90, 300),
    "praias":                 (60, 240),
    "bares_e_discotecas":     (90, 240),
    "museus_e_palacios":      (60, 150),
    "eventos":                (60, 180),
    "campos":                (120, 300),
    "arqueologia":            (30,  90),
    "espacos_verdes":         (30, 180),
    "marinas_e_portos":       (20,  90),
    "termas":                 (90, 180),
    "parques_e_reservas":     (60, 240),
    "parques_de_diversao":   (180, 360),
    "zoos_e_aquarios":       (120, 240),
    "ciencia_e_conhecimento": (60, 120),
    "casinos":                (60, 240),
    "talassoterapia":         (90, 180),
    "grutas":                 (30,  75),
    "academias":              (60, 120),
    "barragens":              (20,  60),
}

def _within_radius(poi_lat: float, poi_lon: float,
                   center_lat: float, center_lon: float,
                   radius_km: float) -> bool:
    """Verifica se um POI esta dentro do raio a partir do centro."""
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
    Sistema baseline: RAG + LLM (Groq) + Otimizacao + SHAP + Mapa OSRM + Day Planning

    Pipeline: LLM -> RAG -> Otimizacao -> SHAP -> Explicacao LLM -> Mapa OSRM -> Day Planning
    """

    def __init__(self, api_key: str = None):
        print("Inicializando sistema...\n")

        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            print("\n" + "=" * 70)
            print("[ERRO] GROQ_API_KEY nao configurada!")
            print("=" * 70)
            print("\nPara configurar:")
            print("\n1. Criar API Key Groq:")
            print("   - Vai a: https://console.groq.com")
            print("   - Clica 'API Keys' -> 'Create API Key'")
            print("   - COPIA a key (gsk_...)")
            print("\n2. Configurar no Projeto:")
            print("   - Cria/edita ficheiro .env na raiz do projeto")
            print("   - Adiciona: GROQ_API_KEY=gsk_TuaKeyAqui")
            print("   - Guarda o ficheiro")
            print("\n3. Reiniciar Terminal e executar novamente\n")
            print("=" * 70 + "\n")
            raise ValueError("GROQ_API_KEY nao configurada!")

        print(f"[OK] Usando Groq")
        print(f"[OK] API Key: {self.api_key[:20]}...{self.api_key[-10:]}\n")

        try:
            self.llm = LlamaOrchestrator(api_key=self.api_key)
        except Exception as e:
            print(f"\n[ERRO] Erro ao conectar com Groq: {e}\n")
            raise

        print("Carregando RAG...")
        self.rag = POI_RAG(data_file="data/portugal_todos_pois_final_enriched.json")

        print("Carregando dados...")
        self.all_pois_data = load_pois_from_json("data/portugal_todos_pois_final_enriched.json")

        print("Carregando resolver geografico...")
        self.location_resolver = LocationResolver()

        self.transit_service = None
        try:
            from src.transit.transit_service import TransitService
            ts = TransitService()
            ts.load(use_cache=True)
            self.transit_service = ts
            print("TransitService carregado!\n")
        except Exception as e:
            print(f"AVISO: TransitService nao disponivel: {e}\n")

        print("[OK] Sistema pronto!\n")

    def plan_route(self,
               user_query: str,
               use_shap: bool = True,
               verbose: bool = True,
               force_algorithm: str = None) -> Dict:
        """
        Pipeline completo: LLM -> RAG -> Otimizacao -> SHAP -> Explicacao LLM -> Mapa -> Day Planning

        Args:
            user_query:      Query em linguagem natural (PT ou EN)
            use_shap:        Se True, gera analise SHAP
            verbose:         Se True, imprime progresso
            force_algorithm: "ACO", "GA", "PSO", "GREEDY", ou None (LLM escolhe)

        Returns:
            Dict com rota, metricas, explicacoes SHAP, LLM, mapa e planeamento por dias
        """

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Query: {user_query}")
            print(f"{'=' * 70}\n")

        # ========== PASSO 1: LLM EXTRAI PREFERENCIAS ==========
        if verbose:
            print("[LLM] Extraindo preferencias...")

        preferences = self.llm.extract_preferences(user_query)

        mode_labels = {
            "foot": "A pe",
            "car": "Carro",
            "public_transport": "Transportes publicos",
            "fastest": "Mais rapido por segmento"
        }

        if verbose:
            label = mode_labels.get(preferences.transport_mode, preferences.transport_mode)
            print(f"   [OK] Tempo: {preferences.max_time} min")
            print(f"   [OK] Orcamento: EUR{preferences.max_cost}")
            print(f"   [OK] Categorias: {preferences.preferred_categories}")
            print(f"   [OK] Interesses: {preferences.interests}")
            print(f"   [OK] Transporte: {label}\n")

        # -- Verificar campos em falta ---------------------------------
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

        # -- Resolucao geografica --------------------------------------
        geo = None
        if preferences.location:
            geo = self.location_resolver.resolve(preferences.location)
            if geo and verbose:
                lat_c, lon_c, radius_c = geo
                print(f"   '{preferences.location}' -> "
                      f"({lat_c:.4f}, {lon_c:.4f}), raio {radius_c:.0f}km\n")
        
        # -- Resolucao do ponto de partida ---
        start_geo = None
        if hasattr(preferences, 'start_location') and preferences.start_location:
            start_geo = self.location_resolver.resolve(preferences.start_location)
            if start_geo and verbose:
                print(f"   Ponto de partida: '{preferences.start_location}' -> ({start_geo[0]:.4f}, {start_geo[1]:.4f})\n")
        if not start_geo and geo:
            start_geo = geo  # fallback: centro da cidade
                
        # ========== PASSO 2: RAG BUSCA POIs ==========
        if verbose:
            print("[RAG] Recuperando POIs relevantes...")

        rag_query = self.llm.generate_rag_query(preferences, user_history=None)

        if verbose:
            print(f"   Query texto: '{rag_query}'")
            print(f"   Category filter: {preferences.preferred_categories}")
            print(f"   Max cost: {preferences.max_cost}\n")

        # Resolucao do destino final (rota A->B)
        end_geo = None
        end_location = getattr(preferences, 'end_location', None)
        if end_location:
            end_geo = self.location_resolver.resolve(end_location)
            if end_geo and verbose:
                print(f"   Destino final '{end_location}' -> ({end_geo[0]:.4f}, {end_geo[1]:.4f})\n")

        lat_min = lat_max = lon_min = lon_max = None
        # Corredor so activo se A e B sao locais diferentes (distancia > 5km)
        _same_location = (
            geo is not None and end_geo is not None and
            abs(geo[0] - end_geo[0]) < 0.05 and abs(geo[1] - end_geo[1]) < 0.05
        )
        is_corridor = geo is not None and end_geo is not None and not _same_location
        CORRIDOR_BUFFER_DEG = 0.45  # ~50km lateral buffer

        if is_corridor:
            lat_a, lon_a, _ = geo
            lat_b, lon_b, _ = end_geo
            lat_min = min(lat_a, lat_b) - CORRIDOR_BUFFER_DEG
            lat_max = max(lat_a, lat_b) + CORRIDOR_BUFFER_DEG
            lon_min = min(lon_a, lon_b) - CORRIDOR_BUFFER_DEG
            lon_max = max(lon_a, lon_b) + CORRIDOR_BUFFER_DEG
            if verbose:
                print(f"   Modo corredor A->B: bbox ({lat_min:.2f},{lon_min:.2f}) -> ({lat_max:.2f},{lon_max:.2f})\n")
        elif geo:
            center_lat, center_lon, radius_km = geo
            delta = radius_km / 111.0
            lat_min = center_lat - delta
            lat_max = center_lat + delta
            lon_min = center_lon - delta
            lon_max = center_lon + delta

        EXCLUDED_CATEGORIES = ["eventos"]
        rag_results = self.rag.query(
            text=rag_query,
            n_results=60,
            category_filter=preferences.preferred_categories,
            category_exclude=EXCLUDED_CATEGORIES,
            max_cost=preferences.max_cost,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        )
        candidate_pois = rag_results['pois']

        # Garantir representacao minima de cada categoria pedida
        if preferences.preferred_categories and len(preferences.preferred_categories) > 1:
            from collections import defaultdict
            by_cat = defaultdict(list)
            for p in candidate_pois:
                by_cat[p['category']].append(p)
            min_per_cat = max(3, 25 // len(preferences.preferred_categories))
            existing_ids = {p['id'] for p in candidate_pois}
            for cat in preferences.preferred_categories:
                if len(by_cat[cat]) < min_per_cat:
                    extra = self.rag.query(
                        text=cat,
                        n_results=min_per_cat * 2,
                        category_filter=[cat],
                        category_exclude=EXCLUDED_CATEGORIES,
                        max_cost=preferences.max_cost,
                        lat_min=lat_min, lat_max=lat_max,
                        lon_min=lon_min, lon_max=lon_max
                    )
                    for p in extra['pois']:
                        if p['id'] not in existing_ids:
                            candidate_pois.append(p)
                            existing_ids.add(p['id'])
                    if verbose:
                        print(f"   Rebalance '{cat}': +{len(extra['pois'])} POIs")

        # Fallback: se POIs insuficientes para preencher 60% do tempo disponivel
        tempo_candidatos = sum(p['duration'] for p in candidate_pois)
        if tempo_candidatos < 0.6 * preferences.max_time:
            if verbose:
                print(f"   AVISO Fallback: {tempo_candidatos}min de candidatos para {preferences.max_time}min disponiveis")
                print(f"   Re-query sem filtro de categorias...")
            rag_results_fallback = self.rag.query(
                text=rag_query,
                n_results=80,
                category_filter=None,
                category_exclude=EXCLUDED_CATEGORIES,
                max_cost=preferences.max_cost,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max
            )
            # Merge: adiciona POIs do fallback que nao estao ja nos candidatos
            existing_ids = {p['id'] for p in candidate_pois}
            for p in rag_results_fallback['pois']:
                if p['id'] not in existing_ids:
                    candidate_pois.append(p)
                    existing_ids.add(p['id'])
            if verbose:
                print(f"   [OK] Candidatos apos fallback: {len(candidate_pois)}")
                
        # Forcar candidatos de alojamento (independente das categorias pedidas)
        num_days = max(1, int(np.ceil(preferences.max_time / 480)))
        accom_needed = max(5, num_days + 3)
        accom_results = self.rag.query(
            text=f"hotel alojamento {preferences.location or ''}",
            n_results=accom_needed * 2,
            category_filter=ACCOMMODATION_BUNDLES,
            max_cost=preferences.max_cost,
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
        )
        existing_ids = {p['id'] for p in candidate_pois}
        accom_added = 0
        for p in accom_results['pois']:
            if p['id'] not in existing_ids:
                candidate_pois.append(p)
                existing_ids.add(p['id'])
                accom_added += 1
        if verbose:
            print(f"   Alojamento: +{accom_added} candidatos forcados (total {len(candidate_pois)})\n")

        # Hard filter geografico pos-RAG
        if is_corridor:
            # Modo corredor: manter POIs dentro de MAX_DIST_KM da linha A->B
            MAX_DIST_KM = 55.0
            lat_a, lon_a, _ = geo
            lat_b, lon_b, _ = end_geo
            from src.utils.distance_calculator import haversine as _hav

            def _dist_to_segment(plat, plon):
                abx, aby = lon_b - lon_a, lat_b - lat_a
                apx, apy = plon - lon_a, plat - lat_a
                denom = abx**2 + aby**2
                t = max(0.0, min(1.0, (apx*abx + apy*aby) / denom)) if denom > 1e-10 else 0.0
                nlat = lat_a + t * aby
                nlon = lon_a + t * abx
                return _hav(plat, plon, nlat, nlon)

            before = len(candidate_pois)
            candidate_pois = [p for p in candidate_pois
                              if _dist_to_segment(p['lat'], p['lon']) <= MAX_DIST_KM]
            if verbose:
                print(f"   Filtro corredor: {before} -> {len(candidate_pois)} POIs (max {MAX_DIST_KM:.0f}km da linha)\n")

            # Density boost: reponderar por densidade local (POIs num raio de 20km)
            if candidate_pois:
                DENSITY_RADIUS_KM = 20.0
                DENSITY_WEIGHT    = 0.25
                for poi in candidate_pois:
                    nearby = sum(1 for other in candidate_pois
                                 if _hav(poi['lat'], poi['lon'], other['lat'], other['lon']) <= DENSITY_RADIUS_KM)
                    poi['_density'] = nearby
                max_density = max(p['_density'] for p in candidate_pois) or 1
                for poi in candidate_pois:
                    poi['relevance_score'] = poi.get('relevance_score', 0.5) * (
                        1 + DENSITY_WEIGHT * poi['_density'] / max_density)
                candidate_pois.sort(key=lambda p: -p.get('relevance_score', 0))
                if verbose:
                    print(f"   Density boost aplicado (raio {DENSITY_RADIUS_KM:.0f}km, peso {DENSITY_WEIGHT})\n")

        elif geo:
            center_lat, center_lon, radius_km = geo
            before = len(candidate_pois)
            candidate_pois = [
                p for p in candidate_pois
                if _within_radius(p['lat'], p['lon'], center_lat, center_lon, radius_km)
            ]
            if verbose:
                print(f"   Hard filter geografico: {before} -> {len(candidate_pois)} POIs (raio {radius_km:.0f}km)\n")

            if len(candidate_pois) == 0:
                if verbose:
                    print("   AVISO: Hard filter removeu todos os POIs - a relaxar para bounding box\n")
                rag_results_nogeo = self.rag.query(
                    text=rag_query,
                    n_results=25,
                    category_filter=preferences.preferred_categories,
                    category_exclude=EXCLUDED_CATEGORIES,
                    max_cost=preferences.max_cost,
                )
                candidate_pois = rag_results_nogeo['pois']

        # Aplicar intervalos de duracao por categoria
        for p in candidate_pois:
            cat = p.get("category", "")
            if cat in DURATION_RANGES:
                d_min, d_max = DURATION_RANGES[cat]
                p["duration"] = max(d_min, min(d_max, p["duration"]))

        if len(candidate_pois) == 0:
            print(f"\n{'=' * 70}")
            print("AVISO: RAG RETORNOU 0 POIs!")
            print(f"{'=' * 70}")
            print("Possiveis causas:")
            print("  1. Categorias extraidas nao existem no JSON")
            print("  2. ChromaDB precisa ser reconstruido -- apaga data/chroma_db")
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
            print(f"   [OK] Query RAG: '{rag_query}'\n")

        # ========== PASSO 3: SELECIONAR ALGORITMO ==========
        if verbose:
            print("[LLM] Selecionando algoritmo...")

        if force_algorithm:
            selected_algo = force_algorithm.upper()
            if selected_algo not in ["ACO", "GA", "PSO", "GREEDY"]:
                print(f"AVISO: Algoritmo invalido '{selected_algo}', usando ACO")
                selected_algo = "ACO"
            if verbose:
                print(f"   [OK] Algoritmo (FORCADO): {selected_algo}\n")
        else:
            selected_algo = select_algorithm_deterministic(len(candidate_pois), preferences.max_time)
            if verbose:
                print(f"   [OK] Algoritmo (deterministico): {selected_algo}\n")

        # ========== PASSO 4: PREPARAR OTIMIZACAO ==========
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
            print("\nAVISO: Nenhum POI para otimizar!")
            return {"error": "NO_OPTIMIZER_POIS", "query": user_query, "n_candidates": 0}

        n_pois = len(optimizer_pois)

        if verbose:
            print(f"   [OK] POIs para otimizacao: {n_pois}\n")

        # Sub-matriz de tempos reais (TransitService) ou Haversine (fallback)
        if self.transit_service is not None:
            if verbose:
                print(f"   A construir matriz de tempos reais ({preferences.transport_mode})...\n")
            sub_distance_matrix = self.transit_service.build_cost_matrix(
                optimizer_pois,
                mode=preferences.transport_mode
            )
        else:
            from src.utils.distance_calculator import haversine
            # Tabela estatica de tempos de viagem (minutos) por distancia e modo
            # Formato: [(max_km, minutos), ...] - ultimo entry e o fallback
            _TIME_TABLE = {
                "foot":             [(1, 12), (2, 25), (5, 60), (float('inf'), 999)],
                "car":              [(2, 4),  (5, 8),  (15, 15), (50, 38), (float('inf'), 75)],
                "public_transport": [(1, 10), (2, 17), (5, 30), (15, 47), (50, 94), (float('inf'), 153)],
                "fastest":          [(2, 4),  (5, 8),  (15, 15), (50, 38), (float('inf'), 75)],
            }
            def _travel_time(d_km, mode):
                table = _TIME_TABLE.get(mode, _TIME_TABLE["public_transport"])
                for max_km, t_min in table:
                    if d_km <= max_km:
                        return float(t_min)
                return 240.0

            if verbose:
                print(f"   Tabela estatica de tempos ({preferences.transport_mode})\n")
            sub_distance_matrix = np.zeros((n_pois, n_pois))
            for i, poi_i in enumerate(optimizer_pois):
                for j, poi_j in enumerate(optimizer_pois):
                    if i != j:
                        d_km = haversine(poi_i.lat, poi_i.lon, poi_j.lat, poi_j.lon)
                        sub_distance_matrix[i][j] = _travel_time(d_km, preferences.transport_mode)

        # Preferencias - sem sentimento, pesos fixos no RouteEvaluator
        mobility_issues = getattr(preferences, 'mobility_issues', False) or False
        has_children    = getattr(preferences, 'has_children', False) or False
        if has_children:
            print("   has_children=True -> modificador contextual activado\n")

        # Matriz de elevacao - so calculada se mobilidade reduzida
        elevation_matrix = None
        if mobility_issues:
            if verbose:
                print("   A calcular perfis de elevacao (mobilidade reduzida)...\n")
            elevation_matrix = self._build_elevation_matrix(optimizer_pois, verbose)

        _num_people = getattr(preferences, "num_people", 1)
        _num_rooms  = getattr(preferences, "num_rooms",  max(1, math.ceil(_num_people / 2)))

        user_prefs_dict = {
            "max_time":   preferences.max_time,
            "max_cost":   preferences.max_cost,
            "num_people": _num_people,
            "num_rooms":  _num_rooms,
            "preferred_categories": preferences.preferred_categories,
            "category_weights": preferences.category_weights,
            "start_location": (optimizer_pois[0].lat, optimizer_pois[0].lon),
            "start_time": preferences.start_time,
            "center_lat": geo[0] if geo else None,
            "center_lon": geo[1] if geo else None,
            "max_radius_km": geo[2] if geo else 30.0,
            "mobility_issues": mobility_issues,
            "has_children": has_children,
            "elevation_matrix": elevation_matrix,
        }

        evaluator = RouteEvaluator(optimizer_pois, sub_distance_matrix, user_prefs_dict)

        # ========== PASSO 5: OTIMIZACAO ==========
        if verbose:
            print(f"[OPTIMIZER-{selected_algo}] Otimizando rota...\n")

        if selected_algo == "ACO":
            optimizer = TourismACO(optimizer_pois, sub_distance_matrix, evaluator,
                                   n_ants=30, n_iterations=100)
        elif selected_algo == "GA":
            optimizer = TourismGA(optimizer_pois, sub_distance_matrix, evaluator,
                                  population_size=80, n_generations=50)
        elif selected_algo == "PSO":
            optimizer = TourismPSOA(optimizer_pois, sub_distance_matrix, evaluator,
                                    n_particles=30, n_iterations=50)
        else:  # GREEDY
            optimizer = GreedyPlanner(optimizer_pois, sub_distance_matrix, evaluator)

        optimization_result = optimizer.optimize()

        if verbose:
            print(f"\n   [OK] Fitness: {optimization_result['fitness']:.2f}")
            print(f"   [OK] POIs selecionados: {len(optimization_result['route'])}\n")

        # ========== PASSO 6: ANALISE SHAP ==========
        shap_explanation = None
        if use_shap and optimization_result['route']:
            if verbose:
                print("[SHAP] Gerando analise interpretavel...\n")
            try:
                explainer = RouteExplainer(optimizer_pois, evaluator)
                shap_explanation = explainer.explain_route(optimization_result['route'])
                if verbose:
                    print(shap_explanation['explanation'])
                    print()
            except Exception as e:
                print(f"AVISO: Erro SHAP: {e}\n")

        # ========== PASSO 7: EXPLICACAO LLM ==========
        if verbose:
            print("[LLM] Gerando explicacao em portugues...\n")

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

        # ========== PASSO 8: PLANEAR DIAS ==========
        if verbose:
            print("[PLANNER] Organizando rota por dias...\n")

        day_plan = None
        try:
            from src.utils.day_planner import DayPlanner

            requested_days = max(1, int(np.ceil(preferences.max_time / 480)))
            actual_route_time = sum(p['duration'] for p in result['route'])
            actual_days_needed = max(1, int(np.ceil(actual_route_time / 480)))
            total_days = min(requested_days, actual_days_needed)

            planner = DayPlanner(
                hours_per_day=8,
                start_time="09:00",
                lunch_break=60
            )
            if start_geo:
                planner.start_lat = start_geo[0]
                planner.start_lon = start_geo[1]

            first_day_start = preferences.start_time if preferences.start_time != "09:00" else None
            last_day_end = getattr(preferences, 'last_day_end_time', None)

            day_plan = planner.plan_days(
                result['route'],
                distance_matrix=sub_distance_matrix,
                total_days=total_days,
                first_day_start_time=first_day_start,
                last_day_end_time=last_day_end,
            )

            result['day_plan'] = day_plan

            if verbose:
                planner.print_itinerary(day_plan)

        except Exception as e:
            print(f"AVISO: Erro ao planear dias: {e}\n")
            import traceback
            traceback.print_exc()

        # ========== PASSO 9: GERAR MAPA OSRM ==========
        if verbose:
            print("[MAP] Gerando mapa interativo com OSRM...\n")

        try:
            from src.utils.map_generator import RouteMapGenerator
            map_gen = RouteMapGenerator()
            map_path = map_gen.generate_map(
                result['route'],
                output_file=None,
                algorithm=selected_algo,
                transport_mode=preferences.transport_mode,
                day_plan=day_plan,
            )
            if map_path:
                result['map_file'] = map_path
                if verbose:
                    print(f"[OK] Mapa disponivel em: {map_path}")
                    print(f"   Abre no browser: file:///{Path(map_path).absolute()}\n")
        except ImportError as e:
            print(f"AVISO: Modulos de mapa nao instalados: {e}")
            print(f"   Execute: pip install folium requests polyline\n")
        except Exception as e:
            print(f"AVISO: Erro ao gerar mapa: {e}\n")

        if verbose:
            self._print_result(result)

        return result

    def _build_elevation_matrix(self, pois, verbose=False) -> np.ndarray:
        """
        Constroi matriz KxK com ganho de elevacao acumulado (em metros)
        entre cada par de POIs, usando OpenTopoData SRTM30m.
        """
        import requests, math

        def haversine_m(lat1, lon1, lat2, lon2):
            R = 6371000
            r = math.radians
            a = math.sin(r(lat2-lat1)/2)**2 + math.cos(r(lat1))*math.cos(r(lat2))*math.sin(r(lon2-lon1)/2)**2
            return R * 2 * math.asin(math.sqrt(a))

        def elevation_gain(lat1, lon1, lat2, lon2):
            dist_m = haversine_m(lat1, lon1, lat2, lon2)
            sample_m = max(50, int(dist_m / 99))
            n = max(2, min(100, int(dist_m / sample_m) + 1))
            points = [
                (lat1 + i*(lat2-lat1)/(n-1), lon1 + i*(lon2-lon1)/(n-1))
                for i in range(n)
            ]
            locations = "|".join(f"{lat},{lon}" for lat, lon in points)
            try:
                r = requests.get(
                    f"https://api.opentopodata.org/v1/srtm30m?locations={locations}",
                    timeout=10
                )
                elevs = [e["elevation"] for e in r.json().get("results", [])
                         if e.get("elevation") is not None]
                gain = sum(max(0, elevs[i+1]-elevs[i]) for i in range(len(elevs)-1))
                return gain
            except Exception:
                return 0.0

        n = len(pois)
        matrix = np.zeros((n, n))
        pairs_done = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = elevation_gain(
                        pois[i].lat, pois[i].lon,
                        pois[j].lat, pois[j].lon
                    )
                    pairs_done += 1
                    if verbose and pairs_done % 10 == 0:
                        print(f"   Elevacao: {pairs_done}/{n*(n-1)} pares calculados...")
        return matrix

    def _print_result(self, result: Dict):
        """Imprime resultado formatado no terminal"""

        if "error" in result:
            print(f"\n[ERRO] Erro: {result['error']}")
            return

        print(f"{'=' * 70}")
        print("[OK] ROTA FINAL")
        print(f"{'=' * 70}\n")

        total_cost = 0
        total_duration = 0

        for i, poi_dict in enumerate(result['route'], 1):
            total_cost += poi_dict['cost']
            total_duration += poi_dict['duration']
            print(f"{i}. {poi_dict['name']} ({poi_dict['category']})")
            print(f"   -- {poi_dict['duration']} min | EUR{poi_dict['cost']:.2f}")

        opt = result.get('optimization', {})
        visit_time = opt.get('visit_time_min', total_duration)
        travel_time = opt.get('travel_time_min', 0)
        total_time = opt.get('total_time_min', visit_time + travel_time)

        print(f"\nCusto Total: EUR{total_cost:.2f}")
        print(f"Tempo de Visitas: {visit_time} min ({visit_time / 60:.1f}h)")
        print(f"Tempo de Deslocacoes: {travel_time:.0f} min ({travel_time / 60:.1f}h)")
        print(f"Tempo Total: {total_time:.0f} min ({total_time / 60:.1f}h)")

        print(f"\n{'=' * 70}")
        print("EXPLICACAO LLM")
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

        print("\n[OK] Resultado guardado em: outputs/route_result.json\n")

    except Exception as e:
        print(f"\n[ERRO]: {e}\n")
        import traceback
        traceback.print_exc()