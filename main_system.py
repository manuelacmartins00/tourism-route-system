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
import shutil
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

POI_DATA_FILE = "data/portugal_todos_pois_final_enriched.json"


def _ensure_poi_data_file(path: str = POI_DATA_FILE):
    """Descarrega o dataset de POIs do dataset privado da HF se nao existir localmente."""
    if os.path.exists(path):
        return
    from huggingface_hub import hf_hub_download
    print(f"'{path}' nao encontrado - a descarregar do dataset privado HF...")
    downloaded = hf_hub_download(
        repo_id="ManuelMartinsTeseISCTE/tourism-pois-data",
        filename=os.path.basename(path),
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shutil.copy(downloaded, path)
    print(f"'{path}' descarregado.\n")

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

# Grupos tematicos (B1): quando o Fill-D nao encontra POIs novos nas
# preferred_categories (ex: ja esgotadas pelo cap=2), procura categorias
# "irmas" do mesmo tema antes de deixar o dia incompleto.
THEMATIC_GROUPS = [
    {"monumentos", "museus_e_palacios", "arqueologia", "ciencia_e_conhecimento", "grutas"},
    {"praias", "parques_e_reservas", "espacos_verdes", "turismo_activo", "barragens", "grutas", "campos"},
    {"parques_de_diversao", "zoos_e_aquarios", "marinas_e_portos", "termas", "talassoterapia", "academias"},
    {"bares_e_discotecas", "casinos", "eventos"},
]


def _thematic_siblings(categories: list) -> list:
    """Devolve categorias do mesmo grupo tematico que `categories`, excluindo
    as proprias `categories` (B1 fallback de Fill-D)."""
    cats = set(categories or [])
    siblings: set = set()
    for group in THEMATIC_GROUPS:
        if group & cats:
            siblings |= group
    siblings -= cats
    return list(siblings)


def _trip_spans_meal_window(start_time: str, max_time_min: int) -> bool:
    """True se a rota sobrepoe uma janela de refeicao (almoco 12-14h ou jantar 19-22h)."""
    try:
        h, m = map(int, (start_time or "09:00").split(":"))
        start_min = h * 60 + m
    except Exception:
        start_min = 9 * 60
    end_min = start_min + max_time_min
    if end_min >= 24 * 60:
        return True
    WINDOWS = [(12 * 60, 14 * 60), (19 * 60, 22 * 60)]
    return any(start_min < w_end and end_min > w_start for w_start, w_end in WINDOWS)


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

        _ensure_poi_data_file()

        print("Carregando RAG...")
        self.rag = POI_RAG(data_file=POI_DATA_FILE)

        print("Carregando dados...")
        self.all_pois_data = load_pois_from_json(POI_DATA_FILE)

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

    def _resolve_location(self, location: str, verbose: bool = False):
        """B3: resolve um toponimo, com fallback ao LLM para termos vagos.

        Se location_resolver.resolve() nao encontrar nada (GeoJSON, overrides
        regionais e Nominatim ja falharam), pede ao LLM para associar o termo
        a uma das regioes canonicas conhecidas (_COUNTRY_REGION_OVERRIDES).
        Restringido a essa lista fechada para evitar coordenadas inventadas —
        e melhor uma regiao aproximada do que nenhum filtro geografico.
        """
        geo = self.location_resolver.resolve(location)
        if geo:
            return geo

        canonical = sorted(self.location_resolver._COUNTRY_REGION_OVERRIDES.keys())
        prompt = (
            f"Termo de localizacao em Portugal: '{location}'.\n"
            f"Qual destas regioes melhor o descreve? Responde APENAS com o "
            f"termo exacto da lista (em minusculas, sem acentos), ou 'nenhuma' "
            f"se nao houver correspondencia razoavel.\n"
            f"Lista: {', '.join(canonical)}"
        )
        try:
            resp = self.llm._call_llm(prompt, max_tokens=15, temperature=0)
        except Exception as e:
            if verbose:
                print(f"   AVISO: [LocationResolver] fallback LLM falhou para '{location}': {e}")
            return None

        candidate = resp.strip().lower().strip(".\"'")
        if candidate in self.location_resolver._COUNTRY_REGION_OVERRIDES:
            if verbose:
                print(f"   [OK] [LocationResolver] '{location}' -> fallback LLM -> '{candidate}'")
            return self.location_resolver.resolve(candidate)

        if verbose:
            print(f"   AVISO: [LocationResolver] fallback LLM nao encontrou correspondencia para '{location}' (resposta: '{resp.strip()}')")
        return None

    def plan_route(self,
               user_query: str,
               use_shap: bool = True,
               verbose: bool = True,
               force_algorithm: str = None,
               include_accommodation: Optional[bool] = None,
               include_meals: Optional[bool] = None,
               generate_map: bool = True,
               num_rooms: Optional[int] = None,
               generate_explanation: bool = True,
               compact_extraction: bool = False,
               fixture_capture_path: str = None,
               direct_preferences: Dict = None,
               disable_algo_fallback: bool = False) -> Dict:
        """
        Pipeline completo: LLM -> RAG -> Otimizacao -> SHAP -> Explicacao LLM -> Mapa -> Day Planning

        Args:
            user_query:             Query em linguagem natural (PT ou EN)
            use_shap:               Se True, gera analise SHAP
            verbose:                Se True, imprime progresso
            force_algorithm:        "ACO", "GA", "PSO", "GREEDY", ou None (LLM escolhe)
            include_accommodation:  True/False=resposta do user; None=pergunta automatica se relevante
            include_meals:          True/False=resposta do user; None=pergunta automatica se relevante

        Returns:
            Dict com rota, metricas, explicacoes SHAP, LLM, mapa e planeamento por dias
        """

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Query: {user_query}")
            print(f"{'=' * 70}\n")

        # ========== PASSO 1: LLM EXTRAI PREFERENCIAS (ou usa directo) ==========
        if direct_preferences is not None:
            from src.llm.llm_orchestrator import UserPreferences as _UP
            preferences = _UP(**{k: v for k, v in direct_preferences.items()
                                 if k in _UP.__dataclass_fields__})
        else:
            if verbose:
                print("[LLM] Extraindo preferencias...")
            preferences = self.llm.extract_preferences(user_query, compact=compact_extraction)

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

        # Detectar vida noturna nas preferencias
        NIGHTLIFE_CATS = {"bares_e_discotecas", "casinos"}
        has_nightlife = any(c in NIGHTLIFE_CATS for c in (preferences.preferred_categories or []))

        # Se ha vida noturna, aumentar max_time para acomodar noites (3h/noite)
        # num_days calculado ANTES do ajuste para nao inflar max_days no evaluator
        num_days_base = max(1, round(preferences.max_time / 480)) if preferences.max_time else 1
        if has_nightlife and preferences.max_time:
            preferences.max_time += num_days_base * 180  # +3h por noite
            if verbose:
                print(f"   [OK] max_time ajustado para vida noturna: {preferences.max_time} min ({num_days_base} noites)")

        # Filtro de seguranca: remover campos que nunca devem ser pedidos
        NEVER_ASK = {"num_rooms", "mobility_issues", "start_location"}
        if preferences.missing_fields:
            preferences.missing_fields = [f for f in preferences.missing_fields
                                          if f not in NEVER_ASK]

        # max_cost e budget_type são sempre pedidos no mesmo passo
        _mf = list(preferences.missing_fields or [])
        if 'max_cost' in _mf and 'budget_type' not in _mf:
            _mf.append('budget_type')
            preferences.missing_fields = _mf

        # -- Perguntas de scope: alojamento e refeicoes ---------------
        # Verificadas ANTES dos missing_fields para aparecerem na primeira interacao.
        scope_questions = []
        if include_accommodation is None:
            if preferences.max_time and preferences.max_time > 480:
                scope_questions.append("include_accommodation")
            else:
                include_accommodation = True
        if include_meals is None:
            if _trip_spans_meal_window(preferences.start_time, preferences.max_time or 480):
                scope_questions.append("include_meals")
            else:
                include_meals = True

        # Se ha scope questions, devolver juntas com eventuais missing_fields
        if scope_questions:
            return {
                "status": "needs_scope_clarification",
                "scope_questions": scope_questions,
                "missing_fields": preferences.missing_fields or [],
                "query": user_query,
                "preferences_so_far": {
                    "max_time": preferences.max_time,
                    "max_cost": preferences.max_cost,
                    "location": preferences.location,
                    "start_time": preferences.start_time,
                    "num_people": getattr(preferences, "num_people", 1),
                },
            }

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

        if include_accommodation is None:
            include_accommodation = True
        if include_meals is None:
            include_meals = True

        # -- Resolucao geografica --------------------------------------
        geo = None
        # Raio máximo proporcional ao nº de dias: 1d→40km, 7d→100km, 14d→170km
        _total_days = max(1, (preferences.max_time or 480) // 480)
        _max_radius = 30 + _total_days * 10
        # Limite de deslocacao de POIs entre dias no day_planner: relativo ao
        # raio da regiao e ao numero de dias, para evitar zigzag oeste-este
        # em regioes grandes (ex: Algarve 3 dias -> 60/3=20km)
        _move_cap_km = max(15.0, _max_radius / _total_days)
        if preferences.location:
            geo = self._resolve_location(preferences.location, verbose=verbose)
            if geo:
                if geo[2] > _max_radius:
                    geo = (geo[0], geo[1], _max_radius)
                if verbose:
                    print(f"   '{preferences.location}' -> "
                          f"({geo[0]:.4f}, {geo[1]:.4f}), raio {geo[2]:.0f}km\n")
        
        # -- Resolucao do ponto de partida ---
        start_geo = None
        if hasattr(preferences, 'start_location') and preferences.start_location:
            start_geo = self._resolve_location(preferences.start_location, verbose=verbose)
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

        # Resolucao de todas as localidades (ate 4) e TSP se nao ordenadas
        from itertools import permutations as _perms
        from src.utils.distance_calculator import haversine as _hav_tsp

        _locations_list = getattr(preferences, 'locations', []) or []
        _locations_ordered = getattr(preferences, 'locations_ordered', False)

        # Resolver coordenadas de todas as localidades
        all_geos = []
        all_loc_names = []
        for loc_name in _locations_list:
            g = self._resolve_location(loc_name, verbose=verbose)
            if g:
                if g[2] > _max_radius:
                    g = (g[0], g[1], _max_radius)
                all_geos.append(g)
                all_loc_names.append(loc_name)

        # Se so temos geo do location principal, adicionar
        if geo and (not all_geos or all_geos[0] != geo):
            all_geos = [geo] + [g for g in all_geos if g != geo]

        # TSP: se mais de 1 local e ordem nao especificada, optimizar
        if len(all_geos) > 1 and not _locations_ordered:
            best_order = list(range(len(all_geos)))
            best_dist = float('inf')
            for perm in _perms(range(len(all_geos))):
                d = sum(_hav_tsp(all_geos[perm[i]][0], all_geos[perm[i]][1],
                                 all_geos[perm[i+1]][0], all_geos[perm[i+1]][1])
                        for i in range(len(perm)-1))
                if d < best_dist:
                    best_dist = d
                    best_order = list(perm)
            all_geos = [all_geos[i] for i in best_order]
            all_loc_names = [all_loc_names[i] for i in best_order] if all_loc_names else []
            if verbose and len(all_geos) > 1:
                print(f"   TSP ordem optima: {all_loc_names} ({best_dist:.0f}km total)\n")

        # Actualizar geo e end_geo para primeiro e ultimo
        if all_geos:
            geo = all_geos[0]
        end_geo = all_geos[-1] if len(all_geos) > 1 else None

        lat_min = lat_max = lon_min = lon_max = None
        CORRIDOR_BUFFER_DEG = 0.45  # ~50km lateral buffer

        # Corredor activo se ha 2+ locais diferentes
        _same_location = (
            geo is not None and end_geo is not None and
            abs(geo[0] - end_geo[0]) < 0.05 and abs(geo[1] - end_geo[1]) < 0.05
        )
        is_corridor = len(all_geos) > 1 and not _same_location

        if is_corridor:
            # Bounding box sobre TODAS as localidades (nao so A e D)
            all_lats = [g[0] for g in all_geos]
            all_lons = [g[1] for g in all_geos]
            lat_min = min(all_lats) - CORRIDOR_BUFFER_DEG
            lat_max = max(all_lats) + CORRIDOR_BUFFER_DEG
            lon_min = min(all_lons) - CORRIDOR_BUFFER_DEG
            lon_max = max(all_lons) + CORRIDOR_BUFFER_DEG
            if verbose:
                print(f"   Modo corredor {len(all_geos)} localidades: bbox ({lat_min:.2f},{lon_min:.2f}) -> ({lat_max:.2f},{lon_max:.2f})\n")
        elif geo:
            # B4: regioes alongadas N-S (litoral/interior) usam bbox explicito
            # em vez de um circulo (centro, raio) que ou nao cobre as pontas
            # ou inclui zonas fora da regiao
            bbox = self.location_resolver.resolve_bbox(preferences.location) if preferences.location else None
            if bbox:
                lat_min, lat_max, lon_min, lon_max = bbox
                if verbose:
                    print(f"   '{preferences.location}' -> bbox regional "
                          f"({lat_min:.2f},{lon_min:.2f}) -> ({lat_max:.2f},{lon_max:.2f})\n")
            else:
                center_lat, center_lon, radius_km = geo
                delta = radius_km / 111.0
                lat_min = center_lat - delta
                lat_max = center_lat + delta
                lon_min = center_lon - delta
                lon_max = center_lon + delta

        # Categorias operacionais/administrativas — nunca incluir em rotas turísticas
        NEVER_INCLUDE_CATEGORIES = [
            "eventos", "postos_de_turismo", "agencias_de_viagem",
            "localidade", "servicos_de_turismo", "outros", "rentacar",
        ]
        EXCLUDED_CATEGORIES = list(NEVER_INCLUDE_CATEGORIES)
        if not include_accommodation:
            EXCLUDED_CATEGORIES.extend(ACCOMMODATION_BUNDLES)
            if verbose:
                print("   [INFO] Alojamento excluido da rota (user trata autonomamente)\n")
        if not include_meals:
            EXCLUDED_CATEGORIES.append("restaurantes_e_cafes")
            if preferences.preferred_categories and "restaurantes_e_cafes" in preferences.preferred_categories:
                preferences.preferred_categories.remove("restaurantes_e_cafes")
            if verbose:
                print("   [INFO] Refeicoes excluidas da rota (user trata autonomamente)\n")

        rag_results = self.rag.query(
            text=rag_query,
            n_results=40,
            category_filter=preferences.preferred_categories,
            category_exclude=EXCLUDED_CATEGORIES,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        )
        candidate_pois = rag_results['pois']

        # Query semantica suplementar sem filtro de categoria — traz POIs contextualmente
        # relevantes de outras categorias (turismo_activo, parques, etc.) que o filtro
        # estrito excluiria, evitando rotas compostas por uma unica categoria
        rag_supplement = self.rag.query(
            text=rag_query,
            n_results=20,
            category_filter=None,
            category_exclude=EXCLUDED_CATEGORIES,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        )
        existing_ids = {p['id'] for p in candidate_pois}
        supp_added = 0
        for p in rag_supplement['pois']:
            if p['id'] not in existing_ids:
                candidate_pois.append(p)
                existing_ids.add(p['id'])
                supp_added += 1
        if verbose and supp_added:
            print(f"   Query semantica suplementar: +{supp_added} POIs de categorias variadas\n")

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
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max
            )
            existing_ids = {p['id'] for p in candidate_pois}
            for p in rag_results_fallback['pois']:
                if p['id'] not in existing_ids:
                    candidate_pois.append(p)
                    existing_ids.add(p['id'])
            if verbose:
                print(f"   [OK] Candidatos apos fallback: {len(candidate_pois)}")

        # Forcar candidatos de alojamento (independente das categorias pedidas)
        if include_accommodation:
            num_days = max(1, int(np.ceil(preferences.max_time / 480)))
            accom_needed = max(5, num_days + 3)
            accom_results = self.rag.query(
                text=f"hotel alojamento {preferences.location or ''}",
                n_results=accom_needed * 2,
                category_filter=ACCOMMODATION_BUNDLES,
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

        # Forcar candidatos de restaurantes quando include_meals=True
        # Apenas num_days restaurantes no pool do GA (1 por dia) — o resto é garantido
        # pelo [Meals-Pre] após otimização. Assim o GA foca em POIs de atividade.
        if include_meals and "restaurantes_e_cafes" not in (preferences.preferred_categories or []):
            num_days = max(1, int(np.ceil(preferences.max_time / 480)))
            meal_results = self.rag.query(
                text=f"restaurante jantar almoco {preferences.location or ''}",
                n_results=num_days,
                category_filter=["restaurantes_e_cafes"],
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
            )
            existing_ids = {p['id'] for p in candidate_pois}
            meal_added = 0
            for p in meal_results['pois']:
                if p['id'] not in existing_ids:
                    candidate_pois.append(p)
                    existing_ids.add(p['id'])
                    meal_added += 1
            if verbose:
                print(f"   Refeicoes: +{meal_added} restaurantes no pool GA (total {len(candidate_pois)})\n")

        # B4: reservar uma fatia do orcamento para refeicoes (almoco+jantar/dia)
        # ANTES de correr o GA, para que este nao gaste o max_cost todo em
        # bilhetes/atracoes e deixe o [Meals-Pre] sem margem para restaurantes.
        meal_budget_reserve = 0.0
        if include_meals and preferences.max_cost:
            MEALS_PER_DAY = 2  # almoco + jantar
            DEFAULT_MEAL_COST = 12.0
            restaurant_costs = [p.get('cost', 0) for p in candidate_pois
                                 if p.get('category') == 'restaurantes_e_cafes' and p.get('cost')]
            avg_meal_cost = (sum(restaurant_costs) / len(restaurant_costs)
                             if restaurant_costs else DEFAULT_MEAL_COST)
            meal_budget_reserve = num_days_base * MEALS_PER_DAY * avg_meal_cost
            # Nunca reservar mais de metade do orcamento (evita pools vazios em viagens curtas)
            meal_budget_reserve = min(meal_budget_reserve, preferences.max_cost * 0.5)
            if verbose:
                print(f"   [Budget] Reserva refeicoes: EUR{meal_budget_reserve:.2f} "
                      f"({num_days_base}d x {MEALS_PER_DAY} x EUR{avg_meal_cost:.2f}) "
                      f"-> orcamento GA: EUR{preferences.max_cost - meal_budget_reserve:.2f}\n")

        # Verificar se há POIs de alojamento disponíveis; se não, relaxar constraint
        if include_accommodation:
            has_accom = any(p['category'] in ACCOMMODATION_BUNDLES for p in candidate_pois)
            if not has_accom:
                include_accommodation = False
                if verbose:
                    print("   AVISO: Sem alojamento disponível na área — constraint relaxada\n")
            else:
                # Dar peso positivo ao alojamento para não penalizar o category_component
                for cat in ACCOMMODATION_BUNDLES:
                    if cat not in preferences.category_weights:
                        preferences.category_weights[cat] = 0.4

        # Filtro geográfico: union de círculos por cidade + corredor entre segmentos
        # - Círculo de cada cidade especificada (raio = city radius do LocationResolver)
        # - Corredor: POIs a <= CORRIDOR_KM do segmento entre cidades consecutivas
        # - Regiões grandes (r > 200km): sem post-filter (bbox do RAG já é suficiente)
        from src.utils.distance_calculator import haversine as _hav

        CORRIDOR_KM = 25.0  # buffer lateral entre cidades (as cidades em si têm o seu raio)

        def _dist_to_seg(plat, plon, alat, alon, blat, blon):
            abx, aby = blon - alon, blat - alat
            apx, apy = plon - alon, plat - alat
            denom = abx**2 + aby**2
            t = max(0.0, min(1.0, (apx*abx + apy*aby) / denom)) if denom > 1e-10 else 0.0
            return _hav(plat, plon, alat + t*aby, alon + t*abx)

        def _in_area(plat, plon):
            # Sem nenhuma localizacao resolvida: nao filtrar geograficamente
            # (o RAG ja aplicou o seu proprio filtro de bbox, se algum)
            if not all_geos:
                return True
            # Regiões grandes (Portugal, Alentejo...): bbox do RAG é suficiente
            if any(r > 200.0 for _, _, r in all_geos):
                return True
            # 1. Dentro do círculo de qualquer cidade especificada
            for clat, clon, r_km in all_geos:
                if _hav(plat, plon, clat, clon) <= r_km:
                    return True
            # 2. Dentro do corredor entre segmentos (só multi-cidade)
            if is_corridor:
                for i in range(len(all_geos) - 1):
                    la, loa, _ = all_geos[i]
                    lb, lob, _ = all_geos[i + 1]
                    if _dist_to_seg(plat, plon, la, loa, lb, lob) <= CORRIDOR_KM:
                        return True
            return False

        before = len(candidate_pois)
        candidate_pois = [p for p in candidate_pois if _in_area(p['lat'], p['lon'])]
        if verbose and len(candidate_pois) < before:
            mode = "circles+corredor" if is_corridor else "circles"
            print(f"   Filtro geo ({mode}): {before} -> {len(candidate_pois)} POIs\n")

        # Hard filter: remover categorias operacionais/administrativas do pool
        _never_set = set(NEVER_INCLUDE_CATEGORIES)
        before_never = len(candidate_pois)
        candidate_pois = [p for p in candidate_pois if p.get('category') not in _never_set]
        if verbose and len(candidate_pois) < before_never:
            print(f"   Filtro categorias excluidas: {before_never} -> {len(candidate_pois)} POIs\n")

        # Density boost para corredores: reponderar por densidade local
        if is_corridor and candidate_pois:
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
                print(f"   Density boost aplicado (raio {DENSITY_RADIUS_KM:.0f}km)\n")

        # Fallback de emergência: se pool ficou vazio, relaxar filtros geográficos
        # (corre tambem quando geo=None — localizacao nao foi resolvida —, caso em
        # que a query nacional sem filtro geografico e a unica alternativa a um erro)
        if len(candidate_pois) == 0:
            if verbose:
                print("   AVISO: pool vazio apos filtro — a relaxar para bbox\n")
            rag_results_nogeo = self.rag.query(
                text=rag_query,
                n_results=25,
                category_filter=preferences.preferred_categories,
                category_exclude=EXCLUDED_CATEGORIES,
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
        ACCOMMODATION_MIN_COST = 60.0  # custo minimo por noite para alojamento
        optimizer_pois = []
        for p in candidate_pois:
            poi_cost = p['cost']
            if p['category'] in ACCOMMODATION_BUNDLES and poi_cost < ACCOMMODATION_MIN_COST:
                poi_cost = ACCOMMODATION_MIN_COST
            poi_obj = POI(
                id=int(p['id']), name=p['name'],
                lat=p['lat'], lon=p['lon'],
                category=p['category'], score=p['score'],
                duration=p['duration'],
                opening_time=p['opening_time'],
                closing_time=p['closing_time'],
                cost=poi_cost
            )
            optimizer_pois.append(poi_obj)

        if len(optimizer_pois) == 0:
            print("\nAVISO: Nenhum POI para otimizar!")
            return {"error": "NO_OPTIMIZER_POIS", "query": user_query, "n_candidates": 0}

        n_pois = len(optimizer_pois)

        if verbose:
            print(f"   [OK] POIs para otimizacao: {n_pois}\n")

        # Matriz sempre com Haversine — TransitService reservado para visualização do mapa
        # (N² Dijkstra em 15818 nós causa timeout independentemente do nº de POIs)
        _use_transit = False
        if _use_transit:
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
        is_elderly      = getattr(preferences, 'is_elderly', False) or False
        if has_children:
            print("   has_children=True -> modificador contextual activado\n")
        if is_elderly:
            print("   is_elderly=True -> modificador contextual activado\n")

        # Matriz de elevacao - calculada pos-optimizacao (so POIs selecionados, max 20)
        # NAO calcular pre-optimizacao: com 60+ candidatos sao milhares de chamadas HTTP
        elevation_matrix = None

        _num_people = getattr(preferences, "num_people", 1)
        _num_rooms  = num_rooms if num_rooms else getattr(preferences, "num_rooms", max(1, math.ceil(_num_people / 2)))

        user_prefs_dict = {
            "max_time":   preferences.max_time,
            "max_cost":   (preferences.max_cost - meal_budget_reserve) if preferences.max_cost else preferences.max_cost,
            "num_people": _num_people,
            "num_rooms":  _num_rooms,
            "preferred_categories": preferences.preferred_categories,
            "category_weights": preferences.category_weights,
            "start_location": (optimizer_pois[0].lat, optimizer_pois[0].lon),
            "start_time": preferences.start_time,
            "center_lat": geo[0] if geo and geo[2] <= 200.0 else None,
            "center_lon": geo[1] if geo and geo[2] <= 200.0 else None,
            "max_radius_km": geo[2] if geo else 30.0,
            "all_geos": [[g[0], g[1], g[2]] for g in all_geos] if all_geos else [],
            "is_corridor": is_corridor,
            "mobility_issues": mobility_issues,
            "has_children": has_children,
            "is_elderly": is_elderly,
            "elevation_matrix": elevation_matrix,
            "include_accommodation": include_accommodation,
            "has_nightlife": has_nightlife,
            "max_days": num_days_base,  # dias base sem o bonus de nightlife
        }

        evaluator = RouteEvaluator(optimizer_pois, sub_distance_matrix, user_prefs_dict)

        # ── CAPTURA DE FIXTURE (benchmark) ────────────────────────────────────
        if fixture_capture_path:
            import json as _json
            _fixture = {
                "selected_algo": selected_algo,
                "n_pois": len(optimizer_pois),
                "user_prefs": {
                    k: (list(v) if isinstance(v, tuple) else
                        [[float(x) for x in row] for row in v] if k == "elevation_matrix" and v else v)
                    for k, v in user_prefs_dict.items()
                    if k != "elevation_matrix"
                },
                "pois": [
                    {"id": p.id, "name": p.name, "lat": p.lat, "lon": p.lon,
                     "category": p.category, "score": p.score, "duration": p.duration,
                     "opening_time": p.opening_time, "closing_time": p.closing_time,
                     "cost": p.cost}
                    for p in optimizer_pois
                ],
                "distance_matrix": sub_distance_matrix.tolist(),
            }
            with open(fixture_capture_path, "w", encoding="utf-8") as _f:
                _json.dump(_fixture, _f, ensure_ascii=False)

        # ========== PASSO 5: OTIMIZACAO ==========
        if verbose:
            print(f"[OPTIMIZER-{selected_algo}] Otimizando rota...\n")

        if selected_algo == "ACO":
            optimizer = TourismACO(optimizer_pois, sub_distance_matrix, evaluator,
                                   n_ants=30, n_iterations=100)
        elif selected_algo == "GA":
            # GA_DYN_046 -- melhor config grid search (pop=100, gen=50, cx=0.6, mut=0.1, dynamic)
            optimizer = TourismGA(optimizer_pois, sub_distance_matrix, evaluator,
                                  population_size=100, n_generations=50,
                                  crossover_prob=0.6, mutation_prob=0.1,
                                  mutation_dynamic=True, mutation_patience=5)
        elif selected_algo == "PSO":
            optimizer = TourismPSOA(optimizer_pois, sub_distance_matrix, evaluator,
                                    n_particles=20, n_iterations=30)
        else:  # GREEDY
            optimizer = GreedyPlanner(optimizer_pois, sub_distance_matrix, evaluator)

        optimization_result = optimizer.optimize()

        # Fallback GA se PSO estagna (best fitness nao melhora em nenhuma iteracao)
        if selected_algo == "PSO" and not disable_algo_fallback:
            best_hist = optimization_result.get('best_history', [])
            pso_stagnated = (
                len(best_hist) >= 2
                and best_hist[-1] <= best_hist[0] * 1.005  # <0.5% melhoria total
            )
            if pso_stagnated:
                if verbose:
                    print(f"   PSO estagnado (best={optimization_result['fitness']:.2f}) — fallback GA\n")
                fallback = TourismGA(optimizer_pois, sub_distance_matrix, evaluator,
                                     population_size=100, n_generations=50,
                                     crossover_prob=0.6, mutation_prob=0.1,
                                     mutation_dynamic=True, mutation_patience=5)
                ga_result = fallback.optimize()
                if ga_result['fitness'] >= optimization_result['fitness']:
                    optimization_result = ga_result
                    selected_algo = "GA"
                    if verbose:
                        print(f"   [OK] GA fallback: {optimization_result['fitness']:.2f}\n")

        # ── Post-otimização: reparação de qualidade ─────────────────────────────
        route = list(optimization_result['route'])
        route_ids = set(route)

        # 1) Garantia de categoria: forcar pelo menos 1 POI de cada categoria pedida
        if preferences.preferred_categories:
            route_cats = {optimizer_pois[i].category for i in route}
            for cat in preferences.preferred_categories:
                if cat not in route_cats:
                    # Encontrar o melhor candidato dessa categoria que caiba no tempo/custo
                    candidates_cat = [
                        (i, p) for i, p in enumerate(optimizer_pois)
                        if p.category == cat and i not in route_ids
                    ]
                    candidates_cat.sort(key=lambda x: -x[1].score)
                    for idx, poi in candidates_cat[:5]:
                        trial = route + [idx]
                        if evaluator._is_feasible(trial):
                            route.append(idx)
                            route_ids.add(idx)
                            route_cats.add(cat)
                            if verbose:
                                print(f"   [Repair] +{poi.name} ({cat}) — cobertura de categoria")
                            break

        # 2) Fill com POIs gratuitos se time_utilization < 70%
        time_used = evaluator._calculate_time(route)
        max_time_val = preferences.max_time or 480
        time_util = time_used / max_time_val
        if time_util < 0.70:
            # Primeiro: usar POIs cost=0 já no pool do otimizador
            free_candidates = [
                (i, p) for i, p in enumerate(optimizer_pois)
                if i not in route_ids and p.cost == 0 and p.duration > 0
            ]
            free_candidates.sort(key=lambda x: -x[1].duration)
            for idx, poi in free_candidates:
                trial = route + [idx]
                if evaluator._is_feasible(trial):
                    route.append(idx)
                    route_ids.add(idx)
                    time_used = evaluator._calculate_time(route)
                    time_util = time_used / max_time_val
                    if verbose:
                        print(f"   [Fill]   +{poi.name} ({poi.category}) gratis (pool) — time_util={time_util:.0%}")
                    if time_util >= 0.70:
                        break

            # Se ainda < 70%, fazer 2ª query RAG específica para POIs cost=0
            if time_util < 0.70:
                if verbose:
                    print(f"   [Fill2]  time_util={time_util:.0%} — 2a query RAG para POIs gratuitos\n")
                extra_free = self.rag.query(
                    text=preferences.location or "portugal",
                    n_results=40,
                    category_filter=None,
                    category_exclude=EXCLUDED_CATEGORIES,
                    max_cost=0,           # só cost=0
                    lat_min=lat_min, lat_max=lat_max,
                    lon_min=lon_min, lon_max=lon_max,
                )
                existing_ids = {optimizer_pois[i].id for i in route_ids}
                new_free_pois = []
                for p in extra_free['pois']:
                    if p['id'] not in existing_ids and p['cost'] == 0:
                        # Clampar duração
                        cat = p.get('category', '')
                        if cat in DURATION_RANGES:
                            d_min, d_max = DURATION_RANGES[cat]
                            p['duration'] = max(d_min, min(d_max, p['duration']))
                        new_free_pois.append(p)
                        existing_ids.add(p['id'])

                if new_free_pois and verbose:
                    print(f"   [Fill2]  {len(new_free_pois)} POIs gratuitos adicionais encontrados\n")

                # Adicionar ao optimizer_pois e à matriz (haversine, cost=0)
                from src.utils.distance_calculator import haversine as _hav_fill
                if new_free_pois:
                    n_old = len(optimizer_pois)
                    for p in new_free_pois:
                        poi_obj = POI(
                            id=int(p['id']), name=p['name'],
                            lat=p['lat'], lon=p['lon'],
                            category=p['category'], score=p['score'],
                            duration=p['duration'],
                            opening_time=p['opening_time'],
                            closing_time=p['closing_time'],
                            cost=0.0,
                        )
                        optimizer_pois.append(poi_obj)

                    # Expandir sub_distance_matrix para novos POIs
                    n_new = len(optimizer_pois)
                    new_matrix = np.zeros((n_new, n_new))
                    new_matrix[:n_old, :n_old] = sub_distance_matrix
                    for i in range(n_old, n_new):
                        for j in range(n_new):
                            if i != j:
                                d = _hav_fill(optimizer_pois[i].lat, optimizer_pois[i].lon,
                                              optimizer_pois[j].lat, optimizer_pois[j].lon)
                                t = _travel_time(d, preferences.transport_mode)
                                new_matrix[i][j] = t
                                new_matrix[j][i] = t
                    sub_distance_matrix = new_matrix
                    evaluator.distances = new_matrix
                    evaluator.pois = optimizer_pois
                    # Atualizar n_available_preferred_pois no evaluator
                    evaluator._n_available_preferred_pois = sum(
                        1 for p in optimizer_pois
                        if p.category in evaluator._preferred_cat_set
                    )

                    # Tentar adicionar os novos POIs gratuitos
                    for idx in range(n_old, len(optimizer_pois)):
                        poi = optimizer_pois[idx]
                        if idx not in route_ids:
                            trial = route + [idx]
                            if evaluator._is_feasible(trial):
                                route.append(idx)
                                route_ids.add(idx)
                                time_used = evaluator._calculate_time(route)
                                time_util = time_used / max_time_val
                                if verbose:
                                    print(f"   [Fill2]  +{poi.name} ({poi.category}) gratis (2a query) — time_util={time_util:.0%}")
                                if time_util >= 0.70:
                                    break

        # 3) Fill3: re-query RAG para categorias preferidas ainda ausentes na rota
        if time_util < 0.70 and preferences.preferred_categories:
            route_cats_now = {optimizer_pois[i].category for i in route}
            missing_cats = [c for c in preferences.preferred_categories if c not in route_cats_now]
            if missing_cats:
                if verbose:
                    print(f"   [Fill3]  time_util={time_util:.0%} — re-query RAG cats ausentes: {missing_cats}\n")
                extra_cat_result = self.rag.query(
                    text=rag_query,
                    n_results=len(missing_cats) * 5,
                    category_filter=missing_cats,
                    category_exclude=EXCLUDED_CATEGORIES,
                    lat_min=lat_min, lat_max=lat_max,
                    lon_min=lon_min, lon_max=lon_max,
                )
                existing_ids = {optimizer_pois[i].id for i in range(len(optimizer_pois))}
                new_cat_pois = []
                for p in extra_cat_result['pois']:
                    if p['id'] not in existing_ids:
                        cat = p.get('category', '')
                        if cat in DURATION_RANGES:
                            d_min, d_max = DURATION_RANGES[cat]
                            p['duration'] = max(d_min, min(d_max, p['duration']))
                        new_cat_pois.append(p)
                        existing_ids.add(p['id'])

                if new_cat_pois:
                    if verbose:
                        print(f"   [Fill3]  {len(new_cat_pois)} POIs de cats ausentes encontrados\n")
                    from src.utils.distance_calculator import haversine as _hav_fill3
                    n_old = len(optimizer_pois)
                    for p in new_cat_pois:
                        poi_obj = POI(
                            id=int(p['id']), name=p['name'],
                            lat=p['lat'], lon=p['lon'],
                            category=p['category'], score=p['score'],
                            duration=p['duration'],
                            opening_time=p['opening_time'],
                            closing_time=p['closing_time'],
                            cost=float(p.get('cost', 0)),
                        )
                        optimizer_pois.append(poi_obj)
                    n_new = len(optimizer_pois)
                    new_matrix = np.zeros((n_new, n_new))
                    new_matrix[:n_old, :n_old] = sub_distance_matrix
                    for i in range(n_old, n_new):
                        for j in range(n_new):
                            if i != j:
                                d = _hav_fill3(optimizer_pois[i].lat, optimizer_pois[i].lon,
                                               optimizer_pois[j].lat, optimizer_pois[j].lon)
                                t = _travel_time(d, preferences.transport_mode)
                                new_matrix[i][j] = t
                                new_matrix[j][i] = t
                    sub_distance_matrix = new_matrix
                    evaluator.distances = new_matrix
                    evaluator.pois = optimizer_pois
                    evaluator._n_available_preferred_pois = sum(
                        1 for p in optimizer_pois if p.category in evaluator._preferred_cat_set
                    )
                    for idx in range(n_old, len(optimizer_pois)):
                        poi = optimizer_pois[idx]
                        if idx not in route_ids:
                            trial = route + [idx]
                            if evaluator._is_feasible(trial):
                                route.append(idx)
                                route_ids.add(idx)
                                time_used = evaluator._calculate_time(route)
                                time_util = time_used / max_time_val
                                if verbose:
                                    print(f"   [Fill3]  +{poi.name} ({poi.category}) — time_util={time_util:.0%}")
                                if time_util >= 0.70:
                                    break

        # Recalcular fitness se a rota foi modificada
        if len(route) != len(optimization_result['route']):
            new_fitness = evaluator.calculate_fitness(route)
            if new_fitness >= optimization_result['fitness']:
                optimization_result['route'] = route
                optimization_result['fitness'] = new_fitness
                optimization_result['pois'] = [optimizer_pois[i] for i in route]
                if verbose:
                    print(f"   [OK] Rota reparada: {len(route)} POIs, fitness={new_fitness:.2f}\n")
            elif verbose:
                print(f"   [Repair] Rota original mantida (fitness reparado={new_fitness:.2f} < {optimization_result['fitness']:.2f})\n")

        if verbose:
            print(f"\n   [OK] Fitness: {optimization_result['fitness']:.2f}")
            print(f"   [OK] POIs selecionados: {len(optimization_result['route'])}\n")

        # Calcular elevacao pos-optimizacao (so POIs selecionados, max 20)
        if (mobility_issues or is_elderly) and optimization_result['pois']:
            selected_pois = optimization_result['pois'][:20]
            if verbose:
                print(f"   A calcular elevacao para {len(selected_pois)} POIs selecionados...\n")
            try:
                elev_matrix = self._build_elevation_matrix(selected_pois, verbose=False)
                evaluator.elevation_matrix = elev_matrix
                evaluator.mobility_issues = True
            except Exception as e:
                print(f"AVISO: Elevacao ignorada: {e}\n")

        # ========== PASSO 6: COMPONENTES FITNESS + SHAP ==========
        # fitness_components calculado ANTES da explicacao LLM para alimentar o prompt
        fitness_components = {}
        if optimization_result['route']:
            try:
                fitness_components = evaluator.calculate_fitness_components(optimization_result['route'])
            except Exception:
                pass

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

        if generate_explanation:
            explanation = self.llm.explain_route(
                route=route_dicts,
                preferences=preferences,
                algorithm_used=selected_algo,
                optimization_metadata=optimization_result,
                fitness_components=fitness_components,
                shap_values=shap_explanation.get('shap_values') if shap_explanation else None,
                mobility_issues=mobility_issues,
                has_children=has_children,
                is_elderly=is_elderly,
                num_people=getattr(preferences, 'num_people', 1),
            )
        else:
            explanation = ""

        route_pois_list = [
            {"id": p.id, "name": p.name, "category": p.category,
             "lat": p.lat, "lon": p.lon, "duration": p.duration, "cost": p.cost}
            for p in optimization_result['pois']
        ]

        visit_time = sum(p['duration'] for p in route_pois_list)
        total_time_with_travel = evaluator._calculate_time(optimization_result['route'])
        travel_time = total_time_with_travel - visit_time

        # Custo por pessoa: alojamento dividido por people_per_room, restantes por pessoa
        cost_per_person = sum(
            p['cost'] / evaluator.people_per_room
            if p['category'] in ACCOMMODATION_BUNDLES else p['cost']
            for p in route_pois_list
        )

        result = {
            "query": user_query,
            "preferences": {
                "max_time": preferences.max_time,
                "max_cost": preferences.max_cost,
                "categories": preferences.preferred_categories,
                "interests": preferences.interests,
                "num_people": _num_people,
                "num_rooms": _num_rooms,
                "transport_mode": preferences.transport_mode,
                "location": preferences.location,
                "start_time": preferences.start_time,
                "mobility_issues": mobility_issues,
                "has_children": has_children,
                "is_elderly": is_elderly,
                "include_accommodation": include_accommodation,
                "include_meals": include_meals,
            },
            "cost_per_person": round(cost_per_person, 2),
            "algorithm_used": selected_algo,
            "route": route_pois_list,
            "optimization": {
                "fitness": optimization_result['fitness'],
                "n_candidates": len(candidate_pois),
                "n_selected": len(optimization_result['route']),
                "visit_time_min": visit_time,
                "travel_time_min": travel_time,
                "total_time_min": total_time_with_travel,
                "fitness_history": optimization_result.get('fitness_history', []),
                "fitness_components": fitness_components,
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

            # Usar num_days_base (antes do ajuste nightlife) para nao inflar os dias
            requested_days = num_days_base
            total_days = requested_days

            planner = DayPlanner(
                hours_per_day=8,
                start_time="09:00",
                lunch_break=60,
                transport_mode=preferences.transport_mode or "car",
                transit_service=self.transit_service,
                max_move_km=_move_cap_km,
            )
            if start_geo:
                planner.start_lat = start_geo[0]
                planner.start_lon = start_geo[1]

            first_day_start = preferences.start_time if preferences.start_time != "09:00" else None
            last_day_end = getattr(preferences, 'last_day_end_time', None)

            # Dedup por nome antes de planear (mesmo local pode ter 2 IDs na BD)
            _seen_names: set = set()
            result['route'] = [
                p for p in result['route']
                if p['name'] not in _seen_names and not _seen_names.add(p['name'])
            ]

            # Garantia pré-plano: 2 restaurantes por dia quando include_meals=True
            if include_meals:
                _existing_ids = {p['id'] for p in result['route']}
                _current_meals = sum(1 for p in result['route']
                                     if p.get('category') == 'restaurantes_e_cafes')
                _needed_meals = max(0, total_days * 2 - _current_meals)
                if _needed_meals > 0:
                    _existing_names = {p['name'] for p in result['route']}
                    # Construir pool dedupando por id E nome (atualiza tracking ao iterar)
                    _meal_pool = []
                    for p in candidate_pois:
                        if (p.get('category') == 'restaurantes_e_cafes'
                                and p['id'] not in _existing_ids
                                and p['name'] not in _existing_names):
                            _meal_pool.append(p)
                            _existing_ids.add(p['id'])
                            _existing_names.add(p['name'])
                    # Suplementar via RAG quando o pool de candidate_pois é insuficiente
                    if len(_meal_pool) < _needed_meals and lat_min is not None:
                        _fresh_meal = self.rag.query(
                            text=f"restaurante jantar almoco {preferences.location or ''}",
                            n_results=_needed_meals * 3,
                            category_filter=["restaurantes_e_cafes"],
                            lat_min=lat_min, lat_max=lat_max,
                            lon_min=lon_min, lon_max=lon_max,
                        )
                        for p in _fresh_meal.get('pois', []):
                            if p['id'] not in _existing_ids and p['name'] not in _existing_names:
                                _meal_pool.append(p)
                                _existing_ids.add(p['id'])
                                _existing_names.add(p['name'])
                        if verbose:
                            print(f"   [Meals-Pre] RAG suplementar: {len(_fresh_meal.get('pois', []))} restaurantes encontrados")
                    _meal_pool.sort(key=lambda p: -p.get('score', 0))
                    for _meal in _meal_pool[:_needed_meals]:
                        result['route'].append(_meal)
                        _existing_ids.add(_meal['id'])
                        _existing_names.add(_meal['name'])
                        if verbose:
                            print(f"   [Meals-Pre] +{_meal['name']} (restaurante garantido)\n")

            # Nightlife: grupo de adultos → 1-2 bares (noite de 6ª ou sáb, se datas conhecidas)
            if getattr(preferences, 'nightlife_suggested', False) and lat_min is not None:
                _bar_results = self.rag.query(
                    text="bar discoteca vida noturna cerveja cocktail",
                    n_results=6,
                    category_filter=["bares_e_discotecas"],
                    lat_min=lat_min, lat_max=lat_max,
                    lon_min=lon_min, lon_max=lon_max,
                )
                _existing_bar_ids = {p['id'] for p in result['route']}
                _bars_added = 0
                _max_bars = 2 if has_nightlife else 1
                for _bar in _bar_results.get('pois', []):
                    if _bar['id'] not in _existing_bar_ids and _bars_added < _max_bars:
                        result['route'].append(_bar)
                        _existing_bar_ids.add(_bar['id'])
                        _bars_added += 1
                if _bars_added and verbose:
                    print(f"   [Nightlife] +{_bars_added} bar(es) para grupo de adultos\n")

            _q_low = user_query.lower()
            _route_direction = None
            if any(kw in _q_low for kw in ["norte a sul", "norte para sul", "north to south"]):
                _route_direction = "N2S"
            elif any(kw in _q_low for kw in ["sul a norte", "sul para norte", "south to north"]):
                _route_direction = "S2N"

            day_plan = planner.plan_days(
                result['route'],
                distance_matrix=sub_distance_matrix,
                total_days=total_days,
                first_day_start_time=first_day_start,
                last_day_end_time=last_day_end,
                all_geos=all_geos,
                start_date=getattr(preferences, 'start_date', None),
                route_direction=_route_direction,
            )

            NON_VISIT = ACCOMMODATION_BUNDLES + ["bares_e_discotecas", "casinos"]

            # Fill-D: dias com <360min de actividades → RAG direcionado + injecção directa
            # Não chama plan_days de novo (evita re-clustering que desfaz o fill)
            if day_plan and day_plan.get('days') and lat_min is not None:
                existing_ids = {p['id'] for p in result['route']}
                existing_names = {p['name'] for p in result['route']}
                for i, day in enumerate(day_plan['days']):
                    day_pois = [p for p in day['pois'] if p.get('category') not in NON_VISIT]
                    day_time = sum(p.get('duration', 0) for p in day_pois)
                    if day_time >= 360:
                        continue
                    # Centróide do dia (usa bbox geral se dia vazio)
                    geo_pois = [p for p in day_pois if p.get('lat') is not None]
                    if geo_pois:
                        clat = sum(p['lat'] for p in geo_pois) / len(geo_pois)
                        clon = sum(p['lon'] for p in geo_pois) / len(geo_pois)
                    else:
                        clat = (lat_min + lat_max) / 2
                        clon = (lon_min + lon_max) / 2
                    d_deg = 25.0 / 111.0
                    extra = self.rag.query(
                        text=rag_query,
                        n_results=8,
                        category_filter=preferences.preferred_categories,
                        category_exclude=EXCLUDED_CATEGORIES,
                        lat_min=max(lat_min, clat - d_deg),
                        lat_max=min(lat_max, clat + d_deg),
                        lon_min=max(lon_min, clon - d_deg),
                        lon_max=min(lon_max, clon + d_deg),
                    )
                    added = []
                    for ep in extra.get('pois', []):
                        if ep['id'] not in existing_ids and ep['name'] not in existing_names:
                            ep_cat = ep.get('category', '')
                            if (ep_cat not in DayPlanner.NOCTURNO_CATEGORIES
                                    and ep_cat not in DayPlanner.ACCOMMODATION_CATEGORIES):
                                added.append(ep)
                                existing_ids.add(ep['id'])
                                existing_names.add(ep['name'])

                    # B1: preferred_categories esgotadas (ex: cap=2 ja atingido) →
                    # tentar categorias do mesmo tema antes de deixar o dia incompleto
                    if not added:
                        sibling_cats = _thematic_siblings(preferences.preferred_categories)
                        if sibling_cats:
                            extra2 = self.rag.query(
                                text=rag_query,
                                n_results=8,
                                category_filter=sibling_cats,
                                category_exclude=EXCLUDED_CATEGORIES,
                                lat_min=max(lat_min, clat - d_deg),
                                lat_max=min(lat_max, clat + d_deg),
                                lon_min=max(lon_min, clon - d_deg),
                                lon_max=min(lon_max, clon + d_deg),
                            )
                            for ep in extra2.get('pois', []):
                                if ep['id'] not in existing_ids and ep['name'] not in existing_names:
                                    ep_cat = ep.get('category', '')
                                    if (ep_cat not in DayPlanner.NOCTURNO_CATEGORIES
                                            and ep_cat not in DayPlanner.ACCOMMODATION_CATEGORIES):
                                        added.append(ep)
                                        existing_ids.add(ep['id'])
                                        existing_names.add(ep['name'])
                            if added and verbose:
                                print(f"   [Fill-D] Dia {day['day']}: categorias preferidas esgotadas, "
                                      f"a usar grupo tematico {sibling_cats}")
                    if added:
                        for ep in added:
                            result['route'].append(ep)
                        day_num = day['day']
                        day_idx = day_num - 1
                        if hasattr(planner, '_last_diurnal_by_day') and day_idx < len(planner._last_diurnal_by_day):
                            planner._last_diurnal_by_day[day_idx].extend(added)
                            day_start = (first_day_start if day_num == 1 and first_day_start
                                         else planner.start_time)
                            hotel = (planner._last_day_hotels[day_idx]
                                     if day_idx < len(planner._last_day_hotels) else None)
                            night = (planner._last_nocturnal_by_day[day_idx]
                                     if day_idx < len(planner._last_nocturnal_by_day) else [])
                            new_day = planner._format_day(
                                day_num,
                                planner._last_diurnal_by_day[day_idx],
                                night,
                                day_start_time=day_start,
                                hotel=hotel,
                            )
                            day_plan['days'][i] = new_day
                        if verbose:
                            print(f"   [Fill-D] Dia {day['day']}: +{len(added)} POI(s)\n")

            result['day_plan'] = day_plan

            # Dedup final por nome: apanha duplicados introduzidos pelo Fill-D
            # (mesmo POI com IDs distintos na BD, ou RAG a retornar o mesmo)
            _seen_final: set = set()
            result['route'] = [
                p for p in result['route']
                if p['name'] not in _seen_final and not _seen_final.add(p['name'])
            ]

            # Sincronizar result['route'] com o day_plan: remover POIs que o day_planner
            # não conseguiu agendar (ex: bares fora da janela 21:00-03:00).
            # Usa nomes em vez de IDs para evitar falsos positivos por type mismatch
            # (IDs do optimizador são int; IDs do RAG podem ser str).
            if day_plan and day_plan.get('days'):
                scheduled_names = {
                    p['name']
                    for day in day_plan['days']
                    for p in day['pois']
                }
                removed = [p for p in result['route'] if p['name'] not in scheduled_names]
                if removed:
                    result['route'] = [p for p in result['route'] if p['name'] in scheduled_names]
                    # Recalcular custo por pessoa com a rota filtrada
                    result['cost_per_person'] = round(sum(
                        p['cost'] / evaluator.people_per_room
                        if p['category'] in ACCOMMODATION_BUNDLES else p['cost']
                        for p in result['route']
                    ), 2)
                    if verbose:
                        names = [p['name'] for p in removed]
                        print(f"   [Sync] {len(removed)} POI(s) removido(s) (não agendados): {names}\n")

            # Notas de transporte (B5): aviso de troços longos a pé -
            # adicionado de forma deterministica (sem LLM) ao final da explicacao
            if (day_plan and day_plan.get('days') and result.get('explanation')
                    and preferences.transport_mode == "foot"):
                foot_fallback_kms = [
                    p['travel_km']
                    for day in day_plan['days']
                    for p in day['pois']
                    if p.get('travel_mode') == 'car' and p.get('travel_km') is not None
                ]
                if foot_fallback_kms:
                    kms_str = ", ".join(f"{km:.1f} km" for km in foot_fallback_kms)
                    result['explanation'] += (
                        f"\n\nAlguns troços deste plano são longos para fazer a pé ({kms_str})"
                        " — para esses, considera apanhar um táxi/Uber."
                    )

            if verbose:
                planner.print_itinerary(day_plan)

        except Exception as e:
            print(f"AVISO: Erro ao planear dias: {e}\n")
            import traceback
            traceback.print_exc()

        # ========== PASSO 9: GERAR MAPA OSRM ==========
        if not generate_map:
            result["map_file"] = None
        else:
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
                    transit_service=self.transit_service,
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