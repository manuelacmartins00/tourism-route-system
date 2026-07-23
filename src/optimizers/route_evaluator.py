# src/optimizers/route_evaluator.py

import numpy as np
import math
from typing import List, Dict
from dataclasses import dataclass
from collections import Counter

@dataclass
class POI:
    id: int
    name: str
    lat: float
    lon: float
    category: str
    score: float
    duration: int
    opening_time: str
    closing_time: str
    cost: float

ACCOMMODATION_CATEGORIES = frozenset({
    "hotelaria", "alojamento_local", "turismo_habitacao",
    "turismo_espaco_rural", "apartamento_turistico",
    "pousadas_da_juventude", "aldeamento_turistico", "parques_de_campismo",
})

class RouteEvaluator:
    """Avalia qualidade de rotas turisticas"""
    
    def __init__(self, pois: List[POI], distance_matrix: np.ndarray, user_prefs: Dict):
        self.pois = pois
        self.distances = distance_matrix
        self.prefs = user_prefs
        
        self.w_distance      = 0.1095  # AHP 5x5 CR=0.0081
        # cat_indata: satisfação de preferência (optimizer usou POIs preferidos disponíveis?)
        # — 3.º critério mais importante, alinhado com literatura OP (score/profit maximization)
        self.w_cat_indata    = 0.1686
        # cat_general: cobertura de categorias pedidas (julga dados + modelo) — peso menor
        self.w_cat_general   = 0.0600
        # diversity: variedade de categorias — qualidade secundária, peso reduzido
        self.w_diversity     = 0.0800
        self.w_time          = 0.3934
        self.w_proximity     = 0.1885  # AHP 5x5 CR=0.0081
        # w_total = 0.1095+0.1686+0.0600+0.0800+0.3934+0.1885 = 1.0000

        self.center_lat = user_prefs.get("center_lat")
        self.center_lon = user_prefs.get("center_lon")
        self.max_radius_km   = user_prefs.get("max_radius_km", 20.0)
        self.mobility_issues = user_prefs.get("mobility_issues", False)
        self.has_children    = user_prefs.get("has_children", False)
        self.has_nightlife   = user_prefs.get("has_nightlife", False)
        self.max_days        = user_prefs.get("max_days", 1)
        self.num_people      = max(1, user_prefs.get("num_people", 1))
        self.num_rooms       = max(1, user_prefs.get("num_rooms", max(1, math.ceil(self.num_people / 2))))
        self.people_per_room = self.num_people / self.num_rooms
        self.elevation_matrix = user_prefs.get("elevation_matrix", None)
        self.w_elevation = 0.15

        self.is_elderly      = user_prefs.get("is_elderly", False)

        self._CHILDREN_PENALTY = {"bares_e_discotecas", "casinos", "turismo_activo"}
        self._CHILDREN_BONUS   = {"espacos_verdes", "parques_e_reservas", "parques_de_diversao",
                                   "zoos_e_aquarios", "ciencia_e_conhecimento"}
        self._MOBILITY_PENALTY = {"turismo_activo", "campos", "parques_e_reservas",
                                   "parques_de_diversao", "grutas"}
        self._MOBILITY_BONUS   = {"restaurantes_e_cafes", "monumentos", "museus_e_palacios",
                                   "espacos_verdes", "termas", "ciencia_e_conhecimento", "talassoterapia"}
        self._ELDERLY_PENALTY  = {"turismo_activo", "bares_e_discotecas", "campos"}
        self._ELDERLY_BONUS    = {"museus_e_palacios", "monumentos", "termas", "espacos_verdes",
                                   "talassoterapia", "ciencia_e_conhecimento", "restaurantes_e_cafes"}
            
        self._empty_warning_shown = False

        # Categorias de actividade disponíveis no pool de candidatos (excluindo alojamento).
        self._available_activity_cats = frozenset(
            p.category for p in pois if p.category not in ACCOMMODATION_CATEGORIES
        )
        preferred_cats = user_prefs.get('preferred_categories') or []
        self._available_preferred = [c for c in preferred_cats
                                      if c in self._available_activity_cats]
        self._missing_preferred   = [c for c in preferred_cats
                                      if c not in self._available_activity_cats]
        self._preferred_cat_set   = frozenset(self._available_preferred)
        # Pre-calcular n de POIs preferidos disponíveis no pool
        self._n_available_preferred_pois = sum(
            1 for p in pois
            if p.category in self._preferred_cat_set
        )

        # Geometria geográfica para distance_penalty e prox_comp
        _geos = user_prefs.get("all_geos") or []
        self.all_geos    = [(g[0], g[1], g[2]) for g in _geos]
        self.is_corridor = user_prefs.get("is_corridor", False)
        self._large_region = any(r > 200.0 for _, _, r in self.all_geos) if self.all_geos else False
    
    def calculate_fitness(self, route: List[int]) -> float:
        """Calcula fitness da rota"""
        
        if not self.pois:
            if not self._empty_warning_shown:
                print("   AVISO: [RouteEvaluator] Lista de POIs vazia!")
                self._empty_warning_shown = True
            return 0.0

        if route:
            max_valid_index = len(self.pois) - 1
            for idx in route:
                if idx > max_valid_index:
                    return 0.0
        
        if not route or not self._is_feasible(route):
            return 0.0

        activity_route = [idx for idx in route
                          if self.pois[idx].category not in ACCOMMODATION_CATEGORIES]
        non_accom_route = activity_route  # alias

        # ── distance_penalty ─────────────────────────────────────────────────
        # 100% se dentro da área declarada; decai linearmente só a partir da fronteira.
        # ratio ≤ 1.0 → dentro → 100%; ratio > 1.0 → fora → max(0, 2 - ratio) × 100%
        if self.all_geos and not self._large_region:
            def _dp_score(ratio):
                return 1.0 if ratio <= 1.0 else max(0.0, 2.0 - ratio)
            dp_scores = [_dp_score(self._geo_ratio(self.pois[idx].lat, self.pois[idx].lon))
                         for idx in activity_route]
            distance_penalty = (sum(dp_scores) / len(dp_scores) * 100) if dp_scores else 100.0
        else:
            # Fallback (sem info geográfica ou região grande)
            total_km = sum(
                self._haversine_km(self.pois[activity_route[pos]], self.pois[activity_route[pos+1]])
                for pos in range(len(activity_route)-1)
            ) if len(activity_route) > 1 else 0
            threshold_km = max(1.0, self.max_radius_km) * 4
            distance_penalty = max(0.0, 100.0 - (total_km / threshold_km) * 100.0)

        # ── cat_indata_comp (julga MODELO) ───────────────────────────────────
        # "O otimizador usou os POIs preferidos disponíveis localmente?"
        # 100% se todos os POIs preferidos disponíveis foram incluídos na rota.
        n_preferred_in_route = sum(1 for idx in non_accom_route
                                   if self.pois[idx].category in self._preferred_cat_set)
        max_achievable = max(1, min(self._n_available_preferred_pois, max(1, len(non_accom_route))))
        cat_indata_comp = min(n_preferred_in_route / max_achievable, 1.0) * 100

        # ── cat_general_comp (julga DADOS + modelo) ──────────────────────────
        # "Quantas das categorias pedidas estão representadas na rota?"
        # Inclui categorias ausentes localmente → baixo quando dados são escassos.
        requested_cats = set(self.prefs.get('preferred_categories') or [])
        route_cats = set(self.pois[idx].category for idx in non_accom_route)
        cats_covered = len(requested_cats & route_cats)
        cat_general_comp = (cats_covered / max(1, len(requested_cats))) * 100

        # ── diversity_component ──────────────────────────────────────────────
        # Cap pelo número de categorias de actividade disponíveis localmente
        unique_categories = len(set(
            self.pois[idx].category for idx in (non_accom_route or route)
        ))
        preferred_count = len(self.prefs.get('preferred_categories') or [])
        n_local_cats = max(1, len(self._available_activity_cats))
        diversity_cap = max(1, min(n_local_cats, max(6, min(preferred_count, 8))))
        diversity_component = min(unique_categories / diversity_cap, 1.0) * 100

        # Time utilization: threshold reduzido 70->50%, multiplicador 0.35->0.55
        # Rationale: o cliff a 70% penalizava desproporcionalmente rotas com orcamento limitado
        time_used = self._calculate_time(route)
        max_time = int(self.prefs.get('max_time', 480))
        time_utilization = min(100, (time_used / max_time) * 100)
        if time_utilization < 50:
            time_efficiency = time_utilization * 0.55
        else:
            time_efficiency = time_utilization
        
        proximity_component = self._proximity_component(route)
        elevation_penalty = self._elevation_component(route) if self.mobility_issues else 100.0

        if self.mobility_issues:
            scale = 1 - self.w_elevation
            fitness = (
                self.w_distance    * scale * distance_penalty +
                self.w_cat_indata  * scale * cat_indata_comp +
                self.w_cat_general * scale * cat_general_comp +
                self.w_diversity   * scale * diversity_component +
                self.w_time        * scale * time_efficiency +
                self.w_proximity   * scale * proximity_component +
                self.w_elevation   * elevation_penalty
            )
        else:
            fitness = (
                self.w_distance    * distance_penalty +
                self.w_cat_indata  * cat_indata_comp +
                self.w_cat_general * cat_general_comp +
                self.w_diversity   * diversity_component +
                self.w_time        * time_efficiency +
                self.w_proximity   * proximity_component
            )
        
        return min(100.0, fitness * self._contextual_modifier(route) * self._duplicate_penalty(route))

    def _duplicate_penalty(self, route: List[int]) -> float:
        """
        Pequena penalizacao multiplicativa por repetir o mesmo POI (por nome)
        mais de uma vez na rota inteira -- excepto praias (repetir a mesma
        praia em dias diferentes de uma viagem de sol-e-praia e normal) e
        alojamento (a mesma pousada em varios dias e o esperado). Nao bloqueia
        a rota -- isso e feito por dia no day_planner (_dedupe_same_day_poi);
        aqui e so um desincentivo suave a repeticao ao longo da viagem toda.
        """
        names = [self.pois[idx].name for idx in route
                 if self.pois[idx].category != "praias"
                 and self.pois[idx].category not in ACCOMMODATION_CATEGORIES]
        if len(names) < 2:
            return 1.0
        counts = Counter(names)
        n_extra = sum(c - 1 for c in counts.values() if c > 1)
        if n_extra == 0:
            return 1.0
        return max(0.85, 1.0 - 0.05 * n_extra)

    def calculate_fitness_components(self, route: List[int]) -> dict:
        """Devolve breakdown dos componentes de fitness (para debug e eval)."""
        if not route or not self._is_feasible(route):
            return {"feasible": False, "fitness": 0.0}
        activity_route = [idx for idx in route
                          if self.pois[idx].category not in ACCOMMODATION_CATEGORIES]
        total_km = sum(
            self._haversine_km(self.pois[activity_route[pos]], self.pois[activity_route[pos+1]])
            for pos in range(len(activity_route)-1)
        ) if len(activity_route) > 1 else 0
        threshold_km = max(1.0, self.max_radius_km) * 4
        distance_penalty = max(0.0, 100.0 - (total_km / threshold_km) * 100.0)
        non_accom_route = [idx for idx in route
                           if self.pois[idx].category not in ACCOMMODATION_CATEGORIES]
        # cat_indata_comp: julga modelo
        n_preferred_in_route = sum(1 for idx in non_accom_route
                                   if self.pois[idx].category in self._preferred_cat_set)
        max_achievable = max(1, min(self._n_available_preferred_pois, max(1, len(non_accom_route))))
        cat_indata_comp = min(n_preferred_in_route / max_achievable, 1.0) * 100
        # cat_general_comp: julga dados + modelo
        requested_cats = set(self.prefs.get('preferred_categories') or [])
        route_cats_set = set(self.pois[idx].category for idx in non_accom_route)
        cat_general_comp = (len(requested_cats & route_cats_set) / max(1, len(requested_cats))) * 100
        # diversity
        unique_categories = len(set(
            self.pois[idx].category for idx in non_accom_route or route))
        preferred_count = len(self.prefs.get('preferred_categories') or [])
        n_local_cats = max(1, len(self._available_activity_cats))
        diversity_cap = max(1, min(n_local_cats, max(6, min(preferred_count, 8))))
        diversity_component = min(unique_categories / diversity_cap, 1.0) * 100
        # distance_penalty: 100% dentro da área, decai fora da fronteira
        if self.all_geos and not self._large_region:
            def _dp_s(ratio): return 1.0 if ratio <= 1.0 else max(0.0, 2.0 - ratio)
            dp_scores = [_dp_s(self._geo_ratio(self.pois[idx].lat, self.pois[idx].lon))
                         for idx in non_accom_route]
            distance_penalty = (sum(dp_scores)/len(dp_scores)*100) if dp_scores else 100.0
        else:
            activity_r = non_accom_route
            total_km = sum(self._haversine_km(self.pois[activity_r[i]],self.pois[activity_r[i+1]])
                           for i in range(len(activity_r)-1)) if len(activity_r)>1 else 0
            distance_penalty = max(0.0, 100.0 - (total_km/max(1.0,self.max_radius_km*4))*100)
        time_used = self._calculate_time(route)
        max_time = int(self.prefs.get('max_time', 480))
        time_utilization = min(100, (time_used / max_time) * 100)
        time_efficiency = time_utilization if time_utilization >= 50 else time_utilization * 0.55
        proximity_component = self._proximity_component(route)
        modifier = self._contextual_modifier(route)
        fitness = self.calculate_fitness(route)
        return {
            "feasible": True,
            "fitness": round(fitness, 3),
            "time_utilization": round(time_utilization, 1),
            "time_efficiency": round(time_efficiency, 1),
            "cat_indata_comp":    round(cat_indata_comp, 1),
            "cat_general_comp":   round(cat_general_comp, 1),
            "category_component": round((cat_indata_comp + cat_general_comp) / 2, 1),  # compatibilidade
            "diversity_component": round(diversity_component, 1),
            "distance_penalty": round(distance_penalty, 1),
            "proximity_component": round(proximity_component, 1),
            "contextual_modifier": round(modifier, 3),
            "duplicate_penalty": round(self._duplicate_penalty(route), 3),
            "n_route": len(route),
            "unique_categories": unique_categories,
            # Cobertura local: para diagnóstico e explicação
            "available_preferred": list(self._available_preferred),
            "missing_preferred":   list(self._missing_preferred),
            "data_coverage_pct": round(
                100 * len(self._available_preferred) /
                max(1, len(self._available_preferred) + len(self._missing_preferred)), 1
            ),
        }

    def _contextual_modifier(self, route: List[int]) -> float:
        """
        Multiplica o fitness por um factor contextual [0.8, 1.2] baseado em
        has_children, mobility_issues e is_elderly. Penaliza categorias
        inapropriadas e bonifica categorias adequadas ao contexto do utilizador.
        """
        if not self.has_children and not self.mobility_issues and not self.is_elderly:
            return 1.0
        total = 0.0
        for poi_idx in route:
            cat = self.pois[poi_idx].category
            delta = 0.0
            if self.has_children:
                if cat in self._CHILDREN_PENALTY:
                    delta -= 0.15
                elif cat in self._CHILDREN_BONUS:
                    delta += 0.10
            if self.mobility_issues:
                if cat in self._MOBILITY_PENALTY:
                    delta -= 0.15
                elif cat in self._MOBILITY_BONUS:
                    delta += 0.10
            if self.is_elderly:
                if cat in self._ELDERLY_PENALTY:
                    delta -= 0.15
                elif cat in self._ELDERLY_BONUS:
                    delta += 0.10
            total += delta
        return max(0.8, min(1.2, 1.0 + total / len(route)))

    def _proximity_component(self, route: List[int]) -> float:
        """
        Proximidade ao conjunto geográfico declarado (cidade(s) + corredor).
        100% = todos os POIs dentro dos raios das cidades.
        Decai quadraticamente conforme os POIs saem da área.
        """
        if not route:
            return 100.0

        activity = [idx for idx in route
                    if self.pois[idx].category not in ACCOMMODATION_CATEGORIES]
        if not activity:
            return 100.0

        if self.all_geos and not self._large_region:
            # 100% dentro da área; decai quadraticamente só a partir da fronteira
            def _ps(ratio):
                return 1.0 if ratio <= 1.0 else max(0.0, 1.0 - (ratio - 1.0) ** 2)
            scores = [_ps(self._geo_ratio(self.pois[idx].lat, self.pois[idx].lon))
                      for idx in activity]
            return (sum(scores) / len(scores)) * 100
        elif self.center_lat is not None:
            # Fallback: centro único
            def _h(lat1,lon1,lat2,lon2):
                R=6371; r=math.radians
                a=math.sin(r(lat2-lat1)/2)**2+math.cos(r(lat1))*math.cos(r(lat2))*math.sin(r(lon2-lon1)/2)**2
                return R*2*math.asin(math.sqrt(a))
            scores = [max(0.0, 1.0 - (_h(self.center_lat,self.center_lon,
                                         self.pois[idx].lat,self.pois[idx].lon)
                                       / self.max_radius_km)**2)
                      for idx in activity]
            return (sum(scores) / len(scores)) * 100
        else:
            return 100.0

    def _is_feasible(self, route: List[int]) -> bool:
        """Verifica se a rota respeita constraints"""

        if not route or not self.pois:
            return False

        max_valid_index = len(self.pois) - 1
        for idx in route:
            if idx > max_valid_index or idx < 0:
                return False

        total_time = self._calculate_time(route)
        max_time = int(self.prefs.get('max_time', 480))
        if total_time > max_time:
            return False

        # Custo: alojamento dividido por pessoas/quarto, restantes por pessoa
        total_cost = 0.0
        for poi_idx in route:
            poi = self.pois[poi_idx]
            if poi.category in ACCOMMODATION_CATEGORIES:
                total_cost += poi.cost / self.people_per_room
            else:
                total_cost += poi.cost
        max_cost = float(self.prefs.get('max_cost', 1000))
        if total_cost > max_cost:
            return False

        # Hard constraint: pelo menos 1 POI de alojamento (só se include_accommodation=True)
        if self.prefs.get("include_accommodation", True):
            has_accommodation = any(
                self.pois[idx].category in ACCOMMODATION_CATEGORIES for idx in route
            )
            if not has_accommodation:
                return False

        # Limite de POIs noturnos: em modo pub crawl (2+ bares/noite) cada bar dura ~60 min
        # → janela 360 min / 60 min = 6 bares/noite × n_noites disponíveis
        NOCTURNAL_CATS = {"bares_e_discotecas", "casinos"}
        NIGHT_WINDOW_MIN   = 360  # 21:00 → 03:00
        PUB_CRAWL_DURATION = 60   # duração por paragem em modo pub crawl
        available_nights   = max(1, self.max_days - 1)
        max_nocturnal      = (NIGHT_WINDOW_MIN // PUB_CRAWL_DURATION) * available_nights  # 6 × noites
        n_nocturnal = sum(1 for idx in route if self.pois[idx].category in NOCTURNAL_CATS)
        if n_nocturnal > max_nocturnal:
            return False

        return True

    def _calculate_time(self, route: List[int]) -> float:
        """Calcula tempo total da rota em minutos"""
        if not route:
            return 0
        n = len(self.pois)
        total_time = sum(self.pois[poi_idx].duration for poi_idx in route if poi_idx < n)
        for pos in range(len(route) - 1):
            a, b = route[pos], route[pos + 1]
            if a < n and b < n:
                total_time += self.distances[a][b]
        return total_time

    def _elevation_component(self, route: List[int]) -> float:
        """
        Penaliza rotas com elevado ganho de elevacao acumulado entre POIs consecutivos.
        So activado se mobility_issues=True e elevation_matrix disponivel.
        Devolve 0-100 (100 = rota plana, 0 = rota muito inclinada).
        """
        if self.elevation_matrix is None or len(route) < 2:
            return 100.0

        THRESHOLD_M = 50
        MAX_GAIN_M  = 200

        n = len(self.elevation_matrix)
        total_gain = 0.0
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            if a >= n or b >= n:
                return 100.0  # índices fora da matriz pós-otimização → sem penalização
            total_gain += self.elevation_matrix[a][b]

        if total_gain <= THRESHOLD_M:
            return 100.0
        score = max(0.0, 1.0 - (total_gain - THRESHOLD_M) / (MAX_GAIN_M - THRESHOLD_M))
        return score * 100
    
    @staticmethod
    def _hav_coords(lat1, lon1, lat2, lon2) -> float:
        R = 6371.0
        r = math.radians
        a = (math.sin(r(lat2-lat1)/2)**2
             + math.cos(r(lat1))*math.cos(r(lat2))*math.sin(r(lon2-lon1)/2)**2)
        return R * 2 * math.asin(math.sqrt(a))

    @staticmethod
    def _dist_to_seg_km(plat, plon, alat, alon, blat, blon) -> float:
        """Distância em km de um ponto ao segmento AB."""
        abx, aby = blon - alon, blat - alat
        apx, apy = plon - alon, plat - alat
        denom = abx**2 + aby**2
        t = max(0.0, min(1.0, (apx*abx + apy*aby) / denom)) if denom > 1e-10 else 0.0
        R = 6371.0; r = math.radians
        nlat, nlon = alat + t*aby, alon + t*abx
        a = (math.sin(r(nlat-plat)/2)**2
             + math.cos(r(plat))*math.cos(r(nlat))*math.sin(r(nlon-plon)/2)**2)
        return R * 2 * math.asin(math.sqrt(a))

    def _geo_ratio(self, poi_lat, poi_lon) -> float:
        """
        Rácio de proximidade ao conjunto geográfico declarado (cidades + corredor).
        0.0 = no centro da cidade mais próxima
        1.0 = exatamente na fronteira do círculo / corredor
        >1.0 = fora da área (penalizado)
        Regiões grandes (r>200km): sempre 0.0.
        """
        if not self.all_geos or self._large_region:
            return 0.0
        # Distância normalizada à cidade mais próxima
        min_ratio = min(
            self._hav_coords(poi_lat, poi_lon, clat, clon) / max(r, 1.0)
            for clat, clon, r in self.all_geos
        )
        # Para corredores: também verificar distância ao segmento (buffer 25km)
        if self.is_corridor and len(self.all_geos) > 1:
            CORRIDOR_KM = 25.0
            for i in range(len(self.all_geos) - 1):
                la, loa, _ = self.all_geos[i]
                lb, lob, _ = self.all_geos[i + 1]
                d_seg = self._dist_to_seg_km(poi_lat, poi_lon, la, loa, lb, lob)
                seg_ratio = d_seg / CORRIDOR_KM
                min_ratio = min(min_ratio, seg_ratio)
        return max(0.0, min_ratio)

    @staticmethod
    def _haversine_km(poi_a, poi_b) -> float:
        R = 6371.0
        r = math.radians
        lat1, lon1 = poi_a.lat, poi_a.lon
        lat2, lon2 = poi_b.lat, poi_b.lon
        a = math.sin(r(lat2-lat1)/2)**2 + math.cos(r(lat1))*math.cos(r(lat2))*math.sin(r(lon2-lon1)/2)**2
        return R * 2 * math.asin(math.sqrt(a))

    def _parse_time(self, time_str: str) -> float:
        """Converte "09:30" para minutos desde meia-noite"""
        try:
            h, m = map(int, time_str.split(':'))
            return h * 60 + m
        except:
            return 540  # 09:00 default