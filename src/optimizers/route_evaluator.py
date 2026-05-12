# src/optimizers/route_evaluator.py (FIX CONSTRAINTS RELAXADOS)

import numpy as np
import math
from typing import List, Dict
from dataclasses import dataclass

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
        
        self.w_distance  = 0.1095  # AHP 5x5 CR=0.0081
        self.w_category  = 0.1885  # AHP 5x5 CR=0.0081
        self.w_diversity = 0.0324  # AHP 5x5 CR=0.0081
        self.w_time      = 0.4810  # AHP 5x5 CR=0.0081
        self.w_proximity = 0.1885  # AHP 5x5 CR=0.0081

        self.center_lat = user_prefs.get("center_lat")
        self.center_lon = user_prefs.get("center_lon")
        self.max_radius_km   = user_prefs.get("max_radius_km", 20.0)
        self.mobility_issues = user_prefs.get("mobility_issues", False)
        self.has_children    = user_prefs.get("has_children", False)
        self.num_people      = max(1, user_prefs.get("num_people", 1))
        self.num_rooms       = max(1, user_prefs.get("num_rooms", max(1, math.ceil(self.num_people / 2))))
        self.people_per_room = self.num_people / self.num_rooms
        self.elevation_matrix = user_prefs.get("elevation_matrix", None)
        self.w_elevation = 0.15

        self._CHILDREN_PENALTY = {"bares_e_discotecas", "casinos", "turismo_activo"}
        self._CHILDREN_BONUS   = {"espacos_verdes", "parques_e_reservas", "parques_de_diversao",
                                   "zoos_e_aquarios", "ciencia_e_conhecimento"}
        self._MOBILITY_PENALTY = {"turismo_activo", "campos", "parques_e_reservas",
                                   "parques_de_diversao", "grutas"}
        self._MOBILITY_BONUS   = {"restaurantes_e_cafes", "monumentos", "museus_e_palacios",
                                   "espacos_verdes", "termas", "ciencia_e_conhecimento", "talassoterapia"}
            
        self._debug_mode = False
        self._empty_warning_shown = False
    
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
                    if self._debug_mode:
                        print(f"   AVISO: Indice {idx} invalido! Max: {max_valid_index}")
                    return 0.0
        
        if not route or not self._is_feasible(route):
            return 0.0
        
        total_distance = sum(
            self.distances[route[pos]][route[pos+1]]
            for pos in range(len(route)-1)
        ) if len(route) > 1 else 0
        distance_penalty = max(0, 100 - (total_distance / 50) * 100)
        
        category_matches = sum(
            self.prefs.get('category_weights', {}).get(self.pois[poi_idx].category, 0)
            for poi_idx in route
        )
        category_component = (category_matches / len(route)) * 100 if route else 0
        
        unique_categories = len(set(self.pois[poi_idx].category for poi_idx in route))
        diversity_component = (unique_categories / len(route)) * 100 if route else 0
        
        time_used = self._calculate_time(route)
        max_time = int(self.prefs.get('max_time', 480))
        time_utilization = min(100, (time_used / max_time) * 100)
        if time_utilization < 70:
            time_efficiency = time_utilization * 0.35
        else:
            time_efficiency = time_utilization
        
        proximity_component = self._proximity_component(route)
        elevation_penalty = self._elevation_component(route) if self.mobility_issues else 100.0

        if self.mobility_issues:
            scale = 1 - self.w_elevation
            fitness = (
                self.w_distance  * scale * distance_penalty +
                self.w_category  * scale * category_component +
                self.w_diversity * scale * diversity_component +
                self.w_time      * scale * time_efficiency +
                self.w_proximity * scale * proximity_component +
                self.w_elevation * elevation_penalty
            )
        else:
            fitness = (
                self.w_distance * distance_penalty +
                self.w_category * category_component +
                self.w_diversity * diversity_component +
                self.w_time * time_efficiency +
                self.w_proximity * proximity_component
            )
        
        return fitness * self._contextual_modifier(route)

    def _contextual_modifier(self, route: List[int]) -> float:
        """
        Multiplica o fitness por um factor contextual (0.7-1.3) baseado em
        has_children e mobility_issues. Penaliza categorias inapropriadas e
        bonifica categorias adequadas ao contexto do utilizador.
        """
        if not self.has_children and not self.mobility_issues:
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
            total += delta
        return max(0.7, min(1.3, 1.0 + total / len(route)))

    def _proximity_component(self, route: List[int]) -> float:
        """
        Penaliza rotas com POIs muito afastados do centro ou entre si.
        Devolve valor entre 0 e 100 (100 = todos dentro do raio ideal).
        """
        if not route or self.center_lat is None:
            return 100.0

        import math
        def haversine_km(lat1, lon1, lat2, lon2):
            R = 6371
            r = math.radians
            a = math.sin(r(lat2-lat1)/2)**2 + math.cos(r(lat1))*math.cos(r(lat2))*math.sin(r(lon2-lon1)/2)**2
            return R * 2 * math.asin(math.sqrt(a))

        dist_scores = []
        for poi_idx in route:
            poi = self.pois[poi_idx]
            d = haversine_km(self.center_lat, self.center_lon, poi.lat, poi.lon)
            score = max(0.0, 1.0 - (d / self.max_radius_km) ** 2)
            dist_scores.append(score)

        return (sum(dist_scores) / len(dist_scores)) * 100

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
            if self._debug_mode:
                print(f"   AVISO: Inviavel (tempo): {total_time:.0f} > {max_time}")
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
            if self._debug_mode:
                print(f"   AVISO: Inviavel (custo): {total_cost:.2f} > {max_cost}")
            return False

        # Hard constraint: pelo menos 1 POI de alojamento
        has_accommodation = any(
            self.pois[idx].category in ACCOMMODATION_CATEGORIES for idx in route
        )
        if not has_accommodation:
            return False

        return True

    def _calculate_time(self, route: List[int]) -> float:
        """Calcula tempo total da rota em minutos"""
        
        if not route:
            return 0
        
        total_time = sum(self.pois[poi_idx].duration for poi_idx in route)
        
        for pos in range(len(route)-1):
            travel_time = self.distances[route[pos]][route[pos+1]]
            total_time += travel_time
        
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

        total_gain = sum(
            self.elevation_matrix[route[i]][route[i+1]]
            for i in range(len(route) - 1)
        )

        if total_gain <= THRESHOLD_M:
            return 100.0
        score = max(0.0, 1.0 - (total_gain - THRESHOLD_M) / (MAX_GAIN_M - THRESHOLD_M))
        return score * 100
    
    def _parse_time(self, time_str: str) -> float:
        """Converte "09:30" para minutos desde meia-noite"""
        try:
            h, m = map(int, time_str.split(':'))
            return h * 60 + m
        except:
            return 540  # 09:00 default