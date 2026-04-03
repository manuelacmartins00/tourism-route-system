# src/optimizers/route_evaluator.py (FIX CONSTRAINTS RELAXADOS)

import numpy as np
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

class RouteEvaluator:
    """Avalia qualidade de rotas turísticas"""
    
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
        self.max_radius_km = user_prefs.get("max_radius_km", 20.0)
            
        # ✅ Flag para debug
        self._debug_mode = False
        self._empty_warning_shown = False
    
    def calculate_fitness(self, route: List[int]) -> float:
        """Calcula fitness da rota"""
        
        if not self.pois:
            if not self._empty_warning_shown:
                print("   ⚠️ [RouteEvaluator] Lista de POIs vazia!")
                self._empty_warning_shown = True
            return 0.0
        
        if route:
            max_valid_index = len(self.pois) - 1
            for idx in route:
                if idx > max_valid_index:
                    if self._debug_mode:
                        print(f"   ⚠️ Índice {idx} inválido! Max: {max_valid_index}")
                    return 0.0
        
        if not route or not self._is_feasible(route):
            return 0.0
        
        total_score = sum(self.pois[i].score for i in route)
        score_component = (total_score / len(route)) * 100 if route else 0
        
        total_distance = sum(
            self.distances[route[i]][route[i+1]] 
            for i in range(len(route)-1)
        ) if len(route) > 1 else 0
        distance_penalty = max(0, 100 - (total_distance / 50) * 100)
        
        category_matches = sum(
            self.prefs.get('category_weights', {}).get(self.pois[i].category, 0)
            for i in route
        )
        category_component = (category_matches / len(route)) * 100 if route else 0
        
        unique_categories = len(set(self.pois[i].category for i in route))
        diversity_component = (unique_categories / len(route)) * 100 if route else 0
        
        time_used = self._calculate_time(route)
        max_time = int(self.prefs.get('max_time', 480))
        # Recompensar rotas que usam entre 70% e 100% do tempo disponível
        time_utilization = min(100, (time_used / max_time) * 100)
        # Penalizar levemente rotas muito curtas (< 50% do tempo)
        if time_utilization < 70:
            time_efficiency = time_utilization * 0.35            ####ATENÇÃO TESTAR, PSO E GA NÃO SE PORTAM BEM COM VALORES ALTOS (ACIMA DE 0.5) NEM COM TIME UTIL < 50
        else:
            time_efficiency = time_utilization
        
        proximity_component = self._proximity_component(route)

        fitness = (
            self.w_distance * distance_penalty +
            self.w_category * category_component +
            self.w_diversity * diversity_component +
            self.w_time * time_efficiency +
            self.w_proximity * proximity_component
        )
        
        return fitness
    
    def _proximity_component(self, route: List[int]) -> float:
        """
        Penaliza rotas com POIs muito afastados do centro ou entre si.
        Devolve valor entre 0 e 100 (100 = todos dentro do raio ideal).
        """
        if not route or self.center_lat is None:
            return 100.0  # sem info de centro, não penaliza

        import math
        def haversine_km(lat1, lon1, lat2, lon2):
            R = 6371
            r = math.radians
            a = math.sin(r(lat2-lat1)/2)**2 + math.cos(r(lat1))*math.cos(r(lat2))*math.sin(r(lon2-lon1)/2)**2
            return R * 2 * math.asin(math.sqrt(a))

        # Penalização por distância ao centro
        dist_scores = []
        for i in route:
            poi = self.pois[i]
            d = haversine_km(self.center_lat, self.center_lon, poi.lat, poi.lon)
            # Score decresce linearmente até max_radius_km, depois é 0
            score = max(0.0, 1.0 - (d / self.max_radius_km) ** 2)
            dist_scores.append(score)

        return (sum(dist_scores) / len(dist_scores)) * 100
    
    def _is_feasible(self, route: List[int]) -> bool:
        """Verifica se a rota respeita constraints (RELAXADO)"""
        
        if not route:
            return False
        
        if not self.pois:
            return False
        
        max_valid_index = len(self.pois) - 1
        for idx in route:
            if idx > max_valid_index or idx < 0:
                return False
        
        # 1. Verificar tempo total
        total_time = self._calculate_time(route)
        max_time = int(self.prefs.get('max_time', 480))
        
        if total_time > max_time:
            if self._debug_mode:
                print(f"   ⚠️ Inviável (tempo): {total_time:.0f} > {max_time}")
            return False
        
        # 2. Verificar orçamento
        total_cost = sum(self.pois[i].cost for i in route)
        max_cost = float(self.prefs.get('max_cost', 1000))
        
        if total_cost > max_cost:
            if self._debug_mode:
                print(f"   ⚠️ Inviável (custo): {total_cost:.2f} > {max_cost}")
            return False
        
        # ✅ 3. HORÁRIOS REMOVIDOS (muito restritivo!)
        # Assumir que todos os POIs estão abertos durante o horário de visita
        # Se quiseres validar horários, adiciona aqui mas de forma mais flexível
        
        return True
    
    def _calculate_time(self, route: List[int]) -> float:
        """Calcula tempo total da rota em minutos"""
        
        if not route:
            return 0
        
        total_time = sum(self.pois[i].duration for i in route)
        
        # Tempo de deslocações (Haversine / 5km/h a pé)
        for i in range(len(route)-1):
            travel_time = self.distances[route[i]][route[i+1]]  # já em minutos
            total_time += travel_time
        
        return total_time
    
    def _parse_time(self, time_str: str) -> float:
        """Converte "09:30" para minutos desde meia-noite"""
        try:
            h, m = map(int, time_str.split(':'))
            return h * 60 + m
        except:
            return 540  # 09:00 default