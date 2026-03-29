# src/optimizers/greedy_planner.py

import numpy as np
from typing import List, Dict
from .route_evaluator import POI, RouteEvaluator

class GreedyPlanner:
    """Baseline greedy algorithm"""
    
    def __init__(self,
                 pois: List[POI],
                 distance_matrix: np.ndarray,
                 evaluator: RouteEvaluator):
        
        self.pois = pois
        self.distances = distance_matrix
        self.evaluator = evaluator
    
    def optimize(self, start_poi: int = 0, strategy: str = "hybrid") -> Dict:
        """
        Strategies:
        - 'score': Maior score primeiro
        - 'nearest': Mais próximo primeiro
        - 'hybrid': Score / distância
        """
        
        route = [start_poi]
        visited = {start_poi}
        current = start_poi
        
        while True:
            best_poi = None
            best_value = -float('inf')
            
            for i in range(len(self.pois)):
                if i in visited:
                    continue
                
                # Testar viabilidade
                temp_route = route + [i]
                if not self.evaluator._is_feasible(temp_route):
                    continue
                
                # Calcular valor
                if strategy == "score":
                    value = self.pois[i].score
                elif strategy == "nearest":
                    dist = self.distances[current][i]
                    value = 1.0 / (dist + 0.001)
                elif strategy == "hybrid":
                    dist = self.distances[current][i]
                    value = self.pois[i].score / (dist + 0.001)
                else:
                    value = self.pois[i].score
                
                if value > best_value:
                    best_value = value
                    best_poi = i
            
            if best_poi is None:
                break
            
            route.append(best_poi)
            visited.add(best_poi)
            current = best_poi
        
        fitness = self.evaluator.calculate_fitness(route)
        
        return {
            'route': route,
            'fitness': fitness,
            'pois': [self.pois[i] for i in route],
            'algorithm': f'Greedy-{strategy}'
        }