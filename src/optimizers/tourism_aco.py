# src/optimizers/tourism_aco.py (FIX PARÂMETROS)

import numpy as np
from typing import List, Dict
from .route_evaluator import POI, RouteEvaluator

class TourismACO:
    """Ant Colony Optimization para rotas turísticas"""
    
    def __init__(self,
                 pois: List[POI],
                 distance_matrix: np.ndarray,
                 evaluator: RouteEvaluator,
                 n_ants: int = 30,        # ✅ Era 20, aumentado para 30
                 n_iterations: int = 100,  # ✅ Era 50, aumentado para 100
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 evaporation: float = 0.5,
                 q0: float = 0.9):
        
        self.pois = pois
        self.n_pois = len(pois)
        self.distances = distance_matrix
        self.evaluator = evaluator
        
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.q0 = q0
        
        self.pheromone = np.ones((self.n_pois, self.n_pois)) * 0.1
        
        self.heuristic = np.zeros((self.n_pois, self.n_pois))
        for i in range(self.n_pois):
            for j in range(self.n_pois):
                if i != j and self.distances[i][j] > 0:
                    self.heuristic[i][j] = 1.0 / self.distances[i][j]
    
    def optimize(self, start_poi: int = 0) -> Dict:
        """Executa otimização ACO"""
        
        best_route = None
        best_fitness = -float('inf')
        fitness_history = []
        
        for iteration in range(self.n_iterations):
            iteration_routes = []
            
            for ant in range(self.n_ants):
                route = self._construct_solution(start_poi)
                fitness = self.evaluator.calculate_fitness(route)
                
                iteration_routes.append((route, fitness))
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_route = route.copy()
            
            self._update_pheromones(iteration_routes)
            
            avg_fitness = np.mean([f for _, f in iteration_routes])
            fitness_history.append(avg_fitness)
            
            if iteration % 10 == 0:
                print(f"  ACO Iter {iteration}: Best={best_fitness:.2f}, Avg={avg_fitness:.2f}")
        
        return {
            'route': best_route,
            'fitness': best_fitness,
            'pois': [self.pois[i] for i in best_route] if best_route else [],
            'fitness_history': fitness_history,
            'algorithm': 'ACO'
        }
    
    def _construct_solution(self, start_poi: int) -> List[int]:
        """Uma formiga constrói uma solução"""
        
        visited = {start_poi}
        route = [start_poi]
        current = start_poi
        
        # ✅ Aumentar tentativas máximas
        max_attempts = self.n_pois * 3  # Era n_pois * 2
        attempts = 0
        stagnation_count = 0
        
        while attempts < max_attempts:
            next_poi = self._select_next_poi(current, visited, route)
            
            if next_poi is None:
                stagnation_count += 1
                if stagnation_count > 5:  # ✅ Dar mais chances
                    break
                attempts += 1
                continue
            
            route.append(next_poi)
            visited.add(next_poi)
            current = next_poi
            attempts = 0
            stagnation_count = 0
        
        return route
    
    def _select_next_poi(self, current: int, visited: set, current_route: List[int]) -> int:
        """Seleciona próximo POI"""
        
        candidates = []
        probabilities = []
        
        for j in range(self.n_pois):
            if j in visited:
                continue
            
            temp_route = current_route + [j]
            if not self.evaluator._is_feasible(temp_route):
                continue
            
            tau = self.pheromone[current][j] ** self.alpha
            eta = self.heuristic[current][j] ** self.beta
            score_bonus = self.pois[j].score + 0.5
            
            prob = tau * eta * score_bonus
            
            candidates.append(j)
            probabilities.append(prob)
        
        if not candidates:
            return None
        
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()
        
        if np.random.random() < self.q0:
            return candidates[np.argmax(probabilities)]
        else:
            return np.random.choice(candidates, p=probabilities)
    
    def _update_pheromones(self, all_routes: List[tuple]):
        """Atualiza feromonas"""
        
        self.pheromone *= (1 - self.evaporation)
        
        for route, fitness in all_routes:
            if fitness > 0:
                delta = fitness / 100.0
                
                for i in range(len(route) - 1):
                    self.pheromone[route[i]][route[i+1]] += delta
                    self.pheromone[route[i+1]][route[i]] += delta
        
        self.pheromone = np.clip(self.pheromone, 0.01, 10.0)