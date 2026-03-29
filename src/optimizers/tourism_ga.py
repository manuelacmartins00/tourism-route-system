# src/optimizers/tourism_ga.py
# Adapta o teu GA_A.py para usar RouteEvaluator

import numpy as np
import random
from typing import List, Dict
from .route_evaluator import POI, RouteEvaluator

class TourismGA:
    """Genetic Algorithm para rotas turísticas"""
    
    def __init__(self,
                 pois: List[POI],
                 distance_matrix: np.ndarray,
                 evaluator: RouteEvaluator,
                 population_size: int = 50,
                 n_generations: int = 30,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.2,
                 tournament_size: int = 3):
        
        self.pois = pois
        self.n_pois = len(pois)
        self.distances = distance_matrix
        self.evaluator = evaluator
        
        self.pop_size = population_size
        self.n_gen = n_generations
        self.cx_prob = crossover_prob
        self.mut_prob = mutation_prob
        self.tournament_size = tournament_size
    
    def optimize(self, start_poi: int = 0) -> Dict:
        """Executa GA"""
        
        # População inicial
        population = [self._generate_random_route(start_poi) for _ in range(self.pop_size)]
        
        best_route = None
        best_fitness = -float('inf')
        fitness_history = []
        
        for gen in range(self.n_gen):
            # Avaliar fitness
            fitnesses = [self.evaluator.calculate_fitness(ind) for ind in population]
            
            # Atualizar melhor
            max_fitness = max(fitnesses)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_route = population[fitnesses.index(max_fitness)].copy()
            
            fitness_history.append(np.mean(fitnesses))
            
            if gen % 5 == 0:
                print(f"  GA Gen {gen}: Best={best_fitness:.2f}, Avg={np.mean(fitnesses):.2f}")
            
            # Nova geração
            new_population = []
            
            while len(new_population) < self.pop_size:
                # Seleção
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                
                # Crossover
                if random.random() < self.cx_prob:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutação
                if random.random() < self.mut_prob:
                    child1 = self._mutate(child1)
                if random.random() < self.mut_prob:
                    child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.pop_size]
        
        return {
            'route': best_route,
            'fitness': best_fitness,
            'pois': [self.pois[i] for i in best_route] if best_route else [],
            'fitness_history': fitness_history,
            'algorithm': 'GA'
        }
    
    def _generate_random_route(self, start_poi: int) -> List[int]:
        """Gera rota inicial viável"""
        available = list(range(self.n_pois))
        available.remove(start_poi)
        random.shuffle(available)
        
        route = [start_poi]
        for poi_idx in available:
            temp_route = route + [poi_idx]
            if self.evaluator._is_feasible(temp_route):
                route.append(poi_idx)
            if len(route) >= 10:  # max POIs
                break
        
        return route
    
    def _tournament_selection(self, population: List, fitnesses: List) -> List[int]:
        """Seleção por torneio"""
        tournament = random.sample(list(zip(population, fitnesses)), self.tournament_size)
        winner = max(tournament, key=lambda x: x[1])
        return winner[0].copy()
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> tuple:
        """Ordered Crossover"""
        size = min(len(parent1), len(parent2))
        if size <= 2:
            return parent1.copy(), parent2.copy()
        
        cx_point1 = random.randint(1, size - 1)
        cx_point2 = random.randint(1, size - 1)
        if cx_point1 > cx_point2:
            cx_point1, cx_point2 = cx_point2, cx_point1
        
        child1 = [None] * size
        child2 = [None] * size
        
        child1[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]
        child2[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]
        
        self._fill_offspring(child1, parent2, cx_point2)
        self._fill_offspring(child2, parent1, cx_point2)
        
        return child1, child2
    
    def _fill_offspring(self, offspring, parent, start_pos):
        """Preencher offspring no OX"""
        parent_idx = start_pos
        offspring_idx = start_pos
        
        while None in offspring:
            if parent[parent_idx % len(parent)] not in offspring:
                offspring[offspring_idx % len(offspring)] = parent[parent_idx % len(parent)]
                offspring_idx += 1
            parent_idx += 1
    
    def _mutate(self, individual: List[int]) -> List[int]:
        """Swap Mutation"""
        if len(individual) < 3:
            return individual
        
        idx1 = random.randint(1, len(individual) - 1)
        idx2 = random.randint(1, len(individual) - 1)
        
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        
        # Reverter se inviável
        if not self.evaluator._is_feasible(individual):
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        
        return individual