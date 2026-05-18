# src/optimizers/tourism_psoa.py

import numpy as np
import random
from typing import List, Dict
from .route_evaluator import POI, RouteEvaluator

class TourismPSOA:
    """
    Particle Swarm Optimization Algorithm para rotas turisticas
    
    Cada particula representa uma rota (sequencia de POIs)
    Particulas movem-se no espaco de solucoes seguindo:
    - Melhor posicao pessoal (pbest)
    - Melhor posicao global (gbest)
    """
    
    def __init__(self,
                 pois: List[POI],
                 distance_matrix: np.ndarray,
                 evaluator: RouteEvaluator,
                 n_particles: int = 30,
                 n_iterations: int = 50,
                 w: float = 0.7,        # Inercia
                 c1: float = 1.5,       # Cognitive component
                 c2: float = 1.5):      # Social component
        
        self.pois = pois
        self.n_pois = len(pois)
        self.distances = distance_matrix
        self.evaluator = evaluator
        
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        
        # Parametros PSO
        self.w = w      # Inercia (exploracao vs exploitation)
        self.c1 = c1    # Peso da melhor posicao pessoal
        self.c2 = c2    # Peso da melhor posicao global
    
    def optimize(self, start_poi: int = 0) -> Dict:
        """Executa otimizacao PSO"""
        
        # Inicializar enxame
        particles = self._initialize_swarm(start_poi)
        
        # Velocidades (mudancas nas rotas)
        velocities = [self._random_velocity() for _ in range(self.n_particles)]
        
        # Melhores posicoes pessoais
        pbest_positions = [p.copy() for p in particles]
        pbest_fitness = [self.evaluator.calculate_fitness(p) for p in particles]
        
        # Melhor posicao global
        gbest_idx = np.argmax(pbest_fitness)
        gbest_position = particles[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]
        
        fitness_history = []
        
        # Iteracoes PSO
        for iteration in range(self.n_iterations):
            
            for i in range(self.n_particles):
                # Calcular fitness atual
                fitness = self.evaluator.calculate_fitness(particles[i])
                
                # Atualizar pbest
                if fitness > pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest_positions[i] = particles[i].copy()
                
                # Atualizar gbest
                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = particles[i].copy()
                
                # Atualizar velocidade
                velocities[i] = self._update_velocity(
                    velocities[i],
                    particles[i],
                    pbest_positions[i],
                    gbest_position
                )
                
                # Atualizar posicao
                particles[i] = self._update_position(
                    particles[i],
                    velocities[i]
                )
            
            # Logging
            avg_fitness = np.mean([self.evaluator.calculate_fitness(p) for p in particles])
            fitness_history.append(avg_fitness)
            
            if iteration % 10 == 0:
                print(f"  PSO Iter {iteration}: Best={gbest_fitness:.2f}, Avg={avg_fitness:.2f}")
        
        return {
            'route': gbest_position,
            'fitness': gbest_fitness,
            'pois': [self.pois[i] for i in gbest_position],
            'fitness_history': fitness_history,
            'algorithm': 'PSOA'
        }
    
    def _initialize_swarm(self, start_poi: int) -> List[List[int]]:
        """Inicializa enxame com rotas aleatorias viaveis"""
        
        particles = []
        
        for _ in range(self.n_particles):
            route = self._generate_random_route(start_poi)
            particles.append(route)
        
        return particles
    
    def _generate_random_route(self, start_poi: int) -> List[int]:
        """Gera rota inicial aleatoria mas viavel"""
        
        available = list(range(self.n_pois))
        available.remove(start_poi)
        random.shuffle(available)
        
        route = [start_poi]
        
        for poi_idx in available:
            temp_route = route + [poi_idx]
            
            if self.evaluator._is_feasible(temp_route):
                route.append(poi_idx)
        
        return route
    
    def _random_velocity(self) -> List[tuple]:
        """
        Velocidade = lista de operacoes (swap, insert, remove)
        """
        n_ops = random.randint(1, 3)
        velocity = []
        
        for _ in range(n_ops):
            op_type = random.choice(['swap', 'insert', 'remove'])
            
            if op_type == 'swap':
                velocity.append(('swap', random.randint(1, 5), random.randint(1, 5)))
            elif op_type == 'insert':
                velocity.append(('insert', random.randint(0, self.n_pois-1), random.randint(1, 5)))
            else:  # remove
                velocity.append(('remove', random.randint(1, 5)))
        
        return velocity
    
    def _update_velocity(self, 
                        velocity: List[tuple],
                        position: List[int],
                        pbest: List[int],
                        gbest: List[int]) -> List[tuple]:
        """
        Atualiza velocidade baseado em:
        - Inercia (velocidade atual)
        - Componente cognitiva (direcao para pbest)
        - Componente social (direcao para gbest)
        """
        
        new_velocity = []
        
        # Inercia
        if random.random() < self.w:
            new_velocity.extend(velocity[:max(1, int(len(velocity) * self.w))])
        
        # Componente cognitiva (move para pbest)
        if random.random() < self.c1:
            ops = self._path_to_target(position, pbest)
            new_velocity.extend(ops[:max(1, int(len(ops) * self.c1))])
        
        # Componente social (move para gbest)
        if random.random() < self.c2:
            ops = self._path_to_target(position, gbest)
            new_velocity.extend(ops[:max(1, int(len(ops) * self.c2))])
        
        return new_velocity
    
    def _path_to_target(self, current: List[int], target: List[int]) -> List[tuple]:
        """
        Calcula operacoes para transformar current em target
        (versao simplificada - retorna swaps)
        """
        ops = []
        
        for i in range(1, min(len(current), len(target))):
            if current[i] != target[i]:
                # Encontrar onde target[i] esta em current
                try:
                    j = current.index(target[i], i)
                    ops.append(('swap', i, j))
                except ValueError:
                    # target[i] nao esta em current
                    ops.append(('insert', target[i], i))
        
        return ops
    
    def _update_position(self, position: List[int], velocity: List[tuple]) -> List[int]:
        """
        Aplica velocidade (operacoes) a posicao (rota)
        """
        
        new_position = position.copy()
        
        for op in velocity:
            if op[0] == 'swap' and len(new_position) > max(op[1], op[2]):
                # Swap
                i, j = op[1] % len(new_position), op[2] % len(new_position)
                if i != 0 and j != 0 and i < len(new_position) and j < len(new_position):
                    new_position[i], new_position[j] = new_position[j], new_position[i]
            
            elif op[0] == 'insert' and len(new_position) < self.n_pois:
                # Insert
                poi_id, pos = op[1], op[2] % (len(new_position) + 1)
                if poi_id not in new_position and poi_id < self.n_pois:
                    new_position.insert(pos, poi_id)
            
            elif op[0] == 'remove' and len(new_position) > 2:
                # Remove
                pos = op[1] % len(new_position)
                if pos != 0:  # Nao remover o POI inicial
                    new_position.pop(pos)
        
        # Validar rota final
        if self.evaluator._is_feasible(new_position):
            return new_position
        else:
            return position  # Retornar posicao original se inviavel