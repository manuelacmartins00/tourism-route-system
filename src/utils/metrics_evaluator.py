# src/utils/metrics_evaluator.py (VERSÃO CORRIGIDA)

import numpy as np
from typing import Dict, List

class MetricsEvaluator:
    """
    Calcula métricas de avaliação para rotas turísticas
    
    Métricas implementadas:
    - Coverage, Constraint Satisfaction
    - Eficiência (POIs/custo, POIs/tempo)
    - Diversidade (Shannon entropy)
    """
    
    def __init__(self):
        pass
    
    def calculate_metrics(self, result: Dict) -> Dict:
        """
        Calcula todas as métricas para um resultado
        
        Args:
            result: Dicionário com resultado do plan_route()
        
        Returns:
            Dict com métricas calculadas
        """
        
        route = result['route']
        preferences = result['preferences']
        optimization = result['optimization']
        
        if not route:
            return self._empty_metrics()
        
        # Valores reais
        total_cost = sum(poi['cost'] for poi in route)
        total_time = sum(poi['duration'] for poi in route)
        n_pois = len(route)
        
        # Valores esperados/ideais
        expected_cost = preferences['max_cost']
        expected_time = preferences['max_time']
        n_candidates = optimization['n_candidates']
        

        # ========== QUALIDADE DA ROTA ==========
        
        # Fitness Score (0-100)
        fitness_score = optimization['fitness']
        
        # Coverage - % de POIs candidatos selecionados
        coverage = (n_pois / n_candidates * 100) if n_candidates > 0 else 0
        
        # ✅ CORRIGIDO: Constraint Satisfaction
        # Mede quão bem os recursos foram utilizados (não o quanto sobrou!)
        cost_usage = min(1.0, total_cost / expected_cost) if expected_cost > 0 else 0
        time_usage = min(1.0, total_time / expected_time) if expected_time > 0 else 0
        
        # Penalizar se ultrapassar
        if total_cost > expected_cost:
            cost_usage = max(0, 2 - (total_cost / expected_cost))  # Penalidade
        if total_time > expected_time:
            time_usage = max(0, 2 - (total_time / expected_time))  # Penalidade
        
        constraint_satisfaction = (cost_usage + time_usage) / 2 * 100
        
        # ========== EFICIÊNCIA ==========
        
        # POIs por Euro
        pois_per_euro = n_pois / total_cost if total_cost > 0 else 0
        
        # POIs por Hora
        pois_per_hour = n_pois / (total_time / 60) if total_time > 0 else 0
        
        # ========== DIVERSIDADE ==========
        
        # Categorias únicas
        categories = [poi['category'] for poi in route]
        unique_categories = len(set(categories))
        
        # Índice de Diversidade (Shannon Entropy)
        diversity_index = self._calculate_diversity_index(categories)
        
        # ========== UTILIZAÇÃO DE RECURSOS ==========
        
        cost_utilization = (total_cost / expected_cost * 100) if expected_cost > 0 else 0
        time_utilization = (total_time / expected_time * 100) if expected_time > 0 else 0
        
        # ========== RESULTADO FINAL ==========
        
        return {

            # Qualidade
            'fitness_score': fitness_score,
            'coverage': coverage,
            'constraint_satisfaction': constraint_satisfaction,
            
            # Eficiência
            'total_cost': total_cost,
            'total_time': total_time,
            'pois_per_euro': pois_per_euro,
            'pois_per_hour': pois_per_hour,
            
            # Utilização
            'cost_utilization': cost_utilization,
            'time_utilization': time_utilization,
            
            # Diversidade
            'unique_categories': unique_categories,
            'diversity_index': diversity_index,
            
            # Contexto
            'n_pois': n_pois,
            'n_candidates': n_candidates
        }
    
    def _calculate_diversity_index(self, categories: List[str]) -> float:
        """
        Calcula índice de diversidade Shannon
        
        H = -Σ(pi * log(pi))
        onde pi = proporção da categoria i
        """
        if not categories:
            return 0.0
        
        from collections import Counter
        counts = Counter(categories)
        
        n = len(categories)
        proportions = [count / n for count in counts.values()]
        
        entropy = -sum(p * np.log(p) for p in proportions if p > 0)
        
        return entropy
    
    def _empty_metrics(self) -> Dict:
        """Retorna métricas vazias quando não há rota"""
        return {
            'fitness_score': 0.0,
            'coverage': 0.0,
            'constraint_satisfaction': 0.0,
            'total_cost': 0.0,
            'total_time': 0.0,
            'pois_per_euro': 0.0,
            'pois_per_hour': 0.0,
            'cost_utilization': 0.0,
            'time_utilization': 0.0,
            'unique_categories': 0,
            'diversity_index': 0.0,
            'n_pois': 0,
            'n_candidates': 0
        }
    
    def compare_algorithms(self, results_dict: Dict[str, Dict]) -> Dict:
        """
        Compara métricas de múltiplos algoritmos
        
        Args:
            results_dict: {algoritmo: result} para cada algoritmo
        
        Returns:
            Dict com estatísticas comparativas
        """
        
        metrics_by_algo = {}
        
        for algo, result in results_dict.items():
            if result is None:
                continue
            metrics_by_algo[algo] = self.calculate_metrics(result)
        
        if not metrics_by_algo:
            return {}
        
        # Calcular estatísticas agregadas

        all_fitness = [m['fitness_score'] for m in metrics_by_algo.values()]
        
        comparison = {
            'individual_metrics': metrics_by_algo,
            'aggregate': {
                'mean_fitness': np.mean(all_fitness),
                'std_fitness': np.std(all_fitness)
            },
            'best_algorithm': {
                'by_fitness': max(metrics_by_algo.items(), key=lambda x: x[1]['fitness_score'])[0],
                'by_mae': min(metrics_by_algo.items(), key=lambda x: x[1]['mae'])[0],
                'by_rmse': min(metrics_by_algo.items(), key=lambda x: x[1]['rmse'])[0]
            }
        }
        
        return comparison