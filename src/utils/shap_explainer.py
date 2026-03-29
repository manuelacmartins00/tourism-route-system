# src/utils/shap_explainer.py

import shap
import numpy as np
from typing import List, Dict
from src.optimizers.route_evaluator import POI, RouteEvaluator

class RouteExplainer:
    """
    Explica decisões do otimizador usando SHAP (KernelExplainer).

    Cada POI candidato é uma feature binária (1 = incluído, 0 = excluído).
    O SHAP calcula a contribuição marginal de cada POI para o fitness final,
    explicando o que o algoritmo de otimização "pensou" ao construir a rota.
    """

    def __init__(self,
                 pois: List[POI],
                 evaluator: RouteEvaluator):
        self.pois = pois
        self.evaluator = evaluator
        self.n_pois = len(pois)

    def _fitness_from_mask(self, mask_matrix: np.ndarray) -> np.ndarray:
        """
        Função de predição para o KernelExplainer.
        Recebe matriz (n_samples, n_pois) com valores 0/1.
        Devolve array (n_samples,) com fitness de cada amostra.
        """
        results = []
        for mask in mask_matrix:
            route = [i for i, included in enumerate(mask) if included == 1]
            if len(route) < 2:
                results.append(0.0)
            else:
                results.append(self.evaluator.calculate_fitness(route))
        return np.array(results)

    def explain_route(self, route: List[int]) -> Dict:
        """
        Gera explicação SHAP para uma rota.

        A instância explicada é o vector binário da rota atual.
        O background é o vector zero (nenhum POI selecionado).
        """
        if not route:
            return {'shap_values': {}, 'explanation': 'Rota vazia.'}

        # Vector binário da rota atual
        instance = np.zeros(self.n_pois)
        for idx in route:
            if idx < self.n_pois:
                instance[idx] = 1

        # Background: nenhum POI selecionado (baseline neutro)
        background = np.zeros((1, self.n_pois))

        # KernelExplainer — model-agnostic, funciona com qualquer otimizador
        explainer = shap.KernelExplainer(
            self._fitness_from_mask,
            background,
            silent=True
        )

        # nsamples=100 é suficiente para baseline
        shap_vals = explainer.shap_values(
            instance.reshape(1, -1),
            nsamples=100,
            silent=True
        )
        shap_array = shap_vals[0] if isinstance(shap_vals, list) else shap_vals.flatten()

        # Construir dicionário só com POIs da rota
        shap_by_poi = {}
        for idx in route:
            if idx < self.n_pois:
                poi = self.pois[idx]
                shap_by_poi[poi.name] = {
                    'shap_value': float(shap_array[idx]),
                    'category': poi.category,
                    'score': poi.score,
                    'cost': poi.cost,
                    'duration': poi.duration
                }

        explanation = self._generate_explanation(route, shap_by_poi)

        return {
            'shap_values': shap_by_poi,
            'explanation': explanation
        }

    def _generate_explanation(self, route: List[int], shap_by_poi: Dict) -> str:
        """
        Gera explicação textual ordenada por contribuição SHAP.
        POIs com SHAP positivo alto foram os mais determinantes
        para o algoritmo construir esta rota.
        """
        if not shap_by_poi:
            return 'Sem dados SHAP disponíveis.'

        sorted_pois = sorted(
            shap_by_poi.items(),
            key=lambda x: x[1]['shap_value'],
            reverse=True
        )

        total_positive = sum(
            v['shap_value'] for v in shap_by_poi.values()
            if v['shap_value'] > 0
        )

        explanation = "📊 Análise SHAP — Contribuição de cada POI:\n\n"
        explanation += "🔝 POIs mais determinantes para o algoritmo:\n"

        for i, (name, data) in enumerate(sorted_pois, 1):
            val = data['shap_value']
            pct = (val / total_positive * 100) if total_positive > 0 else 0
            direction = "▲" if val > 0 else "▼"
            explanation += (
                f"{i}. {name} ({data['category']})\n"
                f"   {direction} SHAP: {val:+.4f} | "
                f"Contribuição: {pct:.1f}% | "
                f"Score: {data['score']:.2f}\n"
            )

        explanation += f"\n💡 Total de POIs analisados: {len(route)}\n"
        explanation += (
            "   Valores SHAP positivos → POI aumentou o fitness da rota\n"
            "   Valores SHAP negativos → POI foi incluído apesar de reduzir fitness\n"
        )

        return explanation

