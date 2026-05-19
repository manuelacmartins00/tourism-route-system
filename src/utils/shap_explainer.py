# src/utils/shap_explainer.py

import shap
import numpy as np
from typing import List, Dict
from src.optimizers.route_evaluator import POI, RouteEvaluator

class RouteExplainer:
    """
    Explica decisoes do otimizador usando SHAP (KernelExplainer).

    Cada POI candidato e uma feature binaria (1 = incluido, 0 = excluido).
    O SHAP calcula a contribuicao marginal de cada POI para o fitness final,
    explicando o que o algoritmo de otimizacao "pensou" ao construir a rota.
    """

    def __init__(self,
                 pois: List[POI],
                 evaluator: RouteEvaluator):
        self.pois = pois
        self.evaluator = evaluator
        self.n_pois = len(pois)

    def _fitness_from_mask(self, mask_matrix: np.ndarray) -> np.ndarray:
        """
        Funcao de predicao para o KernelExplainer.
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
        Gera explicacao SHAP para uma rota.

        A instancia explicada e o vector binario da rota atual.
        O background e o vector zero (nenhum POI selecionado).
        """
        if not route:
            return {'shap_values': {}, 'explanation': 'Rota vazia.'}

        # Vector binario da rota atual
        instance = np.zeros(self.n_pois)
        for idx in route:
            if idx < self.n_pois:
                instance[idx] = 1

        # Background: nenhum POI selecionado (baseline neutro)
        background = np.zeros((1, self.n_pois))

        # KernelExplainer - model-agnostic, funciona com qualquer otimizador
        explainer = shap.KernelExplainer(
            self._fitness_from_mask,
            background,
            silent=True
        )

        # nsamples=20: rapido para producao (~10x menos amostras que 100,
        # ainda representativo para destacar os POIs mais determinantes)
        shap_vals = explainer.shap_values(
            instance.reshape(1, -1),
            nsamples=20,
            silent=True
        )
        shap_array = shap_vals[0] if isinstance(shap_vals, list) else shap_vals.flatten()

        # Categorias com efeito contextual (espelha route_evaluator)
        _CHILDREN_PENALTY  = {"bares_e_discotecas", "casinos", "turismo_activo"}
        _CHILDREN_BONUS    = {"espacos_verdes", "parques_e_reservas", "parques_de_diversao",
                               "zoos_e_aquarios", "ciencia_e_conhecimento"}
        _MOBILITY_PENALTY  = {"turismo_activo", "campos", "parques_e_reservas",
                               "parques_de_diversao", "grutas"}
        _MOBILITY_BONUS    = {"restaurantes_e_cafes", "monumentos", "museus_e_palacios",
                               "espacos_verdes", "termas", "ciencia_e_conhecimento", "talassoterapia"}

        has_children    = getattr(self.evaluator, 'has_children', False)
        mobility_issues = getattr(self.evaluator, 'mobility_issues', False)

        # Construir dicionario so com POIs da rota
        shap_by_poi = {}
        for idx in route:
            if idx < self.n_pois:
                poi = self.pois[idx]
                reasons = []
                if has_children:
                    if poi.category in _CHILDREN_PENALTY:
                        reasons.append("penalizado (viagem com crianças)")
                    elif poi.category in _CHILDREN_BONUS:
                        reasons.append("bónus (adequado para crianças)")
                if mobility_issues:
                    if poi.category in _MOBILITY_PENALTY:
                        reasons.append("penalizado (mobilidade reduzida)")
                    elif poi.category in _MOBILITY_BONUS:
                        reasons.append("bónus (acessível com mobilidade reduzida)")
                shap_by_poi[poi.name] = {
                    'shap_value': float(shap_array[idx]),
                    'category': poi.category,
                    'cost': poi.cost,
                    'duration': poi.duration,
                    'contextual_reason': ', '.join(reasons) if reasons else None,
                }

        explanation = self._generate_explanation(route, shap_by_poi)

        return {
            'shap_values': shap_by_poi,
            'explanation': explanation
        }

    def _generate_explanation(self, route: List[int], shap_by_poi: Dict) -> str:
        """
        Gera explicacao textual ordenada por contribuicao SHAP.
        POIs com SHAP positivo alto foram os mais determinantes
        para o algoritmo construir esta rota.
        """
        if not shap_by_poi:
            return 'Sem dados SHAP disponiveis.'

        sorted_pois = sorted(
            shap_by_poi.items(),
            key=lambda x: x[1]['shap_value'],
            reverse=True
        )

        total_positive = sum(
            v['shap_value'] for v in shap_by_poi.values()
            if v['shap_value'] > 0
        )

        explanation = "Analise SHAP -- Contribuicao de cada POI:\n\n"
        explanation += "POIs mais determinantes para o algoritmo:\n"

        for i, (name, data) in enumerate(sorted_pois, 1):
            val = data['shap_value']
            pct = (val / total_positive * 100) if total_positive > 0 else 0
            direction = "+" if val > 0 else "-"
            explanation += (
                f"{i}. {name} ({data['category']})\n"
                f"   {direction} SHAP: {val:+.4f} | "
                f"Contribuicao: {pct:.1f}%\n"
            )

        explanation += f"\nTotal de POIs analisados: {len(route)}\n"
        explanation += (
            "   Valores SHAP positivos -> POI aumentou o fitness da rota\n"
            "   Valores SHAP negativos -> POI foi incluido apesar de reduzir fitness\n"
        )

        return explanation

