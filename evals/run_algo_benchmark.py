"""
evals/run_algo_benchmark.py
===========================
Grid-search benchmark: corre cada configuração de algoritmo em todos os fixtures,
captura fitness + 6 componentes individuais por run.

Uso:
    python evals/run_algo_benchmark.py \
        --fixtures data/bench_fixtures_direct \
        --out      outputs/bench_results.csv \
        [--seeds 5] \
        [--algos ACO_S ACO_L GA_S GA_L PSO_S PSO_L GREEDY]

Grid search: 2 configurações por algoritmo estocástico (S=leve, L=produção).
Friedman + Nemenyi no R usa as 7 "algoritmos" resultantes.
"""
import os
import sys
import json
import time
import argparse
import csv
import traceback
import contextlib
import io
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimizers.route_evaluator import POI, RouteEvaluator
from src.optimizers.tourism_aco    import TourismACO
from src.optimizers.tourism_ga     import TourismGA
from src.optimizers.tourism_psoa   import TourismPSOA
from src.optimizers.greedy_planner import GreedyPlanner


# ── Grid de hiperparâmetros ───────────────────────────────────────────────────
# S = configuração leve; L = configuração de produção
ALGO_CONFIGS = {
    "ACO_S":  {"n_ants": 10, "n_iterations": 30},
    "ACO_L":  {"n_ants": 30, "n_iterations": 100},
    "GA_S":   {"population_size": 25, "n_generations": 15},
    "GA_L":   {"population_size": 50, "n_generations": 30},
    "PSO_S":  {"n_particles": 10, "n_iterations": 20},
    "PSO_L":  {"n_particles": 20, "n_iterations": 30},
    "GREEDY": {},
}

# Mapeamento do nome de config para a classe base do algoritmo
ALGO_BASE = {
    "ACO_S": "ACO", "ACO_L": "ACO",
    "GA_S":  "GA",  "GA_L":  "GA",
    "PSO_S": "PSO", "PSO_L": "PSO",
    "GREEDY": "GREEDY",
}

FIELDNAMES = [
    "scenario_id", "profile", "query_short",
    "algo", "seed",
    # fitness agregado
    "fitness",
    # 6 componentes individuais
    "time_efficiency", "proximity_component", "diversity_component",
    "distance_penalty", "cat_indata_comp", "cat_general_comp",
    "contextual_modifier",
    # metadados
    "n_pois_selected", "elapsed_s",
    "n_pois_input", "selected_algo_production",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", default="data/bench_fixtures_direct")
    p.add_argument("--out",      default="outputs/bench_results.csv")
    p.add_argument("--seeds",    type=int, default=5,
                   help="Número de runs independentes por (fixture × config)")
    p.add_argument("--algos",    nargs="+",
                   default=list(ALGO_CONFIGS.keys()),
                   choices=list(ALGO_CONFIGS.keys()))
    return p.parse_args()


def load_fixture(fp: Path):
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)

    pois = [
        POI(
            id=p["id"], name=p["name"],
            lat=p["lat"], lon=p["lon"],
            category=p["category"], score=p["score"],
            duration=p["duration"],
            opening_time=p.get("opening_time", "09:00"),
            closing_time=p.get("closing_time", "18:00"),
            cost=p["cost"],
        )
        for p in data["pois"]
    ]

    dist_matrix = np.array(data["distance_matrix"], dtype=float)

    prefs = dict(data["user_prefs"])
    if "start_location" in prefs and isinstance(prefs["start_location"], list):
        prefs["start_location"] = tuple(prefs["start_location"])
    if "all_geos" in prefs and prefs["all_geos"]:
        prefs["all_geos"] = [list(g) for g in prefs["all_geos"]]

    return pois, dist_matrix, prefs, data


def build_optimizer(algo_key: str, pois, dist_matrix, evaluator):
    cfg = ALGO_CONFIGS[algo_key]
    base = ALGO_BASE[algo_key]
    if base == "ACO":
        return TourismACO(pois, dist_matrix, evaluator, **cfg)
    elif base == "GA":
        return TourismGA(pois, dist_matrix, evaluator, **cfg)
    elif base == "PSO":
        return TourismPSOA(pois, dist_matrix, evaluator, **cfg)
    else:
        return GreedyPlanner(pois, dist_matrix, evaluator)


@contextlib.contextmanager
def _silent():
    """Suprime stdout durante a execução do optimizador (prints de iterações)."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def run_one(algo_key: str, pois, dist_matrix, evaluator, seed: int):
    opt = build_optimizer(algo_key, pois, dist_matrix, evaluator)
    t0  = time.perf_counter()
    with _silent():
        if ALGO_BASE[algo_key] == "GREEDY":
            result = opt.optimize()
        else:
            result = opt.optimize(seed=seed)
    elapsed = time.perf_counter() - t0

    route   = result.get("route") or []
    fitness = result.get("fitness", 0.0) or 0.0

    # Componentes individuais
    comps = evaluator.calculate_fitness_components(route) if route else {}
    return fitness, route, elapsed, comps


def main():
    args = parse_args()

    fixture_dir = Path(args.fixtures)
    out_path    = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixtures = sorted(fixture_dir.glob("*.json"))
    if not fixtures:
        sys.exit(f"Nenhum fixture encontrado em {fixture_dir}")

    print(f"[bench] {len(fixtures)} fixtures  |  configs: {args.algos}  |  seeds: {args.seeds}")
    total_runs = len(fixtures) * len(args.algos) * args.seeds
    print(f"[bench] Total runs: {total_runs}\n")

    write_header = not out_path.exists()
    run_count = 0
    t_global  = time.perf_counter()

    with open(out_path, "a", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for fp in fixtures:
            try:
                pois, dist_matrix, prefs, meta = load_fixture(fp)
            except Exception as e:
                print(f"  SKIP {fp.name}: {e}")
                continue

            sid        = meta.get("scenario_id", fp.stem)
            profile    = meta.get("profile", "?")
            query_sh   = meta.get("query", "")[:40]
            prod_algo  = meta.get("selected_algo", "?")

            evaluator = RouteEvaluator(pois, dist_matrix, prefs)

            for algo_key in args.algos:
                fitnesses = []
                for seed in range(args.seeds):
                    try:
                        fitness, route, elapsed, comps = run_one(
                            algo_key, pois, dist_matrix, evaluator, seed)
                    except Exception as e:
                        print(f"  ERRO {sid} {algo_key} seed={seed}: {e}")
                        traceback.print_exc()
                        fitness, route, elapsed, comps = 0.0, [], 0.0, {}

                    writer.writerow({
                        "scenario_id":              sid,
                        "profile":                  profile,
                        "query_short":              query_sh,
                        "algo":                     algo_key,
                        "seed":                     seed,
                        "fitness":                  round(fitness, 4),
                        "time_efficiency":          round(comps.get("time_efficiency",   0), 2),
                        "proximity_component":      round(comps.get("proximity_component", 0), 2),
                        "diversity_component":      round(comps.get("diversity_component", 0), 2),
                        "distance_penalty":         round(comps.get("distance_penalty",   0), 2),
                        "cat_indata_comp":          round(comps.get("cat_indata_comp",    0), 2),
                        "cat_general_comp":         round(comps.get("cat_general_comp",   0), 2),
                        "contextual_modifier":      round(comps.get("contextual_modifier", 1), 4),
                        "n_pois_selected":          len(route),
                        "elapsed_s":                round(elapsed, 3),
                        "n_pois_input":             len(pois),
                        "selected_algo_production": prod_algo,
                    })
                    fitnesses.append(fitness)
                    run_count += 1
                    csvf.flush()

                mean_f   = sum(fitnesses) / len(fitnesses) if fitnesses else 0
                elapsed_total = time.perf_counter() - t_global
                runs_left     = total_runs - run_count
                eta_s         = (elapsed_total / run_count * runs_left) if run_count else 0
                eta_h         = eta_s / 3600
                print(f"  {sid[:20]:20s} {algo_key:7s}  mean={mean_f:.2f}  "
                      f"({run_count}/{total_runs})  ETA={eta_h:.1f}h")

    print(f"\n[bench] Concluído. Resultados em: {out_path}")


if __name__ == "__main__":
    main()
