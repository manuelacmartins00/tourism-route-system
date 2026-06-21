"""
evals/run_algo_benchmark.py
===========================
Para cada fixture capturado por capture_fixtures.py, corre os quatro
algoritmos (ACO, GA, PSO, GREEDY) com N_SEEDS seeds independentes e
escreve os resultados em bench_results.csv.

Uso:
    python evals/run_algo_benchmark.py \
        --fixtures data/bench_fixtures \
        --out      outputs/bench_results.csv \
        [--seeds 20] \
        [--algos ACO GA PSO GREEDY]

Parâmetros dos algoritmos: idênticos aos usados em produção
(main_system.py linhas 878-888), para uma comparação justa.
"""
import os
import sys
import json
import time
import argparse
import csv
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimizers.route_evaluator import POI, RouteEvaluator
from src.optimizers.tourism_aco    import TourismACO
from src.optimizers.tourism_ga     import TourismGA
from src.optimizers.tourism_psoa   import TourismPSOA
from src.optimizers.greedy_planner import GreedyPlanner


# ── Parâmetros de produção ────────────────────────────────────────────────────
ALGO_CONFIGS = {
    "ACO":    {"n_ants": 30,      "n_iterations": 100},
    "GA":     {"population_size": 50, "n_generations": 30},
    "PSO":    {"n_particles": 20, "n_iterations": 30},
    "GREEDY": {},
}

FIELDNAMES = [
    "scenario_id", "profile", "query_short",
    "algo", "seed",
    "fitness", "n_pois_selected",
    "elapsed_s",
    "n_pois_input", "selected_algo_production",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", default="data/bench_fixtures")
    p.add_argument("--out",      default="outputs/bench_results.csv")
    p.add_argument("--seeds",    type=int, default=20,
                   help="Número de runs independentes por (fixture × algo)")
    p.add_argument("--algos",    nargs="+",
                   default=["ACO", "GA", "PSO", "GREEDY"],
                   choices=["ACO", "GA", "PSO", "GREEDY"])
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

    # user_prefs: converter start_location de list para tuple
    prefs = dict(data["user_prefs"])
    if "start_location" in prefs and isinstance(prefs["start_location"], list):
        prefs["start_location"] = tuple(prefs["start_location"])
    if "all_geos" in prefs and prefs["all_geos"]:
        prefs["all_geos"] = [list(g) for g in prefs["all_geos"]]

    return pois, dist_matrix, prefs, data


def build_optimizer(algo: str, pois, dist_matrix, evaluator):
    cfg = ALGO_CONFIGS[algo]
    if algo == "ACO":
        return TourismACO(pois, dist_matrix, evaluator, **cfg)
    elif algo == "GA":
        return TourismGA(pois, dist_matrix, evaluator, **cfg)
    elif algo == "PSO":
        return TourismPSOA(pois, dist_matrix, evaluator, **cfg)
    else:  # GREEDY
        return GreedyPlanner(pois, dist_matrix, evaluator)


def run_one(algo: str, pois, dist_matrix, evaluator, seed: int):
    opt = build_optimizer(algo, pois, dist_matrix, evaluator)
    t0  = time.perf_counter()
    if algo == "GREEDY":
        result = opt.optimize()          # determinístico, seed ignorado
    else:
        result = opt.optimize(seed=seed)
    elapsed = time.perf_counter() - t0
    fitness = result.get("fitness", 0.0) or 0.0
    n_pois  = len(result.get("route", []))
    return fitness, n_pois, elapsed


def main():
    args = parse_args()

    fixture_dir = Path(args.fixtures)
    out_path    = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixtures = sorted(fixture_dir.glob("*.json"))
    if not fixtures:
        sys.exit(f"Nenhum fixture encontrado em {fixture_dir}")

    print(f"[bench] {len(fixtures)} fixtures  |  algos: {args.algos}  |  seeds: {args.seeds}")
    total_runs = len(fixtures) * len(args.algos) * args.seeds
    print(f"[bench] Total runs: {total_runs}\n")

    write_header = not out_path.exists()
    run_count = 0

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

            for algo in args.algos:
                fitnesses = []
                for seed in range(args.seeds):
                    try:
                        fitness, n_sel, elapsed = run_one(algo, pois, dist_matrix, evaluator, seed)
                    except Exception as e:
                        print(f"  ERRO {sid} {algo} seed={seed}: {e}")
                        traceback.print_exc()
                        fitness, n_sel, elapsed = 0.0, 0, 0.0

                    writer.writerow({
                        "scenario_id":              sid,
                        "profile":                  profile,
                        "query_short":              query_sh,
                        "algo":                     algo,
                        "seed":                     seed,
                        "fitness":                  round(fitness, 4),
                        "n_pois_selected":          n_sel,
                        "elapsed_s":                round(elapsed, 3),
                        "n_pois_input":             len(pois),
                        "selected_algo_production": prod_algo,
                    })
                    fitnesses.append(fitness)
                    run_count += 1

                mean_f = sum(fitnesses) / len(fitnesses) if fitnesses else 0
                print(f"  {sid} {algo:6s}  mean={mean_f:.2f}  ({run_count}/{total_runs})")

    print(f"\n[bench] Concluído. Resultados em: {out_path}")


if __name__ == "__main__":
    main()
