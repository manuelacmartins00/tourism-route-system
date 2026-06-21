"""
evals/run_grid_search.py
========================
Grid search de hiperparâmetros para ACO, GA e PSO sobre 100 fixtures estratificadas.

Valores da literatura especializada em Orienteering/TOP:
  ACO: 54 configs (3×2×3×3)  — Ke(2008), Montemanni(2009), irace de Sun(2022)
  GA : 54 configs (3×2×3×3)  — Xiao(2020), Bouly(2010), Tasgetiren(2002)
  PSO: 24 configs (2×3×4)    — Dang(2013), Muthuswamy(2011)
  Total: 132 configs × 100 fixtures × 3 seeds = 39 600 runs

Uso (local):
    python evals/run_grid_search.py
    python evals/run_grid_search.py --fixtures data/bench_fixtures_grid100 \
        --out outputs/grid_search_results.csv --seeds 3 --algos ACO GA PSO

Uso (Hetzner / nohup):
    nohup python evals/run_grid_search.py > logs/grid_search.log 2>&1 &

Retoma automaticamente se o CSV já existir (--resume, activo por omissão).
"""
import os, sys, json, time, argparse, csv, traceback, contextlib, io, itertools
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimizers.route_evaluator import POI, RouteEvaluator
from src.optimizers.tourism_aco    import TourismACO
from src.optimizers.tourism_ga     import TourismGA
from src.optimizers.tourism_psoa   import TourismPSOA
from src.optimizers.greedy_planner import GreedyPlanner


# ── Grid de hiperparâmetros (literatura OP/TOP especializada) ─────────────────

def _build_aco_grid():
    """54 = 3×2×3×3"""
    combos = []
    for n_ants in (20, 30, 50):            # Ke=20; actual=30; Sun=50
        for alpha in (1, 2):               # irace≈1; MAUT=0.74→1; test 2
            for beta in (0.5, 2.0, 5.0):  # Ke=0.5 (OP puro); actual=2; irace=5.46
                for evap in (0.1, 0.3, 0.5):  # Montemanni=0.1; medio=0.3; irace/actual=0.5
                    combos.append({
                        "algo_family": "ACO",
                        "n_ants":      n_ants,
                        "n_iterations": 100,
                        "alpha":       alpha,
                        "beta":        beta,
                        "evaporation": evap,
                    })
    return combos


def _build_ga_grid():
    """54 = 3×2×3×3"""
    combos = []
    for pop in (30, 50, 100):             # pequeno/actual/grande
        for ngen in (30, 50):             # actual=30; Xiao usa early-stop→fixar +alto
            for cx in (0.60, 0.75, 0.90): # Xiao=0.60; actual≈0.70→0.75; alto=0.90
                for mut in (0.05, 0.10, 0.20):  # Xiao=0.05; actual=0.10; liberal=0.20
                    combos.append({
                        "algo_family":   "GA",
                        "population_size": pop,
                        "n_generations": ngen,
                        "crossover_prob": cx,
                        "mutation_prob":  mut,
                    })
    return combos


def _build_pso_grid():
    """24 = 2×3×4"""
    combos = []
    for n_part in (20, 40):               # Dang(2013)=40; Muthuswamy=40; 20 por eficiência
        for w in (0.4, 0.7, 0.9):        # Muthuswamy=0.4; actual=0.7; Dang=0.9 com decay
            for c in (0.5, 1.0, 1.5, 2.0):  # Dang=0.5; mid=1.0; actual=1.5; standard=2.0
                combos.append({
                    "algo_family": "PSO",
                    "n_particles": n_part,
                    "n_iterations": 50,
                    "w":           w,
                    "c1":          c,
                    "c2":          c,
                    # w=0.9 → activar decay (Dang 2013); outros: fixo
                    "w_decay":     (w == 0.9),
                })
    return combos


ALL_CONFIGS = _build_aco_grid() + _build_ga_grid() + _build_pso_grid()

# Atribuir config_id estável (índice 0-based dentro da família)
_fam_counter: dict = {}
for _cfg in ALL_CONFIGS:
    fam = _cfg["algo_family"]
    _fam_counter[fam] = _fam_counter.get(fam, 0)
    _cfg["config_id"] = f"{fam}_{_fam_counter[fam]:03d}"
    _fam_counter[fam] += 1


# ── Schema CSV ────────────────────────────────────────────────────────────────
FIELDNAMES = [
    # identificação da run
    "config_id", "algo_family", "scenario_id", "profile", "seed",
    # hiperparâmetros (por família; NULL para outras)
    "n_ants", "alpha", "beta", "evaporation",           # ACO
    "population_size", "n_generations",                   # GA
    "crossover_prob", "mutation_prob",                    # GA
    "n_particles", "w", "w_decay", "c1", "c2",          # PSO
    # fitness e componentes
    "fitness",
    "time_efficiency", "proximity_component", "diversity_component",
    "distance_penalty", "cat_indata_comp", "cat_general_comp",
    "contextual_modifier",
    # metadados
    "n_pois_selected", "n_pois_input", "elapsed_s",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _silent():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


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
    return pois, dist_matrix, prefs, data


def build_optimizer(cfg: dict, pois, dist_matrix, evaluator):
    fam  = cfg["algo_family"]
    keys = {k: v for k, v in cfg.items()
            if k not in ("algo_family", "config_id")}
    if fam == "ACO":
        return TourismACO(pois, dist_matrix, evaluator,
                          n_ants=keys["n_ants"],
                          n_iterations=keys["n_iterations"],
                          alpha=keys["alpha"],
                          beta=keys["beta"],
                          evaporation=keys["evaporation"])
    elif fam == "GA":
        return TourismGA(pois, dist_matrix, evaluator,
                         population_size=keys["population_size"],
                         n_generations=keys["n_generations"],
                         crossover_prob=keys["crossover_prob"],
                         mutation_prob=keys["mutation_prob"])
    elif fam == "PSO":
        return TourismPSOA(pois, dist_matrix, evaluator,
                           n_particles=keys["n_particles"],
                           n_iterations=keys["n_iterations"],
                           w=keys["w"],
                           c1=keys["c1"],
                           c2=keys["c2"],
                           w_decay=keys.get("w_decay", False))
    else:
        return GreedyPlanner(pois, dist_matrix, evaluator)


def run_one(cfg: dict, pois, dist_matrix, evaluator, seed: int):
    opt = build_optimizer(cfg, pois, dist_matrix, evaluator)
    t0  = time.perf_counter()
    with _silent():
        result = opt.optimize(seed=seed)
    elapsed = time.perf_counter() - t0
    route   = result.get("route") or []
    fitness = result.get("fitness", 0.0) or 0.0
    comps   = evaluator.calculate_fitness_components(route) if route else {}
    return fitness, route, elapsed, comps


def cfg_to_row(cfg: dict, sid: str, profile: str, seed: int,
               fitness: float, comps: dict, n_sel: int, n_inp: int, elapsed: float):
    return {
        "config_id":    cfg["config_id"],
        "algo_family":  cfg["algo_family"],
        "scenario_id":  sid,
        "profile":      profile,
        "seed":         seed,
        # ACO params
        "n_ants":       cfg.get("n_ants", ""),
        "alpha":        cfg.get("alpha", ""),
        "beta":         cfg.get("beta", ""),
        "evaporation":  cfg.get("evaporation", ""),
        # GA params
        "population_size": cfg.get("population_size", ""),
        "n_generations":   cfg.get("n_generations", ""),
        "crossover_prob":  cfg.get("crossover_prob", ""),
        "mutation_prob":   cfg.get("mutation_prob", ""),
        # PSO params
        "n_particles":  cfg.get("n_particles", ""),
        "w":            cfg.get("w", ""),
        "w_decay":      cfg.get("w_decay", ""),
        "c1":           cfg.get("c1", ""),
        "c2":           cfg.get("c2", ""),
        # fitness
        "fitness":      round(fitness, 4),
        "time_efficiency":     round(comps.get("time_efficiency",    0), 3),
        "proximity_component": round(comps.get("proximity_component",0), 3),
        "diversity_component": round(comps.get("diversity_component",0), 3),
        "distance_penalty":    round(comps.get("distance_penalty",   0), 3),
        "cat_indata_comp":     round(comps.get("cat_indata_comp",    0), 3),
        "cat_general_comp":    round(comps.get("cat_general_comp",   0), 3),
        "contextual_modifier": round(comps.get("contextual_modifier",1), 4),
        # meta
        "n_pois_selected": n_sel,
        "n_pois_input":    n_inp,
        "elapsed_s":       round(elapsed, 3),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", default="data/bench_fixtures_grid100")
    p.add_argument("--out",      default="outputs/grid_search_results.csv")
    p.add_argument("--seeds",    type=int, default=3,
                   help="Seeds por (config × fixture)")
    p.add_argument("--algos",    nargs="+", default=["ACO", "GA", "PSO"],
                   choices=["ACO", "GA", "PSO"],
                   help="Famílias de algoritmos a executar")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignorar runs já guardadas (recomeça do zero)")
    return p.parse_args()


def main():
    args   = parse_args()
    out_fp = Path(args.out)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    # Seleccionar configurações pedidas
    configs = [c for c in ALL_CONFIGS if c["algo_family"] in args.algos]
    print(f"[grid] {len(configs)} configs  |  famílias: {args.algos}  |  seeds: {args.seeds}")

    # Carregar fixtures
    fixture_dir = Path(args.fixtures)
    fixtures = sorted(fixture_dir.glob("*.json"),
                      key=lambda f: int(f.stem) if f.stem.isdigit() else f.stem)
    if not fixtures:
        sys.exit(f"[grid] Nenhum fixture em {fixture_dir}")
    print(f"[grid] {len(fixtures)} fixtures em {fixture_dir}")

    total_runs = len(configs) * len(fixtures) * args.seeds
    print(f"[grid] Total runs: {total_runs:,}\n")

    # Detectar runs já feitas (para retoma)
    done_keys: set = set()
    write_header = True
    if out_fp.exists() and not args.no_resume:
        write_header = False
        with open(out_fp, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_keys.add((row["config_id"], row["scenario_id"], row["seed"]))
        print(f"[grid] Retomar: {len(done_keys)} runs já concluídas\n")

    skipped = 0
    run_count = 0
    err_count = 0
    t_global  = time.perf_counter()

    with open(out_fp, "a", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for cfg in configs:
            cid = cfg["config_id"]

            for fp in fixtures:
                try:
                    pois, dist_matrix, prefs, meta = load_fixture(fp)
                except Exception as e:
                    print(f"  SKIP fixture {fp.name}: {e}")
                    continue

                sid     = meta.get("scenario_id", fp.stem)
                profile = meta.get("profile", "?")
                evaluator = RouteEvaluator(pois, dist_matrix, prefs)

                for seed in range(args.seeds):
                    key = (cid, str(sid), str(seed))
                    if key in done_keys:
                        skipped += 1
                        run_count += 1
                        continue

                    try:
                        fitness, route, elapsed, comps = run_one(
                            cfg, pois, dist_matrix, evaluator, seed)
                    except Exception as e:
                        print(f"  ERRO {cid} {sid} seed={seed}: {e}")
                        traceback.print_exc()
                        fitness, route, elapsed, comps = 0.0, [], 0.0, {}
                        err_count += 1

                    row = cfg_to_row(
                        cfg, sid, profile, seed,
                        fitness, comps,
                        len(route), len(pois), elapsed)
                    writer.writerow(row)
                    csvf.flush()
                    run_count += 1

                # Progresso por fixture × config
                elapsed_total = time.perf_counter() - t_global
                runs_done = run_count - skipped
                runs_todo = total_runs - run_count
                eta_s = (elapsed_total / max(runs_done, 1)) * runs_todo
                eta_h = eta_s / 3600
                print(f"  {cid:10s} {sid!s:8s} ({run_count}/{total_runs}) "
                      f"ETA={eta_h:.1f}h  erros={err_count}",
                      end="\r", flush=True)

            # Linha de separação por config
            elapsed_total = time.perf_counter() - t_global
            print(f"\n[{cid}] concluído em {elapsed_total:.0f}s total  "
                  f"({run_count}/{total_runs} runs)")

    print(f"\n[grid] Fim. {run_count} runs  |  {skipped} skip  |  {err_count} erros")
    print(f"[grid] Resultados: {out_fp}")


if __name__ == "__main__":
    main()
