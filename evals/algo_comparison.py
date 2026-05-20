"""
evals/algo_comparison.py
Comparação sistemática dos 4 algoritmos (ACO, GA, PSO, GREEDY) em condições
controladas: cada query corre 4× com force_algorithm.

Dimensões testadas:
  - Duração: curta (≤480 min), média (960-1440 min), longa (≥2400 min)
  - n_candidates: varia com a localização e categorias
  - Tipo: cidade única, corredor multi-cidade, região grande

Uso:
  python evals/algo_comparison.py
  python evals/algo_comparison.py --output outputs/algo_comparison.json
"""

import os, sys, json, time, argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Forcar UTF-8 no stdout para evitar UnicodeEncodeError no Windows cp1252
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))
from main_system import TourismRouteSystem

ALGORITHMS = ["GA", "PSO", "ACO", "GREEDY"]

# ---------------------------------------------------------------------------
# Queries de teste — cobrem os eixos críticos para a comparação
# Todas sem campos em falta (sem clarificação).
# include_accommodation e include_meals explícitos para evitar scope questions.
# ---------------------------------------------------------------------------
TEST_CASES = [
    # --- CURTAS: ≤480 min (candidatos tipicamente 15-30) ---
    {
        "id": "S1",
        "label": "4h cidade, 1 cat",
        "dim_time": "curta",
        "dim_location": "cidade",
        "query": (
            "Quero visitar museus e monumentos em Lisboa durante 4 horas. "
            "Orçamento de 30 euros por pessoa. Vou a pé."
        ),
        "include_accommodation": False,
        "include_meals": False,
    },
    {
        "id": "S2",
        "label": "4h cidade, multi-cat",
        "dim_time": "curta",
        "dim_location": "cidade",
        "query": (
            "Tarde em Porto: museus, espaços verdes e gastronomia. "
            "4 horas, 50 euros por pessoa. Transportes públicos."
        ),
        "include_accommodation": False,
        "include_meals": True,
    },
    {
        "id": "S3",
        "label": "1 dia cidade",
        "dim_time": "curta",
        "dim_location": "cidade",
        "query": (
            "Visita de 1 dia a Coimbra: monumentos históricos, arqueologia e cafés. "
            "Orçamento 60 euros por pessoa. Vou a pé."
        ),
        "include_accommodation": False,
        "include_meals": True,
    },

    # --- MÉDIAS: 960-1440 min (candidatos tipicamente 25-45) ---
    {
        "id": "M1",
        "label": "2 dias cidade",
        "dim_time": "media",
        "dim_location": "cidade",
        "query": (
            "Fim de semana em Lisboa com a família (2 adultos + 2 crianças). "
            "Parques, zoos e monumentos. 2 dias, 80 euros por pessoa. De carro."
        ),
        "include_accommodation": True,
        "include_meals": True,
    },
    {
        "id": "M2",
        "label": "3 dias cidade, noturno",
        "dim_time": "media",
        "dim_location": "cidade",
        "query": (
            "3 dias no Porto: cultura de dia, vida noturna de noite. "
            "Somos 4 amigos. 100 euros por pessoa por dia. De transportes públicos."
        ),
        "include_accommodation": True,
        "include_meals": True,
    },
    {
        "id": "M3",
        "label": "2 dias região, praia",
        "dim_time": "media",
        "dim_location": "regiao",
        "query": (
            "2 dias no Algarve com foco em praias e turismo activo. "
            "Casal, 120 euros por dia para o grupo. De carro."
        ),
        "include_accommodation": True,
        "include_meals": False,
    },
    {
        "id": "M4",
        "label": "3 dias corredor A→B",
        "dim_time": "media",
        "dim_location": "corredor",
        "query": (
            "Road trip de 3 dias de Lisboa a Coimbra, monumentos e gastronomia. "
            "Somos 2 pessoas, 70 euros por pessoa por dia. De carro."
        ),
        "include_accommodation": True,
        "include_meals": True,
    },

    # --- LONGAS: ≥2400 min (candidatos tipicamente 40-60) ---
    {
        "id": "L1",
        "label": "5 dias cidade",
        "dim_time": "longa",
        "dim_location": "cidade",
        "query": (
            "5 dias no Porto: museus, monumentos, gastronomia e espaços verdes. "
            "Viajo sozinho, 80 euros por dia. De transportes públicos."
        ),
        "include_accommodation": True,
        "include_meals": True,
    },
    {
        "id": "L2",
        "label": "1 semana região, bem-estar",
        "dim_time": "longa",
        "dim_location": "regiao",
        "query": (
            "1 semana no Alentejo: termas, espaços verdes e gastronomia alentejana. "
            "Casal, 150 euros por dia para o grupo. De carro."
        ),
        "include_accommodation": True,
        "include_meals": True,
    },
    {
        "id": "L3",
        "label": "5 dias corredor A→B→C",
        "dim_time": "longa",
        "dim_location": "corredor",
        "query": (
            "Road trip de 5 dias de Porto a Coimbra a Lisboa, natureza e cultura. "
            "Somos 3 amigos, 90 euros por pessoa por dia. De carro."
        ),
        "include_accommodation": True,
        "include_meals": True,
    },
    {
        "id": "L4",
        "label": "1 semana região grande, multi-cat",
        "dim_time": "longa",
        "dim_location": "regiao_grande",
        "query": (
            "Viagem de 1 semana pelo Norte de Portugal: natureza, arqueologia e turismo activo. "
            "Somos 2 pessoas, 100 euros por dia para o grupo. De carro."
        ),
        "include_accommodation": True,
        "include_meals": True,
    },

    # --- MUITO LONGAS: ≥4800 min (condição hipotética PSO vantagem) ---
    {
        "id": "XL1",
        "label": "10 dias Portugal, multi-cat",
        "dim_time": "muito_longa",
        "dim_location": "pais",
        "query": (
            "Viagem de 10 dias por Portugal, começando em Lisboa e acabando no Porto. "
            "Museus, monumentos, praias e gastronomia. "
            "Somos 2 pessoas, 80 euros por pessoa por dia. De carro."
        ),
        "include_accommodation": True,
        "include_meals": True,
    },
    {
        "id": "XL2",
        "label": "2 semanas Portugal, família",
        "dim_time": "muito_longa",
        "dim_location": "pais",
        "query": (
            "Duas semanas de férias em Portugal com família (2 adultos e 2 crianças). "
            "Parques, praias, zoos e monumentos. 150 euros por dia para o grupo. De carro."
        ),
        "include_accommodation": True,
        "include_meals": True,
    },
]


def run_test_case(system: TourismRouteSystem, tc: dict, algo: str) -> dict:
    t0 = time.time()
    try:
        result = system.plan_route(
            tc["query"],
            use_shap=False,
            verbose=False,
            force_algorithm=algo,
            include_accommodation=tc["include_accommodation"],
            include_meals=tc["include_meals"],
            generate_map=False,
        )
        elapsed = round(time.time() - t0, 1)

        status = result.get("status", "ok")
        if status in ("needs_clarification", "needs_scope_clarification"):
            return {
                "algo": algo, "status": "clarification",
                "elapsed_s": elapsed,
                "missing": result.get("missing_fields", []) + result.get("scope_questions", []),
            }
        if "error" in result:
            return {"algo": algo, "status": f"error_{result['error']}", "elapsed_s": elapsed}

        opt = result.get("optimization", {})
        fc  = opt.get("fitness_components", {})
        return {
            "algo":            algo,
            "status":          "ok",
            "elapsed_s":       elapsed,
            "fitness":         round(opt.get("fitness", 0), 3),
            "n_candidates":    opt.get("n_candidates", 0),
            "n_pois":          opt.get("n_selected", 0),
            "visit_min":       round(opt.get("visit_time_min", 0)),
            "total_min":       round(opt.get("total_time_min", 0)),
            "max_time":        result.get("preferences", {}).get("max_time", 0),
            "cost_pp":         round(result.get("cost_per_person", 0), 2),
            "max_cost":        result.get("preferences", {}).get("max_cost", 0),
            "unique_cats":     fc.get("unique_categories", 0),
            "time_util":       fc.get("time_utilization", 0),
            "cat_comp":        fc.get("category_component", 0),
            "div_comp":        fc.get("diversity_component", 0),
            "prox_comp":       fc.get("proximity_component", 0),
            "dist_pen":        fc.get("distance_penalty", 0),
            "n_days":          len(result.get("day_plan", {}).get("days", [])),
        }
    except Exception as e:
        return {"algo": algo, "status": "exception", "error": str(e),
                "elapsed_s": round(time.time() - t0, 1)}


def print_case_table(tc: dict, runs: list[dict]):
    """Imprime tabela comparativa para um test case."""
    ok_runs = [r for r in runs if r["status"] == "ok"]
    if not ok_runs:
        print("   [sem resultados OK]")
        return

    max_time  = ok_runs[0].get("max_time", 0)
    n_cands   = ok_runs[0].get("n_candidates", 0)
    print(f"   max_time={max_time}min  n_candidates={n_cands}")
    print(f"   {'Algo':<8} {'Fitness':>7} {'POIs':>4} {'t_util%':>7} {'cat%':>6} "
          f"{'div%':>6} {'prox%':>6} {'uniq':>4} {'€/pp':>6} {'secs':>5}")
    print(f"   {'-'*65}")

    # Ordenar por fitness desc
    ok_runs_sorted = sorted(ok_runs, key=lambda r: -r.get("fitness", 0))
    for r in ok_runs_sorted:
        winner = " <--" if r == ok_runs_sorted[0] else ""
        print(f"   {r['algo']:<8} {r['fitness']:>7.3f} {r['n_pois']:>4} "
              f"{r.get('time_util',0):>7.1f} {r.get('cat_comp',0):>6.1f} "
              f"{r.get('div_comp',0):>6.1f} {r.get('prox_comp',0):>6.1f} "
              f"{r.get('unique_cats',0):>4} {r.get('cost_pp',0):>6.0f} "
              f"{r['elapsed_s']:>5.1f}{winner}")

    for r in runs:
        if r["status"] != "ok":
            print(f"   {r['algo']:<8} [{r['status']}]  {r.get('error','')[:40]}")


def print_summary_table(all_results: dict):
    """Tabela resumo: para cada test case, qual algoritmo ganhou e por quanto."""
    print(f"\n{'='*75}")
    print("TABELA RESUMO — VENCEDOR POR CASO")
    print(f"{'='*75}")
    print(f"{'ID':<4} {'Label':<30} {'Dim':<6} {'Vencedor':>8} {'2o':>8} "
          f"{'Δfit (1-2)':>10} {'Δfit (1-4)':>10}")
    print(f"{'-'*75}")

    wins = {a: 0 for a in ALGORITHMS}
    for tc_id, runs in all_results.items():
        ok_runs = sorted([r for r in runs if r["status"] == "ok"],
                         key=lambda r: -r.get("fitness", 0))
        if not ok_runs:
            continue
        tc = next(t for t in TEST_CASES if t["id"] == tc_id)
        w   = ok_runs[0]
        s2  = ok_runs[1] if len(ok_runs) > 1 else None
        s4  = ok_runs[-1] if len(ok_runs) > 1 else None
        delta12 = round(w["fitness"] - s2["fitness"], 3) if s2 else 0
        delta14 = round(w["fitness"] - s4["fitness"], 3) if s4 else 0
        wins[w["algo"]] += 1
        s2_algo = s2["algo"] if s2 else "-"
        print(f"{tc_id:<4} {tc['label']:<30} {tc['dim_time']:<6} "
              f"{w['algo']:>8} {s2_algo:>8} {delta12:>+10.3f} {delta14:>+10.3f}")

    print(f"\n  VITORIAS: " + "  ".join(f"{a}={wins[a]}" for a in ALGORITHMS))

    # Análise por dimensão de tempo
    print(f"\n  ANALISE POR DURAÇÃO:")
    for dim in ["curta", "media", "longa", "muito_longa"]:
        dim_cases = [t["id"] for t in TEST_CASES if t["dim_time"] == dim]
        dim_wins = {a: 0 for a in ALGORITHMS}
        for tc_id in dim_cases:
            runs = all_results.get(tc_id, [])
            ok_runs = sorted([r for r in runs if r["status"] == "ok"],
                             key=lambda r: -r.get("fitness", 0))
            if ok_runs:
                dim_wins[ok_runs[0]["algo"]] += 1
        total = sum(dim_wins.values())
        if total > 0:
            print(f"    {dim:<12}: " + "  ".join(f"{a}={dim_wins[a]}" for a in ALGORITHMS))

    # Análise por tipo de localização
    print(f"\n  ANALISE POR TIPO DE LOCALIZAÇÃO:")
    for dim in ["cidade", "regiao", "corredor", "regiao_grande", "pais"]:
        dim_cases = [t["id"] for t in TEST_CASES if t["dim_location"] == dim]
        if not dim_cases:
            continue
        dim_wins = {a: 0 for a in ALGORITHMS}
        for tc_id in dim_cases:
            runs = all_results.get(tc_id, [])
            ok_runs = sorted([r for r in runs if r["status"] == "ok"],
                             key=lambda r: -r.get("fitness", 0))
            if ok_runs:
                dim_wins[ok_runs[0]["algo"]] += 1
        total = sum(dim_wins.values())
        if total > 0:
            print(f"    {dim:<14}: " + "  ".join(f"{a}={dim_wins[a]}" for a in ALGORITHMS))

    # Análise por n_candidates
    print(f"\n  ANALISE POR N_CANDIDATES (apenas casos OK):")
    buckets = {"≤20": [], "21-40": [], ">40": []}
    for tc_id, runs in all_results.items():
        ok_runs = sorted([r for r in runs if r["status"] == "ok"],
                         key=lambda r: -r.get("fitness", 0))
        if not ok_runs:
            continue
        nc = ok_runs[0].get("n_candidates", 0)
        key = "≤20" if nc <= 20 else ("21-40" if nc <= 40 else ">40")
        buckets[key].append(ok_runs[0]["algo"])
    for bucket, algos in buckets.items():
        if algos:
            from collections import Counter
            c = Counter(algos)
            print(f"    n_cand {bucket}: " + "  ".join(f"{a}={c.get(a,0)}" for a in ALGORITHMS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Ficheiro JSON para guardar resultados")
    parser.add_argument("--cases",  default=None, help="IDs dos casos a correr (ex: S1,M2,L1)")
    parser.add_argument("--algos",  default=None, help="Algoritmos a comparar (ex: GA,PSO)")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERRO: GROQ_API_KEY nao definida no .env")
        sys.exit(1)

    cases_to_run = TEST_CASES
    if args.cases:
        ids = {c.strip() for c in args.cases.split(",")}
        cases_to_run = [t for t in TEST_CASES if t["id"] in ids]
        if not cases_to_run:
            print(f"ERRO: nenhum caso encontrado com IDs {ids}")
            sys.exit(1)

    algos_to_run = ALGORITHMS
    if args.algos:
        algos_to_run = [a.strip().upper() for a in args.algos.split(",")]

    output_path = args.output
    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/algo_comparison_{ts}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"COMPARACAO DE ALGORITMOS")
    print(f"  Casos    : {len(cases_to_run)} queries")
    print(f"  Algoritmos: {algos_to_run}")
    print(f"  Total runs: {len(cases_to_run) * len(algos_to_run)}")
    print(f"  Output   : {output_path}")
    print(f"{'='*70}\n")

    system = TourismRouteSystem(api_key=api_key)

    all_results = {}
    total_runs = len(cases_to_run) * len(algos_to_run)
    run_idx = 0

    for tc in cases_to_run:
        print(f"\n{'─'*70}")
        print(f"[{tc['id']}] {tc['label']}  ({tc['dim_time']}, {tc['dim_location']})")
        print(f"  Query: {tc['query'][:80]}...")
        runs = []

        for algo in algos_to_run:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] {algo}...", end=" ", flush=True)
            r = run_test_case(system, tc, algo)
            runs.append(r)
            if r["status"] == "ok":
                print(f"fitness={r['fitness']:.3f}  n={r['n_pois']}  {r['elapsed_s']}s")
            else:
                print(f"[{r['status']}] {r.get('error','')[:50]}")
            time.sleep(0.5)  # throttle leve entre runs do mesmo caso

        all_results[tc["id"]] = runs
        print()
        print_case_table(tc, runs)

    # Tabela resumo final
    print_summary_table(all_results)

    # Guardar JSON completo
    output = {
        "timestamp": datetime.now().isoformat(),
        "algorithms": algos_to_run,
        "n_cases": len(cases_to_run),
        "cases": {
            tc["id"]: {
                "label": tc["label"],
                "dim_time": tc["dim_time"],
                "dim_location": tc["dim_location"],
                "query": tc["query"],
                "runs": all_results.get(tc["id"], []),
            }
            for tc in cases_to_run
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Resultados completos: {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
