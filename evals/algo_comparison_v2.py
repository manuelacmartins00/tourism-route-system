"""
evals/algo_comparison_v2.py
Comparação dos 4 algoritmos sem LLM — parâmetros definidos explicitamente.

Cada caso corre RAG + build matrix uma vez, depois corre os 4 algoritmos
sobre os mesmos candidatos. Nenhuma chamada Groq — zero tokens consumidos.

Uso:
  python evals/algo_comparison_v2.py
  python evals/algo_comparison_v2.py --cases S1,M1,L1
  python evals/algo_comparison_v2.py --algos GA,PSO
  python evals/algo_comparison_v2.py --output outputs/algo_v2.json
"""

import os, sys, json, time, math, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.rag_setup import POI_RAG
from src.optimizers.route_evaluator import RouteEvaluator, POI
from src.utils.location_resolver import LocationResolver
from src.utils.distance_calculator import haversine
from src.optimizers.tourism_aco import TourismACO
from src.optimizers.tourism_ga import TourismGA
from src.optimizers.tourism_psoa import TourismPSOA
from src.optimizers.greedy_planner import GreedyPlanner
from src.utils.data_loader import load_pois_from_json

ALGORITHMS = ["GA", "PSO", "ACO", "GREEDY"]

ACCOMMODATION_BUNDLES = [
    "hotelaria", "alojamento_local", "turismo_habitacao",
    "turismo_espaco_rural", "apartamento_turistico",
    "pousadas_da_juventude", "aldeamento_turistico", "parques_de_campismo",
]
NEVER_INCLUDE = [
    "eventos", "postos_de_turismo", "agencias_de_viagem",
    "localidade", "servicos_de_turismo", "outros",
]
DURATION_RANGES = {
    "hotelaria": (30,30), "alojamento_local": (30,30), "turismo_habitacao": (30,30),
    "turismo_espaco_rural": (30,30), "apartamento_turistico": (30,30),
    "pousadas_da_juventude": (30,30), "aldeamento_turistico": (30,30),
    "parques_de_campismo": (30,30),
    "restaurantes_e_cafes": (45,90), "monumentos": (15,60),
    "turismo_activo": (90,300), "praias": (60,240),
    "bares_e_discotecas": (90,240), "museus_e_palacios": (60,150),
    "campos": (120,300), "arqueologia": (30,90), "espacos_verdes": (30,180),
    "marinas_e_portos": (20,90), "termas": (90,180),
    "parques_e_reservas": (60,240), "parques_de_diversao": (180,360),
    "zoos_e_aquarios": (120,240), "ciencia_e_conhecimento": (60,120),
    "casinos": (60,240), "talassoterapia": (90,180),
    "grutas": (30,75), "academias": (60,120), "barragens": (20,60),
}

# ---------------------------------------------------------------------------
# Test cases — parâmetros explícitos, sem LLM
# max_cost = orçamento por pessoa para toda a viagem
# ---------------------------------------------------------------------------
TEST_CASES = [
    # ── CURTAS ≤480 min ──────────────────────────────────────────────────
    {
        "id": "S1", "label": "4h Lisboa, museus+monumentos",
        "dim_time": "curta", "dim_location": "cidade",
        "location": "Lisboa", "max_time": 240, "max_cost": 30.0,
        "categories": ["museus_e_palacios", "monumentos"],
        "transport": "foot", "include_accommodation": False,
        "include_meals": False, "num_people": 1,
    },
    {
        "id": "S2", "label": "4h Porto, multi-cat",
        "dim_time": "curta", "dim_location": "cidade",
        "location": "Porto", "max_time": 240, "max_cost": 50.0,
        "categories": ["museus_e_palacios", "espacos_verdes", "restaurantes_e_cafes"],
        "transport": "public_transport", "include_accommodation": False,
        "include_meals": True, "num_people": 2,
    },
    {
        "id": "S3", "label": "1 dia Coimbra, cultura",
        "dim_time": "curta", "dim_location": "cidade",
        "location": "Coimbra", "max_time": 480, "max_cost": 60.0,
        "categories": ["monumentos", "arqueologia", "museus_e_palacios"],
        "transport": "foot", "include_accommodation": False,
        "include_meals": True, "num_people": 1,
    },
    # ── MÉDIAS 960-1440 min ──────────────────────────────────────────────
    {
        "id": "M1", "label": "2 dias Lisboa, familia+zoo",
        "dim_time": "media", "dim_location": "cidade",
        "location": "Lisboa", "max_time": 960, "max_cost": 160.0,  # 80€/pp x 2 dias
        "categories": ["zoos_e_aquarios", "parques_e_reservas", "monumentos", "espacos_verdes"],
        "transport": "car", "include_accommodation": True,
        "include_meals": True, "num_people": 4, "has_children": True,
    },
    {
        "id": "M2", "label": "3 dias Porto, cultura+noturno",
        "dim_time": "media", "dim_location": "cidade",
        "location": "Porto", "max_time": 1440, "max_cost": 300.0,  # 100€/pp x 3 dias
        "categories": ["museus_e_palacios", "monumentos", "bares_e_discotecas", "restaurantes_e_cafes"],
        "transport": "public_transport", "include_accommodation": True,
        "include_meals": True, "num_people": 4,
    },
    {
        "id": "M3", "label": "2 dias Algarve, praia+activo",
        "dim_time": "media", "dim_location": "regiao",
        "location": "Algarve", "max_time": 960, "max_cost": 120.0,  # 60€/dia / 2pp
        "categories": ["praias", "turismo_activo", "espacos_verdes"],
        "transport": "car", "include_accommodation": True,
        "include_meals": False, "num_people": 2,
    },
    {
        "id": "M4", "label": "3 dias corredor Lisboa-Coimbra",
        "dim_time": "media", "dim_location": "corredor",
        "location": "Lisboa", "end_location": "Coimbra",
        "max_time": 1440, "max_cost": 210.0,  # 70€/pp x 3 dias
        "categories": ["monumentos", "museus_e_palacios", "restaurantes_e_cafes", "arqueologia"],
        "transport": "car", "include_accommodation": True,
        "include_meals": True, "num_people": 2,
    },
    # ── LONGAS ≥2400 min ─────────────────────────────────────────────────
    {
        "id": "L1", "label": "5 dias Porto, multi-cat",
        "dim_time": "longa", "dim_location": "cidade",
        "location": "Porto", "max_time": 2400, "max_cost": 400.0,  # 80€/dia x 5
        "categories": ["museus_e_palacios", "monumentos", "espacos_verdes", "restaurantes_e_cafes"],
        "transport": "public_transport", "include_accommodation": True,
        "include_meals": True, "num_people": 1,
    },
    {
        "id": "L2", "label": "1 semana Alentejo, bem-estar",
        "dim_time": "longa", "dim_location": "regiao",
        "location": "Alentejo", "max_time": 3360, "max_cost": 525.0,  # 75€/dia / 2pp x 7
        "categories": ["termas", "espacos_verdes", "restaurantes_e_cafes", "turismo_activo"],
        "transport": "car", "include_accommodation": True,
        "include_meals": True, "num_people": 2,
    },
    {
        "id": "L3", "label": "5 dias corredor Porto-Coimbra-Lisboa",
        "dim_time": "longa", "dim_location": "corredor",
        "location": "Porto", "end_location": "Lisboa",
        "max_time": 2400, "max_cost": 450.0,  # 90€/pp x 5 dias
        "categories": ["monumentos", "museus_e_palacios", "parques_e_reservas", "restaurantes_e_cafes"],
        "transport": "car", "include_accommodation": True,
        "include_meals": True, "num_people": 3,
    },
    {
        "id": "L4", "label": "1 semana Norte PT, natureza",
        "dim_time": "longa", "dim_location": "regiao_grande",
        "location": "norte", "max_time": 3360, "max_cost": 350.0,
        "categories": ["parques_e_reservas", "turismo_activo", "arqueologia", "espacos_verdes"],
        "transport": "car", "include_accommodation": True,
        "include_meals": True, "num_people": 2,
    },
    # ── MUITO LONGAS ≥4800 min ────────────────────────────────────────────
    {
        "id": "XL1", "label": "10 dias Lisboa-Porto, multi-cat",
        "dim_time": "muito_longa", "dim_location": "pais",
        "location": "Lisboa", "end_location": "Porto",
        "max_time": 4800, "max_cost": 800.0,  # 80€/pp x 10 dias
        "categories": ["museus_e_palacios", "monumentos", "praias", "restaurantes_e_cafes"],
        "transport": "car", "include_accommodation": True,
        "include_meals": True, "num_people": 2,
    },
    {
        "id": "XL2", "label": "2 semanas PT, familia",
        "dim_time": "muito_longa", "dim_location": "pais",
        "location": "Portugal", "max_time": 6720, "max_cost": 525.0,  # 150€/dia / 4pp x 14d
        "categories": ["parques_de_diversao", "praias", "zoos_e_aquarios", "monumentos", "espacos_verdes"],
        "transport": "car", "include_accommodation": True,
        "include_meals": True, "num_people": 4, "has_children": True,
    },
]

_TIME_TABLE = {
    "foot":             [(1,12), (2,25), (5,60), (float('inf'),999)],
    "car":              [(2,4),  (5,8),  (15,15), (50,38), (float('inf'),75)],
    "public_transport": [(1,10), (2,17), (5,30), (15,47), (50,94), (float('inf'),153)],
    "fastest":          [(2,4),  (5,8),  (15,15), (50,38), (float('inf'),75)],
}

def _travel_time(d_km: float, mode: str) -> float:
    for max_km, t_min in _TIME_TABLE.get(mode, _TIME_TABLE["public_transport"]):
        if d_km <= max_km:
            return float(t_min)
    return 240.0

def _within_radius(plat, plon, clat, clon, radius_km):
    R = 6371
    dlat = math.radians(plat - clat)
    dlon = math.radians(plon - clon)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(clat))*math.cos(math.radians(plat))*math.sin(dlon/2)**2
    return R*2*math.atan2(math.sqrt(a), math.sqrt(1-a)) <= radius_km


def build_problem(tc: dict, rag: POI_RAG, resolver: LocationResolver,
                  verbose: bool = True) -> dict:
    """
    Resolve localização, corre RAG e constrói optimizer_pois + matrix + evaluator.
    Retorna dict com todos os ingredientes para os algoritmos, ou None em caso de erro.
    """
    location = tc["location"]
    geo = resolver.resolve(location)
    if not geo:
        print(f"  ERRO: nao foi possivel resolver '{location}'")
        return None

    center_lat, center_lon, radius_km = geo
    end_geo = None
    end_location = tc.get("end_location")
    if end_location:
        end_geo = resolver.resolve(end_location)

    # Bounding box
    is_corridor = end_geo is not None and (
        abs(center_lat - end_geo[0]) > 0.05 or abs(center_lon - end_geo[1]) > 0.05)

    if is_corridor:
        BUFFER = 0.45
        all_lats = [center_lat, end_geo[0]]
        all_lons = [center_lon, end_geo[1]]
        lat_min, lat_max = min(all_lats)-BUFFER, max(all_lats)+BUFFER
        lon_min, lon_max = min(all_lons)-BUFFER, max(all_lons)+BUFFER
    elif radius_km > 200.0:
        # Regiao grande: bbox ampla
        delta = radius_km / 111.0
        lat_min = center_lat - delta
        lat_max = center_lat + delta
        lon_min = center_lon - delta
        lon_max = center_lon + delta
    else:
        delta = radius_km / 111.0
        lat_min = center_lat - delta
        lat_max = center_lat + delta
        lon_min = center_lon - delta
        lon_max = center_lon + delta

    categories = tc["categories"]
    include_accommodation = tc["include_accommodation"]
    include_meals = tc.get("include_meals", True)

    excluded = list(NEVER_INCLUDE)
    if not include_accommodation:
        excluded.extend(ACCOMMODATION_BUNDLES)
    if not include_meals and "restaurantes_e_cafes" not in categories:
        excluded.append("restaurantes_e_cafes")

    # RAG query
    rag_text = " ".join(categories)
    rag_results = rag.query(
        text=rag_text, n_results=60,
        category_filter=categories,
        category_exclude=excluded,
        max_cost=tc["max_cost"],
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
    )
    candidate_pois = rag_results["pois"]

    # Rebalanceamento
    by_cat = defaultdict(list)
    for p in candidate_pois:
        by_cat[p["category"]].append(p)
    min_per = max(3, 25 // len(categories))
    existing_ids = {p["id"] for p in candidate_pois}
    for cat in categories:
        if len(by_cat[cat]) < min_per:
            extra = rag.query(text=cat, n_results=min_per*2,
                              category_filter=[cat], category_exclude=excluded,
                              max_cost=tc["max_cost"],
                              lat_min=lat_min, lat_max=lat_max,
                              lon_min=lon_min, lon_max=lon_max)
            for p in extra["pois"]:
                if p["id"] not in existing_ids:
                    candidate_pois.append(p)
                    existing_ids.add(p["id"])

    # Forcar alojamento
    if include_accommodation:
        num_days = max(1, math.ceil(tc["max_time"] / 480))
        accom_needed = max(5, num_days + 3)
        accom_r = rag.query(
            text=f"hotel alojamento {location}",
            n_results=accom_needed * 2,
            category_filter=ACCOMMODATION_BUNDLES,
            max_cost=tc["max_cost"],
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
        )
        for p in accom_r["pois"]:
            if p["id"] not in existing_ids:
                candidate_pois.append(p)
                existing_ids.add(p["id"])

    # Hard filter geografico
    if not is_corridor and radius_km <= 200.0:
        candidate_pois = [p for p in candidate_pois
                          if _within_radius(p["lat"], p["lon"], center_lat, center_lon, radius_km)]

    # Verificar alojamento disponível
    if include_accommodation:
        has_accom = any(p["category"] in ACCOMMODATION_BUNDLES for p in candidate_pois)
        if not has_accom:
            include_accommodation = False
            if verbose:
                print(f"  AVISO: sem alojamento na area, constraint relaxada")

    # Clamp duracoes
    for p in candidate_pois:
        cat = p.get("category", "")
        if cat in DURATION_RANGES:
            d_min, d_max = DURATION_RANGES[cat]
            p["duration"] = max(d_min, min(d_max, p["duration"]))

    if not candidate_pois:
        print(f"  ERRO: 0 POIs apos filtros")
        return None

    if verbose:
        print(f"  {len(candidate_pois)} candidatos | "
              f"bbox ({lat_min:.2f},{lon_min:.2f})-({lat_max:.2f},{lon_max:.2f})")

    # Construir POI objects
    ACCOM_MIN_COST = 60.0
    optimizer_pois = []
    for p in candidate_pois:
        cost = p["cost"]
        if p["category"] in ACCOMMODATION_BUNDLES and cost < ACCOM_MIN_COST:
            cost = ACCOM_MIN_COST
        optimizer_pois.append(POI(
            id=int(p["id"]), name=p["name"],
            lat=p["lat"], lon=p["lon"],
            category=p["category"], score=p["score"],
            duration=p["duration"],
            opening_time=p["opening_time"],
            closing_time=p["closing_time"],
            cost=cost,
        ))

    # Matriz de distâncias (Haversine + tabela de tempos)
    n = len(optimizer_pois)
    transport = tc.get("transport", "car")
    matrix = np.zeros((n, n))
    for i, pi in enumerate(optimizer_pois):
        for j, pj in enumerate(optimizer_pois):
            if i != j:
                d_km = haversine(pi.lat, pi.lon, pj.lat, pj.lon)
                matrix[i][j] = _travel_time(d_km, transport)

    # category_weights
    cat_weights = {cat: 0.8 for cat in categories}
    if include_accommodation:
        for cat in ACCOMMODATION_BUNDLES:
            cat_weights[cat] = 0.4

    num_people = tc.get("num_people", 1)
    num_rooms  = max(1, math.ceil(num_people / 2))
    num_days   = max(1, round(tc["max_time"] / 480))

    user_prefs = {
        "max_time":   tc["max_time"],
        "max_cost":   tc["max_cost"],
        "num_people": num_people,
        "num_rooms":  num_rooms,
        "preferred_categories": categories,
        "category_weights": cat_weights,
        "start_location": (optimizer_pois[0].lat, optimizer_pois[0].lon),
        "start_time": "09:00",
        "center_lat": center_lat if radius_km <= 200.0 else None,
        "center_lon": center_lon if radius_km <= 200.0 else None,
        "max_radius_km": radius_km,
        "mobility_issues": False,
        "has_children": tc.get("has_children", False),
        "elevation_matrix": None,
        "include_accommodation": include_accommodation,
        "has_nightlife": "bares_e_discotecas" in categories or "casinos" in categories,
        "max_days": num_days,
    }

    evaluator = RouteEvaluator(optimizer_pois, matrix, user_prefs)

    return {
        "optimizer_pois": optimizer_pois,
        "matrix": matrix,
        "evaluator": evaluator,
        "n_candidates": len(optimizer_pois),
        "user_prefs": user_prefs,
        "include_accommodation": include_accommodation,
    }


def run_algorithm(algo: str, optimizer_pois, matrix, evaluator) -> dict:
    t0 = time.time()
    if algo == "GA":
        opt = TourismGA(optimizer_pois, matrix, evaluator, population_size=50, n_generations=30)
    elif algo == "PSO":
        opt = TourismPSOA(optimizer_pois, matrix, evaluator, n_particles=20, n_iterations=30)
    elif algo == "ACO":
        opt = TourismACO(optimizer_pois, matrix, evaluator, n_ants=30, n_iterations=100)
    else:  # GREEDY
        opt = GreedyPlanner(optimizer_pois, matrix, evaluator)

    result = opt.optimize()
    elapsed = round(time.time() - t0, 2)

    fc = {}
    if result.get("route"):
        try:
            fc = evaluator.calculate_fitness_components(result["route"])
        except Exception:
            pass

    return {
        "algo":        algo,
        "status":      "ok",
        "fitness":     round(result.get("fitness", 0), 3),
        "n_pois":      len(result.get("route", [])),
        "elapsed_s":   elapsed,
        "time_util":   fc.get("time_utilization", 0),
        "cat_comp":    fc.get("category_component", 0),
        "div_comp":    fc.get("diversity_component", 0),
        "prox_comp":   fc.get("proximity_component", 0),
        "dist_pen":    fc.get("distance_penalty", 0),
        "unique_cats": fc.get("unique_categories", 0),
        "poi_names":   [p.name for p in (result.get("pois") or [])],
    }


def print_case_table(runs: list):
    ok = sorted([r for r in runs if r["status"] == "ok"],
                key=lambda r: -r["fitness"])
    if not ok:
        print("  [sem resultados OK]")
        return
    print(f"  {'Algo':<8} {'Fitness':>7} {'POIs':>4} {'t_util%':>7} {'cat%':>6} "
          f"{'div%':>6} {'prox%':>6} {'uniq':>4} {'secs':>5}")
    print(f"  {'-'*60}")
    for r in ok:
        winner = " <--" if r == ok[0] else ""
        print(f"  {r['algo']:<8} {r['fitness']:>7.3f} {r['n_pois']:>4} "
              f"{r.get('time_util',0):>7.1f} {r.get('cat_comp',0):>6.1f} "
              f"{r.get('div_comp',0):>6.1f} {r.get('prox_comp',0):>6.1f} "
              f"{r.get('unique_cats',0):>4} {r['elapsed_s']:>5.2f}{winner}")


def print_summary(all_results: dict, algos: list):
    wins  = {a: 0 for a in algos}
    fit_sum = {a: [] for a in algos}

    print(f"\n{'='*72}")
    print("TABELA RESUMO")
    print(f"{'='*72}")
    print(f"{'ID':<4} {'Label':<34} {'Dim':<6} {'Winner':>7} {'Delta GA-PSO':>12} {'Delta 1-last':>12}")
    print(f"{'-'*72}")

    for tc in TEST_CASES:
        tc_id = tc["id"]
        runs = all_results.get(tc_id, [])
        ok = sorted([r for r in runs if r["status"] == "ok"],
                    key=lambda r: -r["fitness"])
        if not ok:
            continue
        wins[ok[0]["algo"]] += 1
        for r in ok:
            fit_sum[r["algo"]].append(r["fitness"])

        w     = ok[0]
        last  = ok[-1]
        ga_r  = next((r for r in ok if r["algo"] == "GA"), None)
        pso_r = next((r for r in ok if r["algo"] == "PSO"), None)
        d_ga_pso = (ga_r["fitness"] - pso_r["fitness"]) if ga_r and pso_r else 0
        d_1_last = w["fitness"] - last["fitness"]
        print(f"{tc_id:<4} {tc['label']:<34} {tc['dim_time']:<6} {w['algo']:>7} "
              f"{d_ga_pso:>+12.3f} {d_1_last:>+12.3f}")

    print(f"\n  VITORIAS TOTAIS: " + "  ".join(f"{a}={wins[a]}" for a in algos))
    print(f"  FITNESS MEDIO:   " + "  ".join(
        f"{a}={sum(fit_sum[a])/len(fit_sum[a]):.3f}" if fit_sum[a] else f"{a}=N/A"
        for a in algos))

    print(f"\n  POR DURACAO:")
    for dim in ["curta", "media", "longa", "muito_longa"]:
        dim_wins = {a: 0 for a in algos}
        dim_fit  = {a: [] for a in algos}
        dim_cases = [t["id"] for t in TEST_CASES if t["dim_time"] == dim]
        for tc_id in dim_cases:
            runs = all_results.get(tc_id, [])
            ok = sorted([r for r in runs if r["status"] == "ok"], key=lambda r: -r["fitness"])
            if ok:
                dim_wins[ok[0]["algo"]] += 1
                for r in ok:
                    dim_fit[r["algo"]].append(r["fitness"])
        total = sum(dim_wins.values())
        if total == 0:
            continue
        vit  = "  ".join(f"{a}={dim_wins[a]}" for a in algos)
        meds = "  ".join(
            f"{a}={sum(dim_fit[a])/len(dim_fit[a]):.2f}" if dim_fit[a] else f"{a}=N/A"
            for a in algos)
        print(f"    {dim:<12}: wins [{vit}]  avg [{meds}]")

    print(f"\n  POR N_CANDIDATES:")
    buckets = {"<=20": [], "21-40": [], "41-60": [], ">60": []}
    for tc_id, runs in all_results.items():
        ok = sorted([r for r in runs if r["status"] == "ok"], key=lambda r: -r["fitness"])
        if not ok:
            continue
        nc = all_results[tc_id][0].get("n_candidates", 0) if all_results[tc_id] else 0
        # get n_candidates from the problem (stored in all_results metadata)
        key = "<=20" if nc <= 20 else ("21-40" if nc <= 40 else ("41-60" if nc <= 60 else ">60"))
        buckets[key].append((ok[0]["algo"], nc))
    for bucket, entries in buckets.items():
        if entries:
            from collections import Counter
            c = Counter(a for a, _ in entries)
            avg_nc = sum(nc for _, nc in entries) / len(entries)
            print(f"    n_cand {bucket} (avg={avg_nc:.0f}): " +
                  "  ".join(f"{a}={c.get(a,0)}" for a in algos))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--cases",  default=None, help="IDs separados por virgula (ex: S1,M2,L1)")
    parser.add_argument("--algos",  default=None, help="Algoritmos (ex: GA,PSO)")
    args = parser.parse_args()

    cases_to_run = TEST_CASES
    if args.cases:
        ids = {c.strip() for c in args.cases.split(",")}
        cases_to_run = [t for t in TEST_CASES if t["id"] in ids]
        if not cases_to_run:
            print(f"ERRO: IDs nao encontrados: {ids}")
            sys.exit(1)

    algos_to_run = ALGORITHMS
    if args.algos:
        algos_to_run = [a.strip().upper() for a in args.algos.split(",")]

    output_path = args.output or f"outputs/algo_comparison_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"COMPARACAO DE ALGORITMOS v2 (sem LLM)")
    print(f"  Casos     : {len(cases_to_run)} queries")
    print(f"  Algoritmos: {algos_to_run}")
    print(f"  Total runs: {len(cases_to_run) * len(algos_to_run)}")
    print(f"  Output    : {output_path}")
    print(f"{'='*70}\n")

    print("A carregar RAG e resolver geografico...")
    rag      = POI_RAG(data_file="data/portugal_todos_pois_final_enriched.json")
    resolver = LocationResolver()
    print("Pronto.\n")

    all_results  = {}
    all_metadata = {}
    total_cases  = len(cases_to_run)

    for ci, tc in enumerate(cases_to_run, 1):
        print(f"\n{'─'*70}")
        print(f"[{ci}/{total_cases}] {tc['id']} — {tc['label']}")
        print(f"  max_time={tc['max_time']}min  max_cost={tc['max_cost']}EUR  "
              f"loc={tc['location']}  cats={tc['categories']}")

        problem = build_problem(tc, rag, resolver, verbose=True)
        if problem is None:
            all_results[tc["id"]] = []
            continue

        nc = problem["n_candidates"]
        print(f"  n_candidates={nc}  transport={tc['transport']}\n")
        all_metadata[tc["id"]] = {"n_candidates": nc}

        runs = []
        for algo in algos_to_run:
            print(f"  {algo}...", end=" ", flush=True)
            r = run_algorithm(
                algo,
                problem["optimizer_pois"],
                problem["matrix"],
                problem["evaluator"],
            )
            r["n_candidates"] = nc
            runs.append(r)
            print(f"fitness={r['fitness']:.3f}  n={r['n_pois']}  {r['elapsed_s']}s")

        all_results[tc["id"]] = runs
        print()
        print_case_table(runs)

    print_summary(all_results, algos_to_run)

    output = {
        "timestamp":  datetime.now().isoformat(),
        "algorithms": algos_to_run,
        "n_cases":    len(cases_to_run),
        "cases": {
            tc["id"]: {
                "label":        tc["label"],
                "dim_time":     tc["dim_time"],
                "dim_location": tc["dim_location"],
                "params": {
                    "max_time":  tc["max_time"],
                    "max_cost":  tc["max_cost"],
                    "location":  tc["location"],
                    "categories": tc["categories"],
                    "transport": tc["transport"],
                },
                "n_candidates": all_metadata.get(tc["id"], {}).get("n_candidates", 0),
                "runs": all_results.get(tc["id"], []),
            }
            for tc in cases_to_run
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  JSON: {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
