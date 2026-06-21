"""
evals/build_fixtures_from_excel.py
====================================
Gera fixtures de benchmark directamente a partir do Excel de features,
SEM usar LLM. Lê os campos estruturados (cidade, perfil, duracao, etc.),
constrói UserPreferences, corre RAG + distâncias, guarda fixture JSON.

490 fixtures em ~5-10 minutos (sem rate limiting, sem API calls).

Uso:
    python evals/build_fixtures_from_excel.py \
        --excel  data/Prompts_teste_standard.xlsx \
        --out    data/bench_fixtures_direct \
        [--n 490]           # quantos rows usar (default: todos)
        [--skip-existing]
"""
import os, sys, json, time, argparse, random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openpyxl
from main_system import TourismRouteSystem

# ── Mapeamentos ───────────────────────────────────────────────────────────────

PERFIL_CATS = {
    "A": ["turismo_activo", "parques_e_reservas", "espacos_verdes", "praias", "barragens"],
    "B": ["bares_e_discotecas", "restaurantes_e_cafes", "monumentos"],
    "C": ["monumentos", "museus_e_palacios", "arqueologia", "eventos"],
    "D": ["parques_de_diversao", "zoos_e_aquarios", "ciencia_e_conhecimento", "praias"],
    "E": ["termas", "talassoterapia", "espacos_verdes", "parques_e_reservas"],
    "F": ["restaurantes_e_cafes", "eventos", "mercados"],
    "G": ["monumentos", "praias", "parques_e_reservas", "restaurantes_e_cafes"],
}

PERFIL_INTERESTS = {
    "A": ["natureza", "aventura", "ar livre"],
    "B": ["vida noturna", "bares", "entretenimento"],
    "C": ["cultura", "história", "monumentos"],
    "D": ["família", "crianças", "diversão"],
    "E": ["bem-estar", "relaxamento", "spa"],
    "F": ["gastronomia", "vinhos", "culinária"],
    "G": ["road trip", "paisagens", "fotografia"],
}

TRANSPORT_MAP = {
    "foot":                   "foot",
    "car":                    "car",
    "public_transport":       "public_transport",
    "public_transport_train": "public_transport",
    "public_transport_bus":   "public_transport",
}

def parse_budget(s: str) -> float:
    """Devolve o limite superior do intervalo de orçamento."""
    s = str(s).strip()
    if s.startswith(">"):
        return 6000.0
    if "-" in s:
        parts = s.split("-")
        try:
            return float(parts[1])
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        return 500.0

def extract_perfil_key(perfil_str: str) -> str:
    """'A - Natureza & Aventura' → 'A'"""
    return str(perfil_str).strip()[0].upper()

def build_locations(row_values: list) -> list:
    """Colunas 2-7: cidade + até 5 adicionais; filtra 'Nenhuma'."""
    locs = []
    for v in row_values[2:8]:
        if v and str(v).strip().lower() not in ("nenhuma", "none", ""):
            locs.append(str(v).strip())
    return locs

def row_to_prefs(row) -> dict:
    """Converte uma linha do Excel num dict compatível com UserPreferences."""
    vals = list(row)
    perfil_key = extract_perfil_key(vals[8] or "A")
    cats       = PERFIL_CATS.get(perfil_key, PERFIL_CATS["A"])
    interests  = PERFIL_INTERESTS.get(perfil_key, [])
    locs       = build_locations(vals)
    location   = locs[0] if locs else "Lisboa"
    duration   = int(vals[9] or 1)
    max_time   = duration * 480               # minutos (8h/dia)
    max_cost   = parse_budget(vals[10] or "250-500")
    transport  = TRANSPORT_MAP.get(str(vals[11] or "car").strip(), "car")
    children_raw = str(vals[12] or "Não").strip()
    has_children = children_raw.lower().startswith("s")
    mobility_raw = str(vals[13] or "Não").strip()
    mobility     = mobility_raw.lower().startswith("s")
    n_people   = int(vals[1] or 1)
    incl_accom = bool(vals[14]) if vals[14] is not None else True
    incl_meals = bool(vals[15]) if vals[15] is not None else True

    # Vida noturna → perfil B
    nightlife = perfil_key == "B"
    if nightlife:
        max_time += duration * 180   # +3h/noite

    # Pesos de categoria uniformes
    weights = {c: 1.0 for c in cats}

    return {
        "max_time":              max_time,
        "max_cost":              max_cost,
        "preferred_categories":  cats,
        "category_weights":      weights,
        "start_time":            "09:00",
        "interests":             interests,
        "secondary_tags":        [],
        "location":              location,
        "locations":             locs,
        "locations_ordered":     len(locs) > 1,
        "missing_fields":        [],
        "transport_mode":        transport,
        "mobility_issues":       mobility,
        "num_people":            n_people,
        "has_children":          has_children,
        "last_day_end_time":     None,
        "end_location":          locs[-1] if len(locs) > 1 else None,
        "include_accommodation": incl_accom,
        "include_meals":         incl_meals,
        "nightlife_suggested":   nightlife,
        "start_date":            None,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--excel",         default="data/Prompts_teste_standard.xlsx")
    p.add_argument("--out",           default="data/bench_fixtures_direct")
    p.add_argument("--n",             type=int, default=490,
                   help="Quantas linhas do Excel usar (default 490, max 2000)")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--seed",          type=int, default=123,
                   help="Seed para amostragem estratificada (default 123)")
    return p.parse_args()


def stratified_sample(rows, n: int, seed: int):
    """Amostra estratificada por perfil."""
    by_perfil = defaultdict(list)
    for r in rows:
        k = extract_perfil_key(r[8] or "A")
        by_perfil[k].append(r)
    rng = random.Random(seed)
    per_perfil = n // len(by_perfil)
    selected = []
    for k in sorted(by_perfil):
        pool = by_perfil[k]
        rng.shuffle(pool)
        selected.extend(pool[:per_perfil])
    # Completar até n se n não for divisível
    remaining = n - len(selected)
    all_remaining = [r for k in sorted(by_perfil) for r in by_perfil[k][per_perfil:]]
    rng.shuffle(all_remaining)
    selected.extend(all_remaining[:remaining])
    return selected


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Carregar Excel
    wb = openpyxl.load_workbook(args.excel)
    ws = wb.active
    all_rows = [tuple(r) for r in ws.iter_rows(min_row=2, values_only=True)
                if r[0] is not None]
    print(f"[excel] {len(all_rows)} linhas no Excel")

    rows = stratified_sample(all_rows, min(args.n, len(all_rows)), args.seed)
    print(f"[excel] {len(rows)} cenários seleccionados (estratificado, seed={args.seed})")
    print(f"[excel] Output: {out_dir}\n")

    # Inicializar sistema (só RAG — sem LLM)
    system = TourismRouteSystem(api_key="dummy")   # key não é usada

    ok = skip = err = 0
    t0_total = time.perf_counter()

    for i, row in enumerate(rows, 1):
        row_id  = int(row[0])
        perfil  = extract_perfil_key(row[8] or "A")
        out_fp  = out_dir / f"{row_id}.json"

        if args.skip_existing and out_fp.exists():
            print(f"[{i:03d}/{len(rows)}] {row_id} (perfil {perfil}) — skip")
            skip += 1
            continue

        prefs = row_to_prefs(row)
        print(f"[{i:03d}/{len(rows)}] {row_id} (perfil {perfil}) "
              f"{prefs['location']} {int(prefs['max_time']//480)}d "
              f"{prefs['transport_mode']}")

        try:
            result = system.plan_route(
                user_query=f"Fixture {row_id}",   # ignorado quando direct_preferences fornecido
                use_shap=False,
                verbose=False,
                generate_map=False,
                generate_explanation=False,
                include_accommodation=prefs["include_accommodation"],
                include_meals=prefs["include_meals"],
                fixture_capture_path=str(out_fp),
                direct_preferences=prefs,
            )

            if out_fp.exists():
                # Enriquecer com metadados
                with open(out_fp, encoding="utf-8") as f:
                    fix = json.load(f)
                fix["scenario_id"] = str(row_id)
                fix["profile"]     = perfil
                fix["excel_row"]   = row_id
                with open(out_fp, "w", encoding="utf-8") as f:
                    json.dump(fix, f, ensure_ascii=False)
                print(f"         -> {fix.get('n_pois','?')} POIs, algo={fix.get('selected_algo','?')}")
                ok += 1
            else:
                status = result.get("status") or result.get("error", "sem fixture")
                print(f"         -> SKIP ({status})")
                skip += 1

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"         -> ERRO: {e}")
            err += 1

    elapsed = time.perf_counter() - t0_total
    print(f"\n[excel] Concluido em {elapsed:.0f}s — {ok} OK | {skip} skip | {err} erros")
    print(f"[excel] Fixtures em: {out_dir}")


if __name__ == "__main__":
    main()
