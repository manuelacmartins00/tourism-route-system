"""
evals/build_fixtures_grid100.py
================================
Gera 100 fixtures estratificadas para o grid search de hiperparâmetros.

Estratificação multi-dimensional a partir do Excel:
  - Perfil (A-G): 14-15 por perfil
  - Dentro de cada perfil: duração (curta/média/longa), transporte,
    has_children, mobilidade, grupo (solo/pequeno/grande), corredor

Sem LLM — usa direct_preferences igual ao build_fixtures_from_excel.py.

Uso:
    python evals/build_fixtures_grid100.py
    python evals/build_fixtures_grid100.py --out data/bench_fixtures_grid100 --seed 42
"""
import os, sys, json, time, argparse, random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openpyxl
from main_system import TourismRouteSystem

# ── Reutilizar mapeamentos do script original ─────────────────────────────────
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
    "foot": "foot", "car": "car",
    "public_transport": "public_transport",
    "public_transport_train": "public_transport",
    "public_transport_bus": "public_transport",
}

def parse_budget(s):
    s = str(s).strip()
    if s.startswith(">"): return 6000.0
    if "-" in s:
        try: return float(s.split("-")[1])
        except: pass
    try: return float(s)
    except: return 500.0

def extract_perfil_key(v):
    return str(v).strip()[0].upper()

def build_locations(vals):
    locs = []
    for v in vals[2:8]:
        if v and str(v).strip().lower() not in ("nenhuma", "none", ""):
            locs.append(str(v).strip())
    return locs

def row_to_prefs(vals):
    perfil_key = extract_perfil_key(vals[8] or "A")
    cats       = PERFIL_CATS.get(perfil_key, PERFIL_CATS["A"])
    interests  = PERFIL_INTERESTS.get(perfil_key, [])
    locs       = build_locations(vals)
    location   = locs[0] if locs else "Lisboa"
    duration   = int(vals[9] or 1)
    max_time   = duration * 480
    max_cost   = parse_budget(vals[10] or "250-500")
    transport  = TRANSPORT_MAP.get(str(vals[11] or "car").strip(), "car")
    has_children = str(vals[12] or "Não").strip().lower().startswith("s")
    mobility     = str(vals[13] or "Não").strip().lower().startswith("s")
    n_people   = int(vals[1] or 1)
    incl_accom = bool(vals[14]) if len(vals) > 14 and vals[14] is not None else True
    incl_meals = bool(vals[15]) if len(vals) > 15 and vals[15] is not None else True
    nightlife  = perfil_key == "B"
    if nightlife:
        max_time += duration * 180
    return {
        "max_time": max_time, "max_cost": max_cost,
        "preferred_categories": cats, "category_weights": {c: 1.0 for c in cats},
        "start_time": "09:00", "interests": interests, "secondary_tags": [],
        "location": location, "locations": locs,
        "locations_ordered": len(locs) > 1, "missing_fields": [],
        "transport_mode": transport, "mobility_issues": mobility,
        "num_people": n_people, "has_children": has_children,
        "last_day_end_time": None,
        "end_location": locs[-1] if len(locs) > 1 else None,
        "include_accommodation": incl_accom, "include_meals": incl_meals,
        "nightlife_suggested": nightlife, "start_date": None,
    }

# ── Funções de bucket para estratificação ─────────────────────────────────────
def dur_bucket(days):
    if days <= 2:   return "short"
    if days <= 5:   return "medium"
    return "long"

def group_bucket(n):
    if n == 1:  return "solo"
    if n <= 3:  return "small"
    return "large"

def transport_bucket(t):
    t = str(t or "car").strip().lower()
    if t == "foot":   return "foot"
    if t == "car":    return "car"
    return "transit"

def row_features(vals):
    """Extrai features de estratificação de uma linha do Excel."""
    return {
        "perfil":      extract_perfil_key(vals[8] or "A"),
        "dur_bucket":  dur_bucket(int(vals[9] or 1)),
        "transport":   transport_bucket(vals[11]),
        "children":    str(vals[12] or "Não").strip().lower().startswith("s"),
        "mobility":    str(vals[13] or "Não").strip().lower().startswith("s"),
        "group":       group_bucket(int(vals[1] or 1)),
        "corridor":    sum(1 for v in vals[2:8]
                          if v and str(v).strip().lower() not in ("nenhuma","none","")) > 1,
    }

# ── Algoritmo de selecção: greedy diversity maximization ──────────────────────
def stratified_select(all_rows, n=100, seed=42):
    """
    Selecciona n linhas do Excel com estratificação multi-dimensional.

    Algoritmo:
      1. Para cada perfil: quota de floor(n/7) ou floor(n/7)+1 slots
      2. Dentro de cada perfil: garantir pelo menos 1 de cada duration bucket
         e 1 de cada transport mode; preencher resto com greedy diversity
         (cada nova linha maximiza a diferença em relação às já seleccionadas)
    """
    rng = random.Random(seed)

    # Agrupar por perfil
    by_perfil = defaultdict(list)
    for vals in all_rows:
        k = extract_perfil_key(vals[8] or "A")
        by_perfil[k].append(vals)

    perfis = sorted(by_perfil.keys())
    base_quota = n // len(perfis)      # 14
    extra      = n % len(perfis)       # 2 (para os primeiros perfis)

    selected = []

    for i, perfil in enumerate(perfis):
        quota = base_quota + (1 if i < extra else 0)
        pool  = list(by_perfil[perfil])
        rng.shuffle(pool)

        chosen = []
        chosen_feats = []

        # Fase 1: garantir cobertura mínima (1 por duration bucket, 1 por transport)
        buckets_needed = {"short", "medium", "long"}
        transports_needed = {"foot", "car", "transit"}
        for vals in pool:
            feats = row_features(vals)
            covered_dur   = feats["dur_bucket"] in buckets_needed
            covered_trans = feats["transport"]  in transports_needed
            if covered_dur or covered_trans:
                chosen.append(vals)
                chosen_feats.append(feats)
                buckets_needed.discard(feats["dur_bucket"])
                transports_needed.discard(feats["transport"])
            if not buckets_needed and not transports_needed:
                break

        # Fase 2: preencher restante com greedy diversity
        remaining_pool = [v for v in pool if v not in chosen]
        rng.shuffle(remaining_pool)

        def diversity_score(feats, chosen_feats):
            """Conta quantas combinações de features ainda não representadas cobre."""
            if not chosen_feats:
                return 1
            score = 0
            feat_keys = ["dur_bucket", "transport", "children", "mobility", "group", "corridor"]
            for k in feat_keys:
                vals_chosen = {f[k] for f in chosen_feats}
                if feats[k] not in vals_chosen:
                    score += 1
            return score

        while len(chosen) < quota and remaining_pool:
            best_val   = None
            best_score = -1
            # Avaliar os primeiros 30 candidatos (eficiência)
            for candidate in remaining_pool[:30]:
                feats = row_features(candidate)
                score = diversity_score(feats, chosen_feats)
                if score > best_score:
                    best_score = score
                    best_val   = candidate
            if best_val is None:
                best_val = remaining_pool[0]
            chosen.append(best_val)
            chosen_feats.append(row_features(best_val))
            remaining_pool.remove(best_val)

        selected.extend(chosen[:quota])
        print(f"  Perfil {perfil}: {len(chosen[:quota])} linhas  "
              f"| dur={set(f['dur_bucket'] for f in chosen_feats[:quota])} "
              f"| transport={set(f['transport'] for f in chosen_feats[:quota])}")

    return selected


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--excel", default="Prompts_teste_standard.xlsx")
    p.add_argument("--out",   default="data/bench_fixtures_grid100")
    p.add_argument("--n",     type=int, default=100)
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Carregar Excel
    wb = openpyxl.load_workbook(args.excel, data_only=True)
    ws = wb.active
    all_rows = [tuple(r) for r in ws.iter_rows(min_row=2, values_only=True)
                if r[0] is not None]
    print(f"[excel] {len(all_rows)} linhas carregadas")

    print(f"\n[seleccao] Estratificacao multi-dimensional (n={args.n}, seed={args.seed})")
    rows = stratified_select(all_rows, n=args.n, seed=args.seed)
    print(f"\n[seleccao] Total: {len(rows)} linhas seleccionadas")

    # Mostrar distribuição final
    from collections import Counter
    feats_all = [row_features(r) for r in rows]
    print(f"\n  dur_bucket : {dict(Counter(f['dur_bucket'] for f in feats_all))}")
    print(f"  transport  : {dict(Counter(f['transport']  for f in feats_all))}")
    print(f"  children   : {dict(Counter(f['children']   for f in feats_all))}")
    print(f"  mobility   : {dict(Counter(f['mobility']   for f in feats_all))}")
    print(f"  group      : {dict(Counter(f['group']      for f in feats_all))}")
    print(f"  corridor   : {dict(Counter(f['corridor']   for f in feats_all))}")

    # Inicializar sistema (RAG apenas, sem LLM)
    print("\n[sistema] A inicializar RAG...")
    system = TourismRouteSystem(api_key="dummy")

    ok = skip = err = 0
    t0 = time.perf_counter()

    for i, vals in enumerate(rows, 1):
        row_id = int(vals[0])
        perfil = extract_perfil_key(vals[8] or "A")
        out_fp = out_dir / f"{row_id}.json"

        if args.skip_existing and out_fp.exists():
            print(f"[{i:03d}/{len(rows)}] {row_id} (perfil {perfil}) — skip")
            skip += 1
            continue

        prefs = row_to_prefs(vals)
        locs_str = "+".join(prefs["locations"][:2]) if prefs["locations"] else "?"
        print(f"[{i:03d}/{len(rows)}] {row_id} | {perfil} | {locs_str} | "
              f"{int(prefs['max_time']//480)}d | {prefs['transport_mode']}", end=" ")

        try:
            system.plan_route(
                user_query=f"Fixture {row_id}",
                use_shap=False, verbose=False,
                generate_map=False, generate_explanation=False,
                include_accommodation=prefs["include_accommodation"],
                include_meals=prefs["include_meals"],
                fixture_capture_path=str(out_fp),
                direct_preferences=prefs,
            )
            if out_fp.exists():
                with open(out_fp, encoding="utf-8") as f:
                    fix = json.load(f)
                fix["scenario_id"] = str(row_id)
                fix["profile"]     = perfil
                fix["excel_row"]   = row_id
                with open(out_fp, "w", encoding="utf-8") as f:
                    json.dump(fix, f, ensure_ascii=False)
                print(f"-> {fix.get('n_pois','?')} POIs")
                ok += 1
            else:
                print("-> SKIP (sem fixture)")
                skip += 1
        except Exception as e:
            print(f"-> ERRO: {e}")
            err += 1

    elapsed = time.perf_counter() - t0
    print(f"\n[excel] Concluido em {elapsed:.0f}s — {ok} OK | {skip} skip | {err} erros")
    print(f"[excel] Fixtures em: {out_dir}")


if __name__ == "__main__":
    main()
