"""
Analisa o JSON dos POIs e reporta campos vazios/curtos na descrição.
Uso: python scripts/check_poi_descriptions.py
"""
import json
from collections import defaultdict

DATA_FILE = "data/portugal_todos_pois_final_enriched.json"
SHORT_THRESHOLD = 30  # caracteres — abaixo disto considera-se descrição insuficiente

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

pois = data.get("pois", data) if isinstance(data, dict) else data
total = len(pois)

empty       = []  # sem descrição ou string vazia
short       = []  # descrição com menos de SHORT_THRESHOLD caracteres
ok          = []  # descrição adequada

by_category_empty = defaultdict(int)
by_category_total = defaultdict(int)

for poi in pois:
    desc = poi.get("description", "") or ""
    cat  = poi.get("source", {}).get("bundle", poi.get("category", "desconhecido"))
    by_category_total[cat] += 1

    if not desc.strip():
        empty.append(poi)
        by_category_empty[cat] += 1
    elif len(desc.strip()) < SHORT_THRESHOLD:
        short.append(poi)
        by_category_empty[cat] += 1  # conta como problemático
    else:
        ok.append(poi)

n_empty = len(empty)
n_short = len(short)
n_ok    = len(ok)
n_problem = n_empty + n_short

print(f"\n{'='*60}")
print(f"  ANÁLISE DE DESCRIÇÕES — {DATA_FILE}")
print(f"{'='*60}")
print(f"  Total de POIs         : {total:>7,}")
print(f"  Sem descrição (vazios): {n_empty:>7,}  ({n_empty/total*100:.1f}%)")
print(f"  Descrição muito curta : {n_short:>7,}  ({n_short/total*100:.1f}%)  [< {SHORT_THRESHOLD} chars]")
print(f"  Descrição adequada    : {n_ok:>7,}  ({n_ok/total*100:.1f}%)")
print(f"  Total problemáticos   : {n_problem:>7,}  ({n_problem/total*100:.1f}%)")
print(f"{'='*60}")

print(f"\n  Por categoria (problemáticos / total):")
print(f"  {'Categoria':<35} {'Prob':>6} {'Total':>7} {'%':>6}")
print(f"  {'-'*55}")
for cat, tot in sorted(by_category_total.items(), key=lambda x: -by_category_empty[x[0]]):
    prob = by_category_empty[cat]
    pct  = prob / tot * 100 if tot > 0 else 0
    bar  = "█" * int(pct / 5)
    print(f"  {cat:<35} {prob:>6} {tot:>7} {pct:>5.1f}%  {bar}")

print(f"\n  Exemplos de POIs sem descrição (primeiros 10):")
for poi in empty[:10]:
    name = poi.get("name", "?")
    cat  = poi.get("source", {}).get("bundle", "?")
    region = poi.get("source", {}).get("region", "?")
    print(f"    - {name} ({cat}, {region})")

if short:
    print(f"\n  Exemplos de POIs com descrição muito curta (primeiros 10):")
    for poi in short[:10]:
        name = poi.get("name", "?")
        desc = (poi.get("description") or "").strip()
        print(f"    - {name}: \"{desc}\"")

print(f"\n{'='*60}\n")
