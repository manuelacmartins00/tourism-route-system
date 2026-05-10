"""
Analisa descricoes dos POIs e classifica o seu conteudo por regex.
Uso: python scripts/check_poi_descriptions.py
"""
import json
import re
from collections import defaultdict, Counter

DATA_FILE = "data/portugal_todos_pois_final_enriched.json"

# Padroes para detetar conteudo na descricao
SCHEDULE_PATTERNS = [
    r'\b\d{1,2}[h:]\d{2}\b',                      # 09:00 / 9h00
    r'\b(segunda|terca|quarta|quinta|sexta|sabado|domingo)\b',
    r'\b(seg|ter|qua|qui|sex|sab|dom)\b',
    r'\b(aberto|encerrado|encerra|fecha|abre|funcionamento|horario)\b',
    r'\b(janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\b',
    r'\b(diariamente|semanalmente|fins de semana)\b',
]

COST_PATTERNS = [
    r'€\s*\d',
    r'\d+\s*(euro|eur)\b',
    r'\b(gratuito|gratis|entrada livre|free|sem custo|entrada gratuita)\b',
    r'\b(bilhete|ingresso|entrada)\s*[:\-]?\s*\d',
    r'\bpreco\b',
]

SCHEDULE_RE = re.compile('|'.join(SCHEDULE_PATTERNS), re.IGNORECASE)
COST_RE     = re.compile('|'.join(COST_PATTERNS), re.IGNORECASE)

SHORT_THRESHOLD = 50

def classify(desc):
    if not desc or not desc.strip():
        return "vazia"
    d = desc.strip()
    if len(d) < SHORT_THRESHOLD:
        return "muito_curta"
    has_schedule = bool(SCHEDULE_RE.search(d))
    has_cost     = bool(COST_RE.search(d))
    has_desc     = len(d) >= SHORT_THRESHOLD
    if has_schedule and has_cost:
        return "horario+custo"
    if has_schedule and has_desc:
        return "horario+descricao"
    if has_cost and has_desc:
        return "custo+descricao"
    if has_schedule:
        return "so_horario"
    if has_cost:
        return "so_custo"
    return "descricao_geral"

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

pois = data.get("pois", data) if isinstance(data, dict) else data
total = len(pois)

classes = Counter()
by_category = defaultdict(Counter)
samples = defaultdict(list)

for poi in pois:
    desc = poi.get("description", "") or ""
    cat  = poi.get("source", {}).get("bundle", poi.get("category", "desconhecido"))
    cls  = classify(desc)
    classes[cls] += 1
    by_category[cat][cls] += 1
    if len(samples[cls]) < 3:
        samples[cls].append((poi.get("name", "?"), cat, desc[:120]))

print(f"\n{'='*65}")
print(f"  CLASSIFICACAO DE DESCRICOES -- {total} POIs")
print(f"{'='*65}")
print(f"  {'Classe':<25} {'N':>6} {'%':>6}")
print(f"  {'-'*38}")
for cls, n in classes.most_common():
    print(f"  {cls:<25} {n:>6} {n/total*100:>5.1f}%")

print(f"\n{'='*65}")
print(f"  POR CATEGORIA (top classes)")
print(f"  {'Categoria':<30} {'vazia':>6} {'curta':>6} {'hor+desc':>8} {'hor+cst':>7} {'desc_g':>6}")
print(f"  {'-'*65}")
for cat in sorted(by_category, key=lambda c: sum(by_category[c].values()), reverse=True)[:20]:
    cc = by_category[cat]
    tot_cat = sum(cc.values())
    print(f"  {cat:<30} {cc.get('vazia',0):>6} {cc.get('muito_curta',0):>6} "
          f"{cc.get('horario+descricao',0):>8} {cc.get('horario+custo',0):>7} "
          f"{cc.get('descricao_geral',0):>6}  (total {tot_cat})")

print(f"\n{'='*65}")
print(f"  AMOSTRAS POR CLASSE")
for cls in ["vazia", "muito_curta", "so_horario", "so_custo", "descricao_geral", "horario+descricao", "horario+custo"]:
    if samples[cls]:
        print(f"\n  [{cls}]")
        for name, cat, excerpt in samples[cls]:
            trunc = excerpt.replace('\n', ' ')
            print(f"    {name} ({cat}): \"{trunc}\"")

print(f"\n{'='*65}\n")
