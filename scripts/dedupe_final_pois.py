# scripts/dedupe_final_pois.py
"""
Remove entradas duplicadas (mesmo nome, coordenadas a <=1km, id diferente)
de data/portugal_todos_pois_final_enriched.json -- o ficheiro realmente
usado em producao (main_system.py:24).

Origem do problema: portugal_todos_pois_limpos_osm_and_google_enriched.json
e uma concatenacao de dois lotes (batch1=5640 + batch2=5755) com apenas 6
duplicados removidos explicitamente no merge; os mesmos 47 pares por-nome
sobrevivem sem alteracao no ficheiro final.

De cada par mantem-se o registo mais completo (mais campos preenchidos);
o outro e removido. Grava um backup antes de escrever por cima do ficheiro
original.

Uso:
  python scripts/dedupe_final_pois.py                 # aplica e grava
  python scripts/dedupe_final_pois.py --dry-run        # so mostra o que faria
"""

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

DEFAULT_PATH = "data/portugal_todos_pois_final_enriched.json"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = math.radians
    dlat, dlon = r(lat2 - lat1), r(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(r(lat1)) * math.cos(r(lat2)) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def richness(poi: dict) -> int:
    """Conta campos nao-vazios (recursivamente) -- usado para decidir qual
    dos dois duplicados manter."""
    def _count(v) -> int:
        if v is None or v == "" or v == [] or v == {}:
            return 0
        if isinstance(v, dict):
            return sum(_count(x) for x in v.values())
        if isinstance(v, list):
            return sum(_count(x) for x in v)
        return 1
    return _count(poi)


def find_duplicate_groups(pois: list, raio_km: float = 1.0) -> list:
    by_name = defaultdict(list)
    for p in pois:
        name = (p.get("name") or "").strip()
        if name:
            by_name[name].append(p)

    groups = []
    for name, group in by_name.items():
        ids = {p["id"] for p in group}
        if len(ids) < 2:
            continue
        coords = [(p["location"]["lat"], p["location"]["lon"]) for p in group if p.get("location")]
        max_dist = 0.0
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                max_dist = max(max_dist, haversine_km(*coords[i], *coords[j]))
        if max_dist <= raio_km:
            groups.append(group)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_PATH)
    ap.add_argument("--raio-km", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true",
                     help="mostra o que seria removido sem escrever nada")
    args = ap.parse_args()

    path = Path(args.input)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pois = data["pois"]
    n_before = len(pois)

    groups = find_duplicate_groups(pois, args.raio_km)
    to_remove_ids = set()
    print(f"[dedupe_final_pois] {len(groups)} pares duplicados encontrados\n")
    for group in groups:
        keep = max(group, key=richness)
        drop = [p for p in group if p is not keep]
        for d in drop:
            to_remove_ids.add(d["id"])
        print(f"  MANTÉM id={keep['id']:>7}  |  REMOVE id={drop[0]['id']:>7}  |  {keep['name']}")

    cleaned = [p for p in pois if p["id"] not in to_remove_ids]
    n_after = len(cleaned)
    print(f"\nTotal antes: {n_before}")
    print(f"Removidos:   {len(to_remove_ids)}")
    print(f"Total depois: {n_after}")

    if args.dry_run:
        print("\n[dry-run] nada foi escrito.")
        return

    backup_path = path.with_suffix(path.suffix + ".pre_dedupe_bak")
    shutil.copy(path, backup_path)
    print(f"\nBackup gravado em: {backup_path}")

    data["pois"] = cleaned
    data["total"] = n_after
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Ficheiro atualizado: {path} ({n_after} POIs)")


if __name__ == "__main__":
    main()
