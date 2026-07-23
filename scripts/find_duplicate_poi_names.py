# scripts/find_duplicate_poi_names.py
"""
Procura entradas com o mesmo nome (ts_poi_nome) mas entity_id diferente em
data/portugal_todos_pois_limpos_osm_and_google_enriched.json.

Nomes genericos (ex: "Camara Municipal") repetem-se legitimamente em varias
localidades -- por isso cada grupo e classificado pela distancia entre as
coordenadas:
  - "provavel duplicado"      : todas as ocorrencias a <1km umas das outras
  - "mesmo nome, locais diferentes" : ocorrencias espalhadas por mais de 1km

Uso:
  python scripts/find_duplicate_poi_names.py
  python scripts/find_duplicate_poi_names.py --input data/outro_ficheiro.json
  python scripts/find_duplicate_poi_names.py --raio-km 0.5 --csv outputs/duplicados.csv
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = math.radians
    dlat, dlon = r(lat2 - lat1), r(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(r(lat1)) * math.cos(r(lat2)) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def load_pois(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["pois"] if isinstance(data, dict) and "pois" in data else data


def get_coords(poi: dict):
    coords = (poi.get("coordenadas") or {}).get("coordinates")
    if coords and len(coords) == 2:
        lon, lat = coords
        return lat, lon
    geo = poi.get("geos_geocoords")
    if geo and "," in geo:
        lat_s, lon_s = geo.split(",", 1)
        try:
            return float(lat_s), float(lon_s)
        except ValueError:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/portugal_todos_pois_limpos_osm_and_google_enriched.json")
    ap.add_argument("--raio-km", type=float, default=1.0,
                     help="distancia maxima entre ocorrencias para classificar como 'provavel duplicado' (default 1.0km)")
    ap.add_argument("--csv", default=None, help="caminho para gravar o relatorio completo em CSV")
    args = ap.parse_args()

    pois = load_pois(Path(args.input))
    print(f"[find_duplicate_poi_names] {len(pois)} POIs carregados de {args.input}\n")

    by_name = defaultdict(list)
    for p in pois:
        name = (p.get("ts_poi_nome") or "").strip()
        if name:
            by_name[name].append(p)

    rows = []
    for name, group in by_name.items():
        ids = {p.get("entity_id") for p in group}
        if len(ids) < 2:
            continue  # mesmo nome mas so 1 entity_id (duplicado exato, nao e o caso que procuramos)

        coords = [get_coords(p) for p in group]
        coords = [c for c in coords if c is not None]
        max_dist = 0.0
        if len(coords) >= 2:
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                    max_dist = max(max_dist, d)

        classification = "provavel duplicado" if max_dist <= args.raio_km else "mesmo nome, locais diferentes"

        for p in group:
            rows.append({
                "nome": name,
                "n_ocorrencias": len(group),
                "distancia_max_km": round(max_dist, 3),
                "classificacao": classification,
                "entity_id": p.get("entity_id"),
                "bundle": p.get("bundle"),
                "regiao": p.get("regiao_origem"),
                "morada": p.get("ts_poi_morada"),
                "lat_lon": get_coords(p),
            })

    rows.sort(key=lambda r: (r["classificacao"] != "provavel duplicado", -r["n_ocorrencias"], r["nome"]))

    n_provaveis = len({r["nome"] for r in rows if r["classificacao"] == "provavel duplicado"})
    n_mesmo_nome = len({r["nome"] for r in rows if r["classificacao"] == "mesmo nome, locais diferentes"})
    print(f"Nomes com >1 entity_id: {n_provaveis + n_mesmo_nome}")
    print(f"  -> provavel duplicado (<= {args.raio_km}km entre ocorrencias): {n_provaveis}")
    print(f"  -> mesmo nome, locais diferentes (> {args.raio_km}km): {n_mesmo_nome}\n")

    print("=== Provaveis duplicados (amostra) ===")
    shown = set()
    for r in rows:
        if r["classificacao"] != "provavel duplicado" or r["nome"] in shown:
            continue
        shown.add(r["nome"])
        dupes = [x for x in rows if x["nome"] == r["nome"]]
        print(f"\n{r['nome']}  ({len(dupes)}x, dist.max={r['distancia_max_km']}km, bundle={r['bundle']})")
        for d in dupes:
            print(f"    entity_id={d['entity_id']}  {d['lat_lon']}  {d['morada']}")
        if len(shown) >= 30:
            print("\n  ... (mais resultados no CSV, se pedido com --csv)")
            break

    if args.csv:
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[find_duplicate_poi_names] Relatorio completo em {out_path} ({len(rows)} linhas)")


if __name__ == "__main__":
    main()
