import sys
sys.path.insert(0, ".")
from src.rag.rag_setup import POI_RAG
from collections import Counter

rag = POI_RAG()

cats_F = ["restaurantes_e_cafes", "eventos", "mercados"]
NEVER_INCLUDE = ["eventos", "postos_de_turismo", "agencias_de_viagem",
                 "localidade", "servicos_de_turismo", "outros"]

print("=== CAUSA 1: include_meals=False (rows 1473, 1580, 1601) ===")
# preferred_categories becomes ["eventos", "mercados"] after restaurantes removed
cats_no_meals = ["eventos", "mercados"]
r = rag.query("gastronomia", n_results=50, category_filter=cats_no_meals,
              category_exclude=NEVER_INCLUDE)
print(f"  categoria_filter={cats_no_meals}, excluindo eventos: {len(r['pois'])} POIs")
print("  -> eventos esta em NEVER_INCLUDE, mercados nao existe na BD => 0 POIs esperado")

print()
print("=== CAUSA 2: row 1471 (Funchal, include_meals=True) ===")
# Test combined $in + $nin filter (exactly as national fallback)
r2 = rag.query("gastronomia restaurante", n_results=25,
               category_filter=cats_F, category_exclude=NEVER_INCLUDE)
print(f"  Nacional, category_filter={cats_F}, excluindo eventos: {len(r2['pois'])} POIs")
cats2 = Counter(p["category"] for p in r2["pois"])
print(f"  Categorias: {dict(cats2)}")

# Check Funchal geo coordinates in LocationResolver
print()
print("=== CHECK: Coordenadas Funchal no GeoJSON ===")
import json
try:
    with open("data/municipios_portugal.geojson", encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj["features"]:
        nome = feat.get("properties", {}).get("Municipio", "") or feat.get("properties", {}).get("NAME_2", "")
        if "unchal" in str(nome):
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [])
            print(f"  {nome}: tipo={geom.get('type')}, primeiras coords={str(coords)[:200]}")
except Exception as e:
    print(f"  Erro: {e}")
