"""
Verifica que todos os centroides calculados a partir do GeoJSON estao
dentro dos limites geograficos esperados de Portugal e proximos do centro
do bounding box de cada municipio.
"""
import json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.location_resolver import _polygon_centroid, _bbox_radius_km

GEOJSON = Path("data/Portugal_Municipalities.geojson")

# Bounds aceitaveis para territorio portugues (continente + ilhas)
VALID_BOUNDS = [
    # (lat_min, lat_max, lon_min, lon_max, descricao)
    (36.8,  42.3, -9.6,  -6.1, "Continente"),
    (32.4,  33.2, -17.4, -16.2, "Madeira"),
    (36.9,  40.0, -31.4, -24.9, "Acores"),
    (29.8,  30.2, -16.0, -15.8, "Selvagens"),   # ilhas remotas do Funchal
]

def in_portugal(lat, lon):
    return any(
        lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
        for lat_min, lat_max, lon_min, lon_max, _ in VALID_BOUNDS
    )

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

with open(GEOJSON, encoding="utf-8") as f:
    gj = json.load(f)

problems = []
ok = 0

for feat in gj["features"]:
    props = feat.get("properties", {})
    name = (props.get("Concelho") or props.get("MUNICIPIO") or
            props.get("NAME_2") or props.get("Municipio") or
            props.get("name") or props.get("NOME") or "?")
    geom = feat.get("geometry", {})
    coords = geom.get("coordinates", [])
    geom_type = geom.get("type", "")

    try:
        if geom_type == "Polygon":
            lat, lon = _polygon_centroid([coords])
        elif geom_type == "MultiPolygon":
            lat, lon = _polygon_centroid(coords)
        else:
            continue
    except Exception as e:
        problems.append(f"  ERRO centroide {name}: {e}")
        continue

    if lat is None:
        problems.append(f"  NULO: {name}")
        continue

    # 1. Centroide fora dos limites de Portugal?
    if not in_portugal(lat, lon):
        problems.append(f"  FORA DE PORTUGAL: {name} -> ({lat:.4f}, {lon:.4f})")
        continue

    # 2. Centroide muito longe do centro do bbox?
    all_pts = []
    def flatten(c):
        if c and isinstance(c[0], list):
            for x in c: flatten(x)
        else:
            all_pts.append(c)
    flatten(coords)
    if all_pts and isinstance(all_pts[0], (int, float)):
        # pontos simples
        pass
    else:
        lats_all = [p[1] for p in all_pts if isinstance(p, list) and len(p) >= 2]
        lons_all = [p[0] for p in all_pts if isinstance(p, list) and len(p) >= 2]
        if lats_all and lons_all:
            bbox_clat = (min(lats_all) + max(lats_all)) / 2
            bbox_clon = (min(lons_all) + max(lons_all)) / 2
            dist = haversine(lat, lon, bbox_clat, bbox_clon)
            if dist > 30:
                problems.append(
                    f"  CENTROIDE DESVIADO: {name} "
                    f"centroide=({lat:.4f},{lon:.4f}) "
                    f"bbox_centro=({bbox_clat:.4f},{bbox_clon:.4f}) "
                    f"distancia={dist:.1f}km"
                )
                continue

    ok += 1

print(f"Verificados: {ok + len(problems)} municipios")
print(f"OK: {ok}")
print(f"Problemas: {len(problems)}")
if problems:
    print("\nProblemas encontrados:")
    for p in problems:
        print(p)
else:
    print("\nTodos os centroides estao corretos!")
