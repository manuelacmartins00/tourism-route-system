# src/utils/location_resolver.py

import json
import math
import time
import unicodedata
import re
from pathlib import Path
from typing import Optional, Tuple
import urllib.request
import urllib.parse


def _normalize(text: str) -> str:
    """Remove acentos e converte para lowercase para comparacao robusta."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _polygon_centroid(coordinates) -> Tuple[float, float]:
    """
    Calcula centroide de um poligono GeoJSON.
    Suporta Polygon e MultiPolygon.
    Retorna (lat, lon).
    """
    # Achata MultiPolygon para lista de aneis
    if isinstance(coordinates[0][0][0], list):
        # MultiPolygon: coordinates = [[anel1, anel2], [anel1], ...]
        rings = [ring for polygon in coordinates for ring in polygon]
    else:
        # Polygon: coordinates = [anel_exterior, anel_interior, ...]
        rings = coordinates

    # Usar apenas o anel exterior de cada poligono
    all_points = []
    for ring in rings:
        all_points.extend(ring[0] if isinstance(ring[0][0], list) else ring)

    if not all_points:
        return None, None

    lon = sum(p[0] for p in all_points) / len(all_points)
    lat = sum(p[1] for p in all_points) / len(all_points)
    return lat, lon


def _bbox_radius_km(bbox) -> float:
    """
    Calcula raio aproximado a partir de um bounding box [minLon, minLat, maxLon, maxLat].
    Usa metade da diagonal como raio.
    """
    if not bbox or len(bbox) < 4:
        return 30.0
    min_lon, min_lat, max_lon, max_lat = bbox
    # Haversine da diagonal
    R = 6371
    dlat = math.radians(max_lat - min_lat)
    dlon = math.radians(max_lon - min_lon)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(min_lat)) * math.cos(math.radians(max_lat)) *
         math.sin(dlon / 2) ** 2)
    diagonal_km = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return min(30.0, max(8.0, diagonal_km / 2))


class LocationResolver:
    """
    Resolve toponimos portugueses para coordenadas (lat, lon, radius_km).

    Estrategia:
      1. GeoJSON estatico dos municipios portugueses (primario, offline)
      2. Nominatim / OpenStreetMap (fallback, cobre regioes, serras, praias, etc.)
      3. None - sem filtro geografico se ambos falharem
    """

    GEOJSON_PATH = Path("data/Portugal_Municipalities.geojson")

    # Raio default por tipo de localizacao
    RADIUS_MUNICIPIO = 25.0   # km - municipio concreto
    RADIUS_NOMINATIM = 50.0   # km - fallback quando nao ha bbox

    # User-Agent obrigatorio pela politica do Nominatim
    NOMINATIM_UA = "TourismRouteSystem/1.0 (thesis project; contact: tourism@thesis.pt)"
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

    # Throttle: Nominatim exige max 1 req/s
    _last_nominatim_call: float = 0.0

    # Paises/regioes grandes: overrides estaticos para evitar que Nominatim
    # devolva o centro geográfico com raio <30km (ex: "Portugal" -> Santarém, r=30km)
    _COUNTRY_REGION_OVERRIDES: dict = {
        "portugal":         (39.60, -8.00, 350.0),
        "algarve":          (37.10, -8.10, 120.0),
        "alentejo":         (38.20, -7.80, 150.0),
        "alentejo litoral": (37.80, -8.50, 80.0),
        "costa vicentina":  (37.50, -8.90, 75.0),   # Sines -> Sagres, ~150km de costa SW
        "costa sudoeste":   (37.50, -8.90, 75.0),
        "ribatejo":         (39.30, -8.40, 80.0),
        "beiras":           (40.30, -7.70, 120.0),
        "centro":           (40.00, -8.00, 120.0),
        "norte":            (41.50, -8.00, 120.0),
        "minho":            (41.70, -8.30, 80.0),
        "tras-os-montes":   (41.80, -7.00, 100.0),
        "douro":            (41.10, -7.50, 80.0),
        "madeira":          (32.76, -16.96, 80.0),
        "ilha da madeira":  (32.76, -16.96, 80.0),
        "acores":           (38.66, -27.22, 200.0),
        "arquipelago dos acores": (38.66, -27.22, 200.0),
    }

    def __init__(self):
        self._geojson_index: dict = {}   # nome_normalizado -> {lat, lon, radius}
        self._loaded = False

    def _load_geojson(self):
        """Carrega e indexa o GeoJSON na primeira utilizacao."""
        if self._loaded:
            return

        if not self.GEOJSON_PATH.exists():
            print(f"   AVISO: [LocationResolver] GeoJSON nao encontrado: {self.GEOJSON_PATH}")
            self._loaded = True
            return

        with open(self.GEOJSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        for feature in data.get("features", []):
            props = feature.get("properties", {})
            geom  = feature.get("geometry", {})

            # Tentar varios campos de nome comuns em GeoJSON de municipios PT
            name = (props.get("Concelho") or props.get("MUNICIPIO") or
                    props.get("NAME_2") or props.get("NAME_1") or
                    props.get("Municipio") or props.get("name") or
                    props.get("NOME") or props.get("concelho") or "")

            if not name:
                continue

            coords = geom.get("coordinates", [])
            geom_type = geom.get("type", "")

            try:
                if geom_type == "Polygon":
                    lat, lon = _polygon_centroid([coords])
                elif geom_type == "MultiPolygon":
                    lat, lon = _polygon_centroid(coords)
                else:
                    continue
            except Exception:
                continue

            if lat is None:
                continue

            # Calcular raio a partir do bounding box se disponivel
            bbox = props.get("bbox") or feature.get("bbox")
            radius = _bbox_radius_km(bbox) if bbox else self.RADIUS_MUNICIPIO

            key = _normalize(name)
            self._geojson_index[key] = {
                "name": name,
                "lat": lat,
                "lon": lon,
                "radius_km": radius,
                "source": "geojson"
            }

        print(f"   [OK] [LocationResolver] {len(self._geojson_index)} municipios indexados do GeoJSON")
        self._loaded = True

    def _query_nominatim(self, location: str) -> Optional[dict]:
        """
        Chama Nominatim para resolver um toponimo.
        Respeita throttle de 1 req/s.
        Retorna dict com lat, lon, radius_km ou None.
        """
        # Throttle
        elapsed = time.time() - self._last_nominatim_call
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)

        params = urllib.parse.urlencode({
            "q": f"{location}, Portugal",
            "format": "json",
            "limit": 1,
            "countrycodes": "pt",
            "addressdetails": 0
        })

        url = f"{self.NOMINATIM_URL}?{params}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.NOMINATIM_UA})
            with urllib.request.urlopen(req, timeout=5) as resp:
                results = json.loads(resp.read().decode())

            LocationResolver._last_nominatim_call = time.time()

            if not results:
                return None

            r = results[0]
            lat = float(r["lat"])
            lon = float(r["lon"])

            # Calcular raio a partir do boundingbox Nominatim [minLat, maxLat, minLon, maxLon]
            bb = r.get("boundingbox")
            if bb and len(bb) == 4:
                bbox = [float(bb[2]), float(bb[0]), float(bb[3]), float(bb[1])]
                radius = _bbox_radius_km(bbox)
            else:
                radius = self.RADIUS_NOMINATIM

            return {
                "name": r.get("display_name", location),
                "lat": lat,
                "lon": lon,
                "radius_km": radius,
                "source": "nominatim"
            }

        except Exception as e:
            print(f"   AVISO: [LocationResolver] Nominatim falhou para '{location}': {e}")
            LocationResolver._last_nominatim_call = time.time()
            return None

    def resolve(self, location: str) -> Optional[Tuple[float, float, float]]:
        """
        Resolve um toponimo para (lat, lon, radius_km).

        Args:
            location: Nome do local ("Lisboa", "Algarve", "Serra da Estrela", etc.)

        Returns:
            (lat, lon, radius_km) ou None se nao conseguiu resolver.
        """
        if not location or not location.strip():
            return None

        self._load_geojson()

        key = _normalize(location)

        # 0. Override para paises/regioes grandes (tem prioridade sobre GeoJSON e Nominatim)
        if key in self._COUNTRY_REGION_OVERRIDES:
            lat, lon, radius = self._COUNTRY_REGION_OVERRIDES[key]
            print(f"   [OK] [LocationResolver] '{location}' -> override regional "
                  f"({lat:.4f}, {lon:.4f}, r={radius:.0f}km)")
            return lat, lon, radius

        # 1. Tentativa GeoJSON (exact match)
        if key in self._geojson_index:
            entry = self._geojson_index[key]
            print(f"   [OK] [LocationResolver] '{location}' -> GeoJSON "
                  f"({entry['lat']:.4f}, {entry['lon']:.4f}, r={entry['radius_km']:.0f}km)")
            return entry["lat"], entry["lon"], entry["radius_km"]

        # 2. Tentativa GeoJSON (partial match)
        #    Util para "Lisboa" encontrar "Lisboa (Lisboa)" etc.
        matches = [k for k in self._geojson_index if key in k or k in key]
        if len(matches) == 1:
            entry = self._geojson_index[matches[0]]
            print(f"   [OK] [LocationResolver] '{location}' -> GeoJSON partial "
                  f"({entry['lat']:.4f}, {entry['lon']:.4f}, r={entry['radius_km']:.0f}km)")
            return entry["lat"], entry["lon"], entry["radius_km"]

        # 3. Fallback Nominatim
        print(f"   [LocationResolver] '{location}' nao no GeoJSON, tentando Nominatim...")
        result = self._query_nominatim(location)

        if result:
            print(f"   [OK] [LocationResolver] '{location}' -> Nominatim "
                  f"({result['lat']:.4f}, {result['lon']:.4f}, r={result['radius_km']:.0f}km)")
            return result["lat"], result["lon"], result["radius_km"]

        print(f"   AVISO: [LocationResolver] Nao foi possivel resolver '{location}' - sem filtro geografico")
        return None