# src/transit/transit_service.py
import math
import pickle
import requests
import numpy as np
import networkx as nx
from datetime import date
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from .gtfs_loader import GTFSLoader, _haversine_minutes
from .calendar_resolver import get_active_services

# Operadores e respectivos prefixos e pastas
OPERATORS = {
    "metro_lisboa":          ("ML", "data/gtfs/metro_lisboa"),
    "metro_porto":           ("MP", "data/gtfs/metro_porto"),
    "stcp":                  ("STCP", "data/gtfs/stcp"),
    "carris_metropolitana":  ("CM", "data/gtfs/carris_metropolitana"),
    "cp":                    ("CP", "data/gtfs/cp"),
}

# Distância máxima a pé para considerar transbordo entre operadores (metros)
TRANSFER_WALK_METERS = 300
TRANSFER_WALK_SPEED_KMH = 4.5

CACHE_PATH = Path("data/gtfs/unified_graph.pkl")


class TransitService:
    """
    Interface pública para routing de transportes públicos.
    Carregada uma vez no arranque do FastAPI e mantida em memória.
    """

    def __init__(self):
        self.graph: Optional[nx.DiGraph] = None
        self._stops: Dict[str, dict] = {}  # node_id → {lat, lon, name, operator}

    # ──────────────────────────────────────────
    # Inicialização
    # ──────────────────────────────────────────

    def load(self, query_date: Optional[date] = None, use_cache: bool = True):
        """
        Carrega o grafo unificado.
        Se existir cache e use_cache=True, usa o pickle.
        Caso contrário, reconstrói a partir dos GTFS.
        """
        if use_cache and CACHE_PATH.exists():
            with open(CACHE_PATH, "rb") as f:
                data = pickle.load(f)
            self.graph = data["graph"]
            self._stops = data["stops"]
            print(f"[TransitService] Grafo carregado do cache: "
                  f"{self.graph.number_of_nodes()} nós, "
                  f"{self.graph.number_of_edges()} arestas")
            return

        if query_date is None:
            query_date = date.today()

        self.graph = nx.DiGraph()
        self._stops = {}

        # Carregar cada operador
        for operator, (prefix, gtfs_path) in OPERATORS.items():
            path = Path(gtfs_path)
            if not path.exists():
                print(f"[TransitService] GTFS não encontrado: {gtfs_path}, a saltar.")
                continue
            print(f"[TransitService] A carregar {operator}...")
            loader = GTFSLoader(operator, path, prefix)
            sub_graph = loader.build(query_date)
            # Merge no grafo unificado
            self.graph = nx.compose(self.graph, sub_graph)
            self._stops.update({
                nid: self.graph.nodes[nid]
                for nid in sub_graph.nodes
            })

        # Adicionar arestas de transbordo a pé entre operadores próximos
        self._add_transfer_edges()

        # Guardar cache
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"graph": self.graph, "stops": self._stops}, f)

        print(f"[TransitService] Grafo unificado: "
              f"{self.graph.number_of_nodes()} nós, "
              f"{self.graph.number_of_edges()} arestas")

    def _add_transfer_edges(self):
        """
        Liga paragens de operadores diferentes que estão a <300m a pé.
        Exemplo: Metro Lisboa Cais do Sodré ↔ CP Cais do Sodré
        """
        nodes = list(self.graph.nodes(data=True))
        threshold_km = TRANSFER_WALK_METERS / 1000

        for i, (nid_a, data_a) in enumerate(nodes):
            for nid_b, data_b in nodes[i+1:]:
                if data_a.get("operator") == data_b.get("operator"):
                    continue  # mesmo operador — transbordos já no GTFS
                d = _haversine_km(
                    data_a["lat"], data_a["lon"],
                    data_b["lat"], data_b["lon"]
                )
                if d <= threshold_km:
                    walk_min = (d / TRANSFER_WALK_SPEED_KMH) * 60
                    # Bidireccional
                    self.graph.add_edge(nid_a, nid_b,
                                        weight=walk_min,
                                        route_id="WALK",
                                        operator="walk")
                    self.graph.add_edge(nid_b, nid_a,
                                        weight=walk_min,
                                        route_id="WALK",
                                        operator="walk")

    # ──────────────────────────────────────────
    # Interface pública
    # ──────────────────────────────────────────

    def nearest_stop(self, lat: float, lon: float,
                     operator: Optional[str] = None,
                     max_dist_km: float = 1.0) -> Optional[str]:
        """
        Devolve o node_id da paragem mais próxima de (lat, lon).
        Se operator especificado, filtra por operador.
        """
        best_node, best_dist = None, float("inf")
        for nid, data in self.graph.nodes(data=True):
            if operator and data.get("operator") != operator:
                continue
            d = _haversine_km(lat, lon, data["lat"], data["lon"])
            if d < best_dist:
                best_dist = d
                best_node = nid
        if best_dist > max_dist_km:
            return None
        return best_node

    def get_route(self, origin_coords: Tuple[float, float],
                  dest_coords: Tuple[float, float]) -> Optional[Dict]:
        """
        Devolve o itinerário mais rápido de transportes públicos
        entre dois pontos geográficos.
        """
        if self.graph is None:
            return None

        stop_a = self.nearest_stop(*origin_coords)
        stop_b = self.nearest_stop(*dest_coords)

        if not stop_a or not stop_b or stop_a == stop_b:
            return None

        try:
            path = nx.dijkstra_path(self.graph, stop_a, stop_b, weight="weight")
            total_min = nx.dijkstra_path_length(
                self.graph, stop_a, stop_b, weight="weight"
            )
        except nx.NetworkXNoPath:
            return None

        # Construir itinerário legível
        segments = []
        for i in range(len(path) - 1):
            edge_data = self.graph[path[i]][path[i+1]]
            segments.append({
                "from": self.graph.nodes[path[i]].get("name", path[i]),
                "to": self.graph.nodes[path[i+1]].get("name", path[i+1]),
                "route": edge_data.get("route_id", ""),
                "operator": edge_data.get("operator", ""),
                "minutes": round(edge_data["weight"], 1),
            })

        return {
            "origin_stop": self.graph.nodes[stop_a].get("name", stop_a),
            "dest_stop": self.graph.nodes[stop_b].get("name", stop_b),
            "total_minutes": round(total_min, 1),
            "segments": segments,
        }

    def build_cost_matrix(self, pois: List, mode: str = "public_transport") -> np.ndarray:
        k = len(pois)
        matrix = np.zeros((k, k))

        if mode == "fastest":
            # Para cada par, calcula todos os modos e usa o mais rápido
            tp_matrix   = self._build_tp_matrix(pois)
            car_matrix  = _osrm_matrix(pois, "car")
            foot_matrix = _osrm_matrix(pois, "foot")
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    matrix[i][j] = min(
                        tp_matrix[i][j],
                        car_matrix[i][j],
                        foot_matrix[i][j]
                    )
        elif mode == "public_transport":
            matrix = self._build_tp_matrix(pois)
        else:
            matrix = _osrm_matrix(pois, mode)

        return matrix

    def _build_tp_matrix(self, pois: List) -> np.ndarray:
        """Matriz K×K via Dijkstra no grafo GTFS."""
        k = len(pois)
        matrix = np.zeros((k, k))
        for i, poi_i in enumerate(pois):
            for j, poi_j in enumerate(pois):
                if i == j:
                    continue
                result = self.get_route(
                    (poi_i.lat, poi_i.lon),
                    (poi_j.lat, poi_j.lon)
                )
                if result:
                    matrix[i][j] = result["total_minutes"]
                else:
                    d = _haversine_km(poi_i.lat, poi_i.lon,
                                    poi_j.lat, poi_j.lon)
                    matrix[i][j] = (d / 20) * 60
        return matrix


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def _osrm_matrix(pois: List, mode: str) -> np.ndarray:
    """
    OSRM Table API — 1 chamada HTTP para a matriz completa.
    Fallback para Haversine se falhar.
    """
    osrm_profile = {"car": "driving", "foot": "walking",
                    "bicycle": "cycling"}.get(mode, "driving")
    coords_str = ";".join(f"{p.lon},{p.lat}" for p in pois)
    url = (f"http://router.project-osrm.org/table/v1/"
           f"{osrm_profile}/{coords_str}?annotations=duration")
    k = len(pois)
    try:
        resp = requests.get(url, timeout=8)
        data = resp.json()
        if data.get("code") == "Ok":
            durations = data["durations"]
            matrix = np.array(durations, dtype=float)
            matrix /= 60  # segundos → minutos
            return matrix
    except Exception as e:
        print(f"[TransitService] OSRM falhou ({e}), a usar Haversine.")

    # Fallback Haversine
    speed = {"car": 50, "foot": 5, "bicycle": 15}.get(mode, 5)
    matrix = np.zeros((k, k))
    for i, pi in enumerate(pois):
        for j, pj in enumerate(pois):
            if i != j:
                d = _haversine_km(pi.lat, pi.lon, pj.lat, pj.lon)
                matrix[i][j] = (d / speed) * 60
    return matrix