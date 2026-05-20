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

OPERATORS = {
    "metro_lisboa":          ("ML", "data/gtfs/metro_lisboa"),
    "metro_porto":           ("MP", "data/gtfs/metro_porto"),
    "stcp":                  ("STCP", "data/gtfs/stcp"),
    "carris_metropolitana":  ("CM", "data/gtfs/carris_metropolitana"),
    "cp":                    ("CP", "data/gtfs/cp"),
}

TRANSFER_WALK_METERS = 300
TRANSFER_WALK_SPEED_KMH = 4.5

CACHE_PATH = Path("data/gtfs/unified_graph.pkl")


class TransitService:
    """
    Interface publica para routing de transportes publicos.
    O grafo contem TODOS os trips de todos os operadores.
    Ao fazer routing, filtra as arestas pelos servicos activos na data pedida.
    """

    def __init__(self):
        self.graph: Optional[nx.MultiDiGraph] = None
        self._stops: Dict[str, dict] = {}

    # ------------------------------------------
    # Inicializacao
    # ------------------------------------------

    def load(self, use_cache: bool = True):
        if use_cache and CACHE_PATH.exists():
            with open(CACHE_PATH, "rb") as f:
                data = pickle.load(f)
            self.graph = data["graph"]
            self._stops = data["stops"]
            print(f"[TransitService] Grafo carregado do cache: "
                  f"{self.graph.number_of_nodes()} nos, "
                  f"{self.graph.number_of_edges()} arestas")
            return

        self.graph = nx.MultiDiGraph()
        self._stops = {}

        for operator, (prefix, gtfs_path) in OPERATORS.items():
            path = Path(gtfs_path)
            if not path.exists():
                print(f"[TransitService] GTFS nao encontrado: {gtfs_path}, a saltar.")
                continue
            print(f"[TransitService] A carregar {operator}...")
            loader = GTFSLoader(operator, path, prefix)
            sub_graph = loader.build()
            self.graph = nx.compose(self.graph, sub_graph)
            self._stops.update({
                nid: dict(self.graph.nodes[nid])
                for nid in sub_graph.nodes
            })

        self._add_transfer_edges()

        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print(f"[TransitService] Grafo unificado: {n_nodes} nos, {n_edges} arestas")

        if n_nodes == 0:
            print("[TransitService] Grafo vazio - cache nao guardada.")
            return

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"graph": self.graph, "stops": self._stops}, f)

    def _add_transfer_edges(self):
        """Liga paragens de operadores diferentes que estao a <300m a pe."""
        nodes = list(self.graph.nodes(data=True))
        threshold_km = TRANSFER_WALK_METERS / 1000

        for i, (nid_a, data_a) in enumerate(nodes):
            for nid_b, data_b in nodes[i+1:]:
                if data_a.get("operator") == data_b.get("operator"):
                    continue
                d = _haversine_km(
                    data_a["lat"], data_a["lon"],
                    data_b["lat"], data_b["lon"]
                )
                if d <= threshold_km:
                    walk_min = (d / TRANSFER_WALK_SPEED_KMH) * 60
                    self.graph.add_edge(nid_a, nid_b,
                                        weight=walk_min,
                                        route_id="WALK",
                                        service_id="WALK",
                                        operator="walk",
                                        departure_min=0)
                    self.graph.add_edge(nid_b, nid_a,
                                        weight=walk_min,
                                        route_id="WALK",
                                        service_id="WALK",
                                        operator="walk",
                                        departure_min=0)

    # ------------------------------------------
    # Interface publica
    # ------------------------------------------

    def _active_subgraph(self, query_date: Optional[date]) -> nx.DiGraph:
        """
        Devolve um DiGraph simples com apenas as arestas activas na data pedida.
        Se query_date=None, usa todas as arestas (fallback para planeamento
        sem data especifica).
        """
        if query_date is None:
            # Sem data: usar a aresta mais rapida por par de paragens
            simple = nx.DiGraph()
            simple.add_nodes_from(self.graph.nodes(data=True))
            for u, v, data in self.graph.edges(data=True):
                if not simple.has_edge(u, v) or simple[u][v]["weight"] > data["weight"]:
                    simple.add_edge(u, v, **data)
            return simple

        # Determinar servicos activos por operador
        active_services: set = {"WALK"}
        for operator, (_, gtfs_path) in OPERATORS.items():
            path = Path(gtfs_path)
            if not path.exists():
                continue
            try:
                active_services |= get_active_services(operator, path, query_date)
            except Exception as _e:
                print(f"[TransitService] Aviso ao carregar servicos de {operator}: {_e}")

        # Construir subgrafo simples com apenas essas arestas
        simple = nx.DiGraph()
        simple.add_nodes_from(self.graph.nodes(data=True))
        for u, v, data in self.graph.edges(data=True):
            if data.get("service_id") not in active_services:
                continue
            if not simple.has_edge(u, v) or simple[u][v]["weight"] > data["weight"]:
                simple.add_edge(u, v, **data)
        return simple

    def nearest_stop(self, lat: float, lon: float,
                     operator: Optional[str] = None,
                     max_dist_km: float = 1.0) -> Optional[str]:
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
                  dest_coords: Tuple[float, float],
                  query_date: Optional[date] = None) -> Optional[Dict]:
        if self.graph is None:
            return None

        stop_a = self.nearest_stop(*origin_coords)
        stop_b = self.nearest_stop(*dest_coords)

        if not stop_a or not stop_b or stop_a == stop_b:
            return None

        subgraph = self._active_subgraph(query_date)

        try:
            path = nx.dijkstra_path(subgraph, stop_a, stop_b, weight="weight")
            total_min = nx.dijkstra_path_length(subgraph, stop_a, stop_b, weight="weight")
        except nx.NetworkXNoPath:
            return None

        segments = []
        for i in range(len(path) - 1):
            try:
                edge_data = subgraph[path[i]][path[i+1]]
                # MultiDiGraph devolve dict de dicts — pegar o primeiro edge
                if isinstance(edge_data, dict) and edge_data and not isinstance(next(iter(edge_data.values())), (int, float, str)):
                    edge_data = next(iter(edge_data.values()))
                segments.append({
                    "from": subgraph.nodes[path[i]].get("name", path[i]),
                    "to": subgraph.nodes[path[i+1]].get("name", path[i+1]),
                    "route": edge_data.get("route_id", ""),
                    "operator": edge_data.get("operator", ""),
                    "minutes": round(edge_data.get("weight", 0), 1),
                })
            except (KeyError, StopIteration):
                continue

        return {
            "origin_stop": subgraph.nodes[stop_a].get("name", stop_a),
            "dest_stop": subgraph.nodes[stop_b].get("name", stop_b),
            "total_minutes": round(total_min, 1),
            "segments": segments,
        }

    def build_cost_matrix(self, pois: List, mode: str = "public_transport",
                          query_date: Optional[date] = None) -> np.ndarray:
        k = len(pois)
        matrix = np.zeros((k, k))

        if mode == "fastest":
            tp_matrix   = self._build_tp_matrix(pois, query_date)
            car_matrix  = _osrm_matrix(pois, "car")
            foot_matrix = _osrm_matrix(pois, "foot")
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    matrix[i][j] = min(tp_matrix[i][j], car_matrix[i][j], foot_matrix[i][j])
        elif mode == "public_transport":
            matrix = self._build_tp_matrix(pois, query_date)
        else:
            matrix = _osrm_matrix(pois, mode)

        return matrix

    def get_route_geometry(self,
                           origin_coords: Tuple[float, float],
                           dest_coords: Tuple[float, float],
                           query_date: Optional[date] = None) -> Optional[List]:
        """
        Devolve lista [[lat,lon],...] das paragens ao longo da rota de TP,
        ou None se não existir rota ou cobertura GTFS.
        Usado pelo mapa para visualizar paragens reais.
        """
        if self.graph is None:
            return None
        stop_a = self.nearest_stop(*origin_coords, max_dist_km=2.0)
        stop_b = self.nearest_stop(*dest_coords, max_dist_km=2.0)
        if not stop_a or not stop_b or stop_a == stop_b:
            return None
        subgraph = self._active_subgraph(query_date)
        try:
            path = nx.dijkstra_path(subgraph, stop_a, stop_b, weight="weight")
        except Exception:
            return None
        geometry = []
        for nid in path:
            d = subgraph.nodes[nid]
            if d.get("lat") and d.get("lon"):
                geometry.append([d["lat"], d["lon"]])
        return geometry if len(geometry) >= 2 else None

    def _build_tp_matrix(self, pois: List, query_date: Optional[date] = None) -> np.ndarray:
        k = len(pois)
        matrix = np.zeros((k, k))
        for i, poi_i in enumerate(pois):
            for j, poi_j in enumerate(pois):
                if i == j:
                    continue
                result = self.get_route(
                    (poi_i.lat, poi_i.lon),
                    (poi_j.lat, poi_j.lon),
                    query_date=query_date,
                )
                if result:
                    matrix[i][j] = result["total_minutes"]
                else:
                    d = _haversine_km(poi_i.lat, poi_i.lon, poi_j.lat, poi_j.lon)
                    matrix[i][j] = (d / 20) * 60
        return matrix


# ------------------------------------------
# Helpers
# ------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def _osrm_matrix(pois: List, mode: str) -> np.ndarray:
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
            matrix /= 60
            return matrix
    except Exception as e:
        print(f"[TransitService] OSRM falhou ({e}), a usar Haversine.")

    speed = {"car": 50, "foot": 5, "bicycle": 15}.get(mode, 5)
    matrix = np.zeros((k, k))
    for i, pi in enumerate(pois):
        for j, pj in enumerate(pois):
            if i != j:
                d = _haversine_km(pi.lat, pi.lon, pj.lat, pj.lon)
                matrix[i][j] = (d / speed) * 60
    return matrix
