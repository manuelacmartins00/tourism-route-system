# src/transit/gtfs_loader.py
import csv
import math
import networkx as nx
from datetime import date
from pathlib import Path
from typing import Dict, Optional
from .calendar_resolver import get_active_services


def _haversine_minutes(lat1, lon1, lat2, lon2, speed_kmh=4.5) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    km = R * 2 * math.asin(math.sqrt(a))
    return (km / speed_kmh) * 60


class GTFSLoader:
    """
    Carrega um feed GTFS e constrói um MultiDiGraph NetworkX.

    Nós:  stop_id prefixado (ex: "ML_BC", "CP_94_2006")
    Atributos do nó: lat, lon, name, operator, zone_id

    Arestas: stop_i → stop_j  (uma por trip — preserva service_id e departure_time)
    Atributos da aresta: weight (minutos), route_id, service_id, operator, departure_min
    """

    def __init__(self, operator: str, gtfs_dir: Path, prefix: str):
        self.operator = operator
        self.gtfs_dir = Path(gtfs_dir)
        self.prefix = prefix
        self.stops: Dict[str, dict] = {}
        self.graph = nx.MultiDiGraph()

    def _pid(self, stop_id: str) -> str:
        return f"{self.prefix}_{stop_id}"

    def _load_stops(self):
        for row in self._read("stops.txt"):
            sid = row.get("stop_id", "").strip()
            if not sid or sid == ".":
                continue
            lt = row.get("location_type", "0").strip()
            if lt == "1":
                continue
            try:
                lat = float(row["stop_lat"])
                lon = float(row["stop_lon"])
            except (ValueError, KeyError):
                continue
            self.stops[sid] = {
                "lat": lat, "lon": lon,
                "name": row.get("stop_name", "").strip(),
                "zone_id": row.get("zone_id", ""),
                "operator": self.operator,
            }
            self.graph.add_node(
                self._pid(sid),
                lat=lat, lon=lon,
                name=row.get("stop_name", "").strip(),
                zone_id=row.get("zone_id", ""),
                operator=self.operator,
            )

    def _load_edges(self):
        """
        Carrega TODOS os trips do feed.
        Cada aresta guarda service_id para que o routing possa filtrar
        por data em tempo de consulta via calendar_resolver.
        Por par de paragens e service_id guarda apenas a aresta mais rápida
        para manter o grafo compacto.
        """
        trip_to_service: Dict[str, str] = {}
        trip_to_route: Dict[str, str] = {}
        for row in self._read("trips.txt"):
            tid = row["trip_id"]
            trip_to_service[tid] = row.get("service_id", "")
            trip_to_route[tid] = row.get("route_id", "")

        # Agrupar stop_times por trip
        trip_stops: Dict[str, list] = {}
        for row in self._read("stop_times.txt"):
            tid = row.get("trip_id", "")
            if tid not in trip_to_service:
                continue
            sid = row.get("stop_id", "").strip()
            if sid not in self.stops:
                continue
            try:
                seq = int(row["stop_sequence"])
            except (ValueError, KeyError):
                continue
            dep = row.get("departure_time", row.get("arrival_time", ""))
            trip_stops.setdefault(tid, []).append((seq, sid, dep))

        # Uma aresta por (pa, pb, service_id) — guarda a mais rápida
        best: Dict[tuple, float] = {}

        for tid, stops_seq in trip_stops.items():
            stops_seq.sort(key=lambda x: x[0])
            service_id = trip_to_service[tid]
            route_id = trip_to_route[tid]
            for i in range(len(stops_seq) - 1):
                _, sid_a, dep_a = stops_seq[i]
                _, sid_b, dep_b = stops_seq[i + 1]
                weight = self._time_diff_minutes(dep_a, dep_b)
                if weight <= 0:
                    sa, sb = self.stops[sid_a], self.stops[sid_b]
                    weight = _haversine_minutes(sa["lat"], sa["lon"],
                                               sb["lat"], sb["lon"])
                pa, pb = self._pid(sid_a), self._pid(sid_b)
                key = (pa, pb, service_id)
                if key in best and best[key] <= weight:
                    continue
                best[key] = weight
                dep_min = self._to_minutes(dep_a)
                self.graph.add_edge(
                    pa, pb,
                    weight=weight,
                    route_id=route_id,
                    service_id=service_id,
                    operator=self.operator,
                    departure_min=dep_min,
                )

    @staticmethod
    def _time_diff_minutes(t1: str, t2: str) -> float:
        def to_m(t):
            parts = t.strip().split(":")
            if len(parts) < 2:
                return 0
            return int(parts[0]) * 60 + int(parts[1])
        try:
            diff = to_m(t2) - to_m(t1)
            return diff if diff > 0 else 0
        except Exception:
            return 0

    @staticmethod
    def _to_minutes(t: str) -> int:
        try:
            parts = t.strip().split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            return 0

    def _read(self, filename: str):
        path = self.gtfs_dir / filename
        if not path.exists():
            return []
        with open(path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def build(self) -> nx.MultiDiGraph:
        self._load_stops()
        self._load_edges()
        return self.graph
