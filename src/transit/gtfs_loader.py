# src/transit/gtfs_loader.py
import csv
import math
import pickle
import networkx as nx
from datetime import date
from pathlib import Path
from typing import Dict, Tuple, Optional
from .calendar_resolver import get_active_services


def _haversine_minutes(lat1, lon1, lat2, lon2, speed_kmh=4.5) -> float:
    """Distância a pé entre duas paragens em minutos."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    km = R * 2 * math.asin(math.sqrt(a))
    return (km / speed_kmh) * 60


class GTFSLoader:
    """
    Carrega um feed GTFS e constrói um DiGraph NetworkX.
    
    Nós:  stop_id prefixado (ex: "ML_BC", "CP_94_2006")
    Atributos do nó: lat, lon, name, operator, zone_id
    
    Arestas: stop_i → stop_j
    Atributos da aresta: weight (minutos), route_id, operator
    """

    def __init__(self, operator: str, gtfs_dir: Path, prefix: str):
        self.operator = operator
        self.gtfs_dir = Path(gtfs_dir)
        self.prefix = prefix          # ex: "ML", "CP", "MP", "STCP", "CM"
        self.stops: Dict[str, dict] = {}   # stop_id → {lat, lon, name, zone_id}
        self.graph = nx.DiGraph()

    def _pid(self, stop_id: str) -> str:
        """Adiciona prefixo ao stop_id para evitar colisões entre operadores."""
        return f"{self.prefix}_{stop_id}"

    def _load_stops(self):
        for row in self._read("stops.txt"):
            sid = row.get("stop_id", "").strip()
            if not sid or sid == ".":
                continue
            # Metro Lisboa tem parent_station — usar só stops filho (location_type=0 ou vazio)
            lt = row.get("location_type", "0").strip()
            if lt == "1":   # estação pai — não é uma paragem real
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

    def _load_edges(self, query_date: date):
        """
        Para cada trip activo hoje, adiciona arestas stop_i → stop_i+1
        com peso = tempo de viagem em minutos.
        """
        active_services = get_active_services(
            self.operator, self.gtfs_dir, query_date
        )

        # trip_id → route_id (para os metadados da aresta)
        trip_to_route: Dict[str, str] = {}
        active_trips: set = set()
        for row in self._read("trips.txt"):
            if row.get("service_id", "") in active_services:
                tid = row["trip_id"]
                active_trips.add(tid)
                trip_to_route[tid] = row.get("route_id", "")

        # Agrupar stop_times por trip
        trip_stops: Dict[str, list] = {}
        for row in self._read("stop_times.txt"):
            tid = row.get("trip_id", "")
            if tid not in active_trips:
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

        # Adicionar arestas
        for tid, stops_seq in trip_stops.items():
            stops_seq.sort(key=lambda x: x[0])
            route_id = trip_to_route.get(tid, "")
            for i in range(len(stops_seq) - 1):
                _, sid_a, dep_a = stops_seq[i]
                _, sid_b, dep_b = stops_seq[i + 1]
                weight = self._time_diff_minutes(dep_a, dep_b)
                if weight <= 0:
                    # Fallback: Haversine a pé entre paragens consecutivas
                    sa, sb = self.stops[sid_a], self.stops[sid_b]
                    weight = _haversine_minutes(sa["lat"], sa["lon"],
                                                sb["lat"], sb["lon"])
                pa, pb = self._pid(sid_a), self._pid(sid_b)
                # Guardar a aresta mais rápida se já existir
                if self.graph.has_edge(pa, pb):
                    if self.graph[pa][pb]["weight"] <= weight:
                        continue
                self.graph.add_edge(pa, pb,
                                    weight=weight,
                                    route_id=route_id,
                                    operator=self.operator)

    @staticmethod
    def _time_diff_minutes(t1: str, t2: str) -> float:
        """Diferença entre dois tempos HH:MM:SS em minutos. Suporta >24h (CP)."""
        def to_minutes(t):
            parts = t.strip().split(":")
            if len(parts) < 2:
                return 0
            return int(parts[0]) * 60 + int(parts[1])
        try:
            diff = to_minutes(t2) - to_minutes(t1)
            return diff if diff > 0 else 0
        except Exception:
            return 0

    def _read(self, filename: str):
        path = self.gtfs_dir / filename
        if not path.exists():
            return []
        with open(path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def build(self, query_date: date) -> nx.DiGraph:
        self._load_stops()
        self._load_edges(query_date)
        return self.graph