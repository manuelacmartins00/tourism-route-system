# src/utils/day_planner.py

from typing import List, Dict, Optional
import numpy as np
import math


class DayPlanner:
    NOCTURNO_CATEGORIES = {"bares_e_discotecas", "casinos"}
    DIURNO_CATEGORIES = {"monumentos", "museus_e_palacios", "espacos_verdes",
                          "parques_e_reservas", "arqueologia", "grutas",
                          "turismo_activo", "praias", "zoos_e_aquarios"}
    NOCTURNAL_START = "21:00"

    def __init__(self, hours_per_day: int = 8, start_time: str = "09:00", lunch_break: int = 60):
        self.start_lat = None
        self.start_lon = None
        self.hours_per_day = hours_per_day
        self.minutes_per_day = hours_per_day * 60
        self.start_time = start_time
        self.lunch_break = lunch_break

    # ── Public API ──────────────────────────────────────────────────────────

    def plan_days(self, route: List[Dict], distance_matrix: np.ndarray = None,
                  total_days: int = None) -> Dict:
        if not route:
            return {"days": [], "total_days": 0}

        if total_days is None:
            total_time = sum(p['duration'] for p in route)
            total_days = max(1, math.ceil(total_time / self.minutes_per_day))

        diurnal  = [p for p in route if p.get("category") not in self.NOCTURNO_CATEGORIES]
        nocturnal = [p for p in route if p.get("category") in self.NOCTURNO_CATEGORIES]

        print(f"\n📅 Planejando {len(route)} POIs em {total_days} dias "
              f"({len(diurnal)} diurnos, {len(nocturnal)} noturnos)...")
        print(f"   Tempo por dia: {self.minutes_per_day} min ({self.hours_per_day}h)\n")

        # Distribuir POIs diurnos por dia (clustering geográfico se possível)
        diurnal_by_day = self._distribute_diurnal(diurnal, total_days)

        # Distribuir POIs noturnos em round-robin pelos dias
        nocturnal_by_day: List[List[Dict]] = [[] for _ in range(total_days)]
        for i, poi in enumerate(nocturnal):
            nocturnal_by_day[i % total_days].append(poi)

        days = []
        for day_num in range(1, total_days + 1):
            d = diurnal_by_day[day_num - 1] if day_num <= len(diurnal_by_day) else []
            n = nocturnal_by_day[day_num - 1]
            if d or n:
                days.append(self._format_day(day_num, d, n))

        return {
            "days": days,
            "total_days": len(days),
            "total_pois": len(route),
            "summary": self._generate_summary(days),
        }

    # ── Distribution ────────────────────────────────────────────────────────

    def _distribute_diurnal(self, diurnal: List[Dict], n_days: int) -> List[List[Dict]]:
        if not diurnal:
            return [[] for _ in range(n_days)]
        if len(diurnal) <= n_days:
            result = [[p] for p in diurnal]
            result += [[] for _ in range(n_days - len(diurnal))]
            return result
        try:
            from sklearn.cluster import KMeans
            coords = np.array([[p["lat"], p["lon"]] for p in diurnal])
            labels = KMeans(n_clusters=n_days, random_state=42, n_init=10).fit_predict(coords)
            by_day: List[List[Dict]] = [[] for _ in range(n_days)]
            for i, poi in enumerate(diurnal):
                by_day[labels[i]].append(poi)
            # Reequilibrar categorias antes de ordenar geograficamente
            by_day = self._rebalance_category_diversity(by_day, max_same_cat=2)
            # Ordem nearest-neighbour dentro de cada dia
            if self.start_lat:
                by_day = [self._nearest_neighbor_order(d, self.start_lat, self.start_lon)
                          if len(d) > 1 else d for d in by_day]
            return by_day
        except Exception:
            # Fallback: divisão sequencial
            by_day = [[] for _ in range(n_days)]
            for i, poi in enumerate(diurnal):
                by_day[i % n_days].append(poi)
            return by_day

    def _rebalance_category_diversity(self, by_day: List[List[Dict]], max_same_cat: int = 2) -> List[List[Dict]]:
        """
        Move POIs excedentes da mesma categoria para dias com menos dessa categoria,
        respeitando o budget de tempo por dia. Máximo max_same_cat POIs da mesma
        categoria por dia.
        """
        from collections import defaultdict
        n_days = len(by_day)
        if n_days <= 1:
            return by_day

        for _ in range(n_days * 3):  # iterações suficientes para convergir
            moved = False
            for src in range(n_days):
                cat_counts = defaultdict(list)
                for poi in by_day[src]:
                    cat_counts[poi['category']].append(poi)

                for cat, pois_in_cat in cat_counts.items():
                    if len(pois_in_cat) <= max_same_cat:
                        continue
                    # Mover os excedentes (os últimos da lista — menos prioritários)
                    for poi in pois_in_cat[max_same_cat:]:
                        # Encontrar o dia destino com menos desta categoria e com espaço
                        best_dst, best_count = None, float('inf')
                        for dst in range(n_days):
                            if dst == src:
                                continue
                            dst_cat_count = sum(1 for p in by_day[dst] if p['category'] == cat)
                            dst_time = sum(p['duration'] for p in by_day[dst])
                            # Só mover se o dia destino tem espaço e menos desta categoria
                            if dst_cat_count < best_count and dst_time + poi['duration'] <= self.minutes_per_day:
                                best_count = dst_cat_count
                                best_dst = dst
                        if best_dst is not None and best_count < len(pois_in_cat) - max_same_cat:
                            by_day[src].remove(poi)
                            by_day[best_dst].append(poi)
                            moved = True
                            break  # recalcular cat_counts após cada movimento
                    if moved:
                        break
            if not moved:
                break

        return by_day

    def _nearest_neighbor_order(self, pois: List[Dict], start_lat: float, start_lon: float) -> List[Dict]:
        remaining = list(pois)
        ordered = []
        cur_lat, cur_lon = start_lat, start_lon
        while remaining:
            nearest = min(remaining, key=lambda p: self._haversine(cur_lat, cur_lon, p['lat'], p['lon']))
            ordered.append(nearest)
            cur_lat, cur_lon = nearest['lat'], nearest['lon']
            remaining.remove(nearest)
        return ordered

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        R = 6371
        r = math.radians
        a = math.sin(r(lat2-lat1)/2)**2 + math.cos(r(lat1))*math.cos(r(lat2))*math.sin(r(lon2-lon1)/2)**2
        return R * 2 * math.asin(math.sqrt(a))

    # ── Formatting ──────────────────────────────────────────────────────────

    def _format_day(self, day_num: int, diurnal: List[Dict], nocturnal: List[Dict]) -> Dict:
        schedule = []
        order = 1

        # Manhã/tarde — começa em start_time (09:00)
        current = self._parse_time(self.start_time)
        for i, poi in enumerate(diurnal):
            arr = self._fmt(current)
            dep = self._fmt(current + poi['duration'])
            schedule.append({**poi, "arrival_time": arr, "departure_time": dep, "order": order})
            order += 1
            current += poi['duration']
            # pausa de almoço
            if i < len(diurnal) - 1 and 12 * 60 < current < 14 * 60:
                current += self.lunch_break

        # Noite — começa às 21:00
        current = self._parse_time(self.NOCTURNAL_START)
        for poi in nocturnal:
            arr = self._fmt(current)
            dep = self._fmt(current + poi['duration'])
            schedule.append({**poi, "arrival_time": arr, "departure_time": dep, "order": order})
            order += 1
            current += poi['duration']

        # end_time: última saída (noturna se existir, senão diurna)
        if schedule:
            end_time = schedule[-1]['departure_time']
        else:
            end_time = self.start_time

        total_cost = sum(p['cost'] for p in diurnal + nocturnal)
        total_time = sum(p['duration'] for p in diurnal + nocturnal)

        return {
            "day": day_num,
            "pois": schedule,
            "total_time": total_time,
            "total_cost": total_cost,
            "n_pois": len(schedule),
            "start_time": self.start_time,
            "end_time": end_time,
        }

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _parse_time(self, time_str: str) -> int:
        h, m = map(int, time_str.split(':'))
        return h * 60 + m

    def _fmt(self, minutes: int) -> str:
        h = int(minutes // 60) % 24
        m = int(minutes % 60)
        return f"{h:02d}:{m:02d}"

    def _generate_summary(self, days: List[Dict]) -> str:
        lines = []
        for d in days:
            lines.append(f"Dia {d['day']}: {d['n_pois']} POIs, {d['total_time']} min "
                         f"({d['total_time']/60:.1f}h), €{d['total_cost']:.2f}")
        return "\n".join(lines)

    def print_itinerary(self, day_plan: Dict):
        print(f"\n{'='*70}")
        print(f"📅 ITINERÁRIO - {day_plan['total_days']} DIAS")
        print(f"{'='*70}\n")
        for day in day_plan['days']:
            print(f"📆 DIA {day['day']} - {day['start_time']} às {day['end_time']}")
            print(f"   {day['n_pois']} POIs | {day['total_time']} min | €{day['total_cost']:.2f}\n")
            for poi in day['pois']:
                prefix = "🌙" if poi.get("category") in self.NOCTURNO_CATEGORIES else "  "
                print(f"   {prefix} {poi['order']}. {poi['arrival_time']} - {poi['departure_time']}")
                print(f"         {poi['name']} ({poi['category']})")
                print(f"         Duração: {poi['duration']} min | Custo: €{poi['cost']:.2f}\n")
        print(f"{'='*70}\n")
        print(day_plan['summary'])
        print(f"\n{'='*70}\n")
