# src/utils/day_planner.py

from typing import List, Dict, Optional
import numpy as np
import math


class DayPlanner:
    NOCTURNO_CATEGORIES    = {"bares_e_discotecas", "casinos"}
    NOCTURNAL_START = "21:00"
    NOCTURNAL_END   = "03:00"   # hora de fim da vida noturna (dia seguinte)
    MAX_NOCTURNAL_PER_DAY = 3   # máximo de POIs noturnos por dia
    ACCOMMODATION_CATEGORIES = frozenset({
        "hotelaria", "alojamento_local", "turismo_habitacao",
        "turismo_espaco_rural", "apartamento_turistico",
        "pousadas_da_juventude", "aldeamento_turistico", "parques_de_campismo",
    })
    DIURNO_CATEGORIES = {"monumentos", "museus_e_palacios", "espacos_verdes",
                          "parques_e_reservas", "arqueologia", "grutas",
                          "turismo_activo", "praias", "zoos_e_aquarios"}
    # Categorias "imersivas": fica-se horas (praia, parque, trilho)
    # Sem restaurante → 1×6h (dia inteiro na mesma); com restaurante → 1 manhã + 1 tarde
    IMMERSIVE_CATEGORIES = frozenset({
        "praias", "parques_e_reservas", "espacos_verdes", "turismo_activo"
    })

    # Tabela de tempos de viagem por modo (min) — espelho da tabela em main_system.py
    _TRAVEL_TABLE = {
        "foot":             [(1, 12), (2, 25), (5, 60), (float('inf'), 999)],
        "car":              [(2, 4),  (5, 8),  (15, 15), (50, 38), (float('inf'), 75)],
        "public_transport": [(1, 10), (2, 17), (5, 30), (15, 47), (50, 94), (float('inf'), 153)],
        "bike":             [(2, 8),  (5, 18), (15, 50), (float('inf'), 120)],
        "fastest":          [(2, 4),  (5, 8),  (15, 15), (50, 38), (float('inf'), 75)],
    }

    def __init__(self, hours_per_day: int = 8, start_time: str = "09:00",
                 lunch_break: int = 60, transport_mode: str = "car",
                 transit_service=None, max_move_km: float = 80.0):
        self.start_lat = None
        self.start_lon = None
        self.hours_per_day = hours_per_day
        self.minutes_per_day = hours_per_day * 60
        self.start_time = start_time
        self.lunch_break = lunch_break
        self.transport_mode = transport_mode
        self.transit_service = transit_service
        self.max_move_km = max_move_km

    def _travel_minutes(self, d_km: float, mode: str = None) -> int:
        table = self._TRAVEL_TABLE.get(mode or self.transport_mode, self._TRAVEL_TABLE["car"])
        for max_km, t_min in table:
            if d_km <= max_km:
                return int(t_min)
        return 75

    def _segment_transport(self, d_km: float, lat1: float = None, lon1: float = None,
                           lat2: float = None, lon2: float = None) -> tuple:
        """Modo de transporte sugerido para este segmento (S2), com nota opcional (B5 v2).

        Mesmo que o modo global seja 'foot', percursos >2km nao sao
        razoaveis a pe — sugerir 'car' nesse caso.

        Para 'public_transport', tenta decompor o troco em 1a/ultima milha +
        perna de TP real (transit_service.get_transit_plan). Se nao houver
        cobertura GTFS entre paragens proximas, cai para 'car' com nota a
        avisar o utilizador que os dados de TP nao estavam disponiveis.
        """
        if (self.transport_mode == "public_transport" and self.transit_service is not None
                and lat1 is not None):
            plan = self.transit_service.get_transit_plan((lat1, lon1), (lat2, lon2))
            if plan:
                return "public_transport", plan["note"]
            return "car", ("Sem dados de transporte público para este troço — "
                            "pode existir autocarro local; estimativa baseada em carro/táxi.")
        if self.transport_mode == "foot" and d_km > 2.0:
            return "car", (f"Troço de {d_km:.1f} km — distância elevada para fazer a pé; "
                            "considera apanhar um táxi/Uber para este percurso.")
        return self.transport_mode, None

    # -- Public API ----------------------------------------------------------

    def plan_days(self, route: List[Dict], distance_matrix: np.ndarray = None,
                  total_days: int = None, first_day_start_time: str = None,
                  last_day_end_time: str = None, all_geos: List = None,
                  start_date: str = None, route_direction: str = None) -> Dict:
        if not route:
            return {"days": [], "total_days": 0}

        if total_days is None:
            total_time = sum(p['duration'] for p in route)
            total_days = max(1, math.ceil(total_time / self.minutes_per_day))

        # Quando o 1o dia começa à tarde/noite (>=18h), reduzir quota de POIs diurnos
        # para evitar museus e monumentos às 22-04h
        self._day1_max_diurnal = None
        if first_day_start_time:
            start_h = self._parse_time(first_day_start_time)
            if start_h >= self._parse_time("18:00"):
                available_min = self._parse_time("21:00") - start_h  # até às 21h
                self._day1_max_diurnal = max(1, available_min // 90)  # ~1 POI por 90min

        accommodation = [p for p in route if p.get("category") in self.ACCOMMODATION_CATEGORIES]
        non_accom     = [p for p in route if p.get("category") not in self.ACCOMMODATION_CATEGORIES]
        diurnal   = [p for p in non_accom if p.get("category") not in self.NOCTURNO_CATEGORIES]
        nocturnal = [p for p in non_accom if p.get("category") in self.NOCTURNO_CATEGORIES]

        print(f"\nPlaneando {len(route)} POIs em {total_days} dias "
              f"({len(diurnal)} diurnos, {len(nocturnal)} noturnos)...")
        print(f"   Tempo por dia: {self.minutes_per_day} min ({self.hours_per_day}h)\n")
        if first_day_start_time and first_day_start_time != self.start_time:
            print(f"   Dia 1 comeca as {first_day_start_time} (dias seguintes: {self.start_time})")
        if last_day_end_time:
            print(f"   Ultimo dia termina as {last_day_end_time}")

        # Distribuir POIs diurnos por dia (clustering geografico se possivel)
        diurnal_by_day = self._distribute_diurnal(diurnal, total_days, all_geos=all_geos,
                                                   route_direction=route_direction)
        # Limitar POIs no Dia 1 se começa à tarde (ex: sexta à noite)
        if self._day1_max_diurnal is not None and diurnal_by_day and len(diurnal_by_day[0]) > self._day1_max_diurnal:
            overflow = diurnal_by_day[0][self._day1_max_diurnal:]
            diurnal_by_day[0] = diurnal_by_day[0][:self._day1_max_diurnal]
            # Redistribuir overflow pelos outros dias
            if len(diurnal_by_day) > 1:
                for i, poi in enumerate(overflow):
                    diurnal_by_day[1 + (i % (len(diurnal_by_day) - 1))].append(poi)

        # Selecionar alojamento por dia: hotel mais proximo do centroide de cada dia
        day_hotels = self._assign_hotels(accommodation, total_days, diurnal_by_day)

        # Calcular weekday de cada dia se start_date conhecido (0=seg, 4=sex, 5=sáb)
        _weekday_by_day = None
        if start_date:
            try:
                from datetime import date as _dt, timedelta as _td
                _sd = _dt.fromisoformat(start_date)
                _weekday_by_day = [(_sd + _td(days=i)).weekday() for i in range(total_days)]
                _day_names = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
                print(f"   [Calendar] Dias: {[_day_names[w] for w in _weekday_by_day]}")
            except Exception:
                _weekday_by_day = None

        # Distribuir POIs noturnos com consciência de tempo (greedy, não round-robin)
        # Garante que nenhuma noite recebe mais bares do que cabem na janela 21:00-03:00
        nocturnal_by_day: List[List[Dict]] = [[] for _ in range(total_days)]
        last_day_blocks_night = (
            last_day_end_time is not None and
            total_days > 0 and
            self._parse_time(last_day_end_time) <= self._parse_time(self.NOCTURNAL_START)
        )
        available_night_days = total_days - 1 if last_day_blocks_night and total_days > 1 else total_days
        NIGHT_WINDOW = self._parse_time("24:00") + self._parse_time(self.NOCTURNAL_END) - self._parse_time(self.NOCTURNAL_START)
        night_time_used = [0] * total_days

        # Noites candidatas para bares: nunca penúltimo nem último dia
        _candidate_nights = [n for n in range(total_days) if n <= total_days - 3]
        # Se datas conhecidas: preferir 6ª (4) e sáb (5); fallback para qualquer noite
        if _weekday_by_day:
            _fri_sat = [n for n in _candidate_nights if _weekday_by_day[n] in (4, 5)]
            _other   = [n for n in _candidate_nights if _weekday_by_day[n] not in (4, 5)]
            _ordered_nights = _fri_sat + _other
        else:
            # Sem datas: escolher ~1 noite por semana aleatoriamente
            import random as _rand
            _n_bar_nights = max(1, round(total_days / 7))
            if len(_candidate_nights) <= _n_bar_nights:
                _ordered_nights = _candidate_nights
            else:
                _chosen = _rand.sample(_candidate_nights, _n_bar_nights)
                _other  = [n for n in _candidate_nights if n not in _chosen]
                _ordered_nights = _chosen + _other

        for poi in nocturnal:
            # Encontrar a primeira noite disponível onde o bar cabe
            assigned = False
            for night_idx in _ordered_nights:
                if (len(nocturnal_by_day[night_idx]) < self.MAX_NOCTURNAL_PER_DAY and
                        night_time_used[night_idx] + poi['duration'] <= NIGHT_WINDOW):
                    nocturnal_by_day[night_idx].append(poi)
                    night_time_used[night_idx] += poi['duration']
                    assigned = True
                    break
            # Se nenhuma noite tem espaço, o POI não é agendado (já foi removido de result.route)

        # Guardar estado dos clusters para Fill-D injectar POIs sem re-clustering
        self._last_diurnal_by_day   = diurnal_by_day
        self._last_nocturnal_by_day = nocturnal_by_day
        self._last_day_hotels       = day_hotels
        self._last_first_day_start  = first_day_start_time

        days = []
        is_last_day_departure = bool(last_day_end_time)
        for day_num in range(1, total_days + 1):
            d = diurnal_by_day[day_num - 1] if day_num <= len(diurnal_by_day) else []
            n = nocturnal_by_day[day_num - 1]
            # Sem hotel no último dia se o utilizador parte nesse dia
            is_last = (day_num == total_days)
            hotel = (day_hotels[day_num - 1] if day_num <= len(day_hotels) else None) if not (is_last and is_last_day_departure) else None
            day_start = first_day_start_time if day_num == 1 and first_day_start_time else self.start_time
            if d or n or hotel:
                days.append(self._format_day(day_num, d, n, day_start_time=day_start, hotel=hotel))

        return {
            "days": days,
            "total_days": len(days),
            "total_pois": len(route),
            "summary": self._generate_summary(days),
        }

    # -- Distribution --------------------------------------------------------

    def _assign_hotels(self, accommodation: List[Dict], n_days: int,
                        diurnal_by_day: List[List[Dict]] = None) -> List[Dict]:
        """
        Atribui 1 hotel por dia. Escolhe o hotel mais proximo do centroide
        dos POIs de cada dia. Nunca volta a um hotel já abandonado (anti A-B-A):
        uma vez que se muda de hotel X para Y, X fica na lista de "abandonados"
        e não é reconsiderado (excepto como fallback absoluto se não houver outro).
        """
        if not accommodation:
            return [None] * n_days
        if len(accommodation) == 1:
            return [accommodation[0]] * n_days

        result = []
        abandoned: set = set()  # nomes de hoteis já deixados para trás

        for day_idx in range(n_days):
            day_pois = (diurnal_by_day[day_idx]
                        if diurnal_by_day and day_idx < len(diurnal_by_day)
                        else [])

            if day_pois:
                clat = sum(p["lat"] for p in day_pois) / len(day_pois)
                clon = sum(p["lon"] for p in day_pois) / len(day_pois)
                by_dist = sorted(accommodation,
                                 key=lambda h: self._haversine(clat, clon, h["lat"], h["lon"]))
            else:
                by_dist = sorted(accommodation, key=lambda h: -h.get("score", 0.5))

            # Melhor = mais próximo não-abandonado; fallback ao mais próximo absoluto
            best = next((h for h in by_dist if h["name"] not in abandoned), by_dist[0])

            # "Ficar na mesma zona": se ontem's hotel está perto do melhor de hoje,
            # manter ontem's (evita mudanças desnecessárias em zonas próximas)
            if result and result[-1] is not None:
                prev = result[-1]
                if (prev["name"] not in abandoned and
                        self._haversine(prev["lat"], prev["lon"],
                                        best["lat"], best["lon"]) < 30.0):
                    best = prev

            # Se estamos a mudar de hotel, abandonar o de ontem
            if result and result[-1] is not None and result[-1]["name"] != best["name"]:
                abandoned.add(result[-1]["name"])

            result.append(best)
        return result

    @staticmethod
    def _seed_centroids(all_geos: Optional[List], n_days: int):
        """
        B2: gera `n_days` sementes (lat, lon) para o K-means a partir das
        cidades-foco da rota (all_geos). Se houver mais ou menos cidades do
        que dias, interpola pontos ao longo da sequencia de cidades — mantem
        a ordem do corredor. Devolve None se nao houver pelo menos 2 cidades
        (deixa o K-means inicializar normalmente).
        """
        if not all_geos or len(all_geos) < 2 or n_days < 1:
            return None
        pts = [(g[0], g[1]) for g in all_geos]
        n = len(pts)
        if n == n_days:
            return np.array(pts)
        seeds = []
        for i in range(n_days):
            t = i * (n - 1) / max(1, n_days - 1)
            idx = int(t)
            frac = t - idx
            if idx >= n - 1:
                seeds.append(pts[-1])
            else:
                lat = pts[idx][0] + frac * (pts[idx + 1][0] - pts[idx][0])
                lon = pts[idx][1] + frac * (pts[idx + 1][1] - pts[idx][1])
                seeds.append((lat, lon))
        return np.array(seeds)

    def _repair_city_coverage(self, by_day: List[List[Dict]], all_geos: List,
                               coverage_radius_km: float = 20.0) -> List[List[Dict]]:
        """
        B2: garante que cada cidade-foco (all_geos) tem pelo menos um POI
        atribuido a algum dia. Se nenhuma cidade tiver POIs num raio de
        `coverage_radius_km`, move o POI mais proximo dessa cidade (de
        qualquer dia, ate self.max_move_km — mesmo limite do _rebalance_category_diversity)
        para o dia cujo centroide atual esta mais perto da cidade.
        """
        n_days = len(by_day)
        for city_lat, city_lon, _radius in all_geos:
            covered = any(
                self._haversine(city_lat, city_lon, p['lat'], p['lon']) <= coverage_radius_km
                for cluster in by_day for p in cluster
            )
            if covered:
                continue

            best_poi, best_src, best_dist = None, None, float('inf')
            for src_idx, cluster in enumerate(by_day):
                for poi in cluster:
                    dist = self._haversine(city_lat, city_lon, poi['lat'], poi['lon'])
                    if dist < best_dist:
                        best_dist, best_poi, best_src = dist, poi, src_idx

            if best_poi is None or best_dist > self.max_move_km:
                continue

            def _centroid(cluster):
                if not cluster:
                    return (city_lat, city_lon)
                return (sum(p['lat'] for p in cluster) / len(cluster),
                        sum(p['lon'] for p in cluster) / len(cluster))

            target = min(range(n_days),
                          key=lambda i: self._haversine(city_lat, city_lon, *_centroid(by_day[i])))
            if target != best_src:
                by_day[best_src].remove(best_poi)
                by_day[target].append(best_poi)
        return by_day

    def _distribute_diurnal(self, diurnal: List[Dict], n_days: int,
                              all_geos: List = None, route_direction: str = None) -> List[List[Dict]]:
        if not diurnal:
            return [[] for _ in range(n_days)]

        # Separar restaurantes: atribuídos por proximidade ao centroide de cada dia,
        # garantindo ≥2 por dia independentemente do K-means geográfico
        restaurants = [p for p in diurnal if p.get('category') == 'restaurantes_e_cafes']
        non_meal    = [p for p in diurnal if p.get('category') != 'restaurantes_e_cafes']

        if not non_meal:
            by_day = [[] for _ in range(n_days)]
            for i, r in enumerate(restaurants):
                by_day[i % n_days].append(r)
            return by_day

        if len(non_meal) <= n_days:
            by_day = [[p] for p in non_meal]
            by_day += [[] for _ in range(n_days - len(non_meal))]
            self._assign_restaurants_to_days(by_day, restaurants)
            return by_day

        try:
            from sklearn.cluster import KMeans
            coords = np.array([[p["lat"], p["lon"]] for p in non_meal])
            # B2: sementes de K-means a partir das cidades-foco (all_geos), quando
            # disponiveis — melhora a separacao geografica em rotas multi-cidade
            seed_centroids = self._seed_centroids(all_geos, n_days)
            if seed_centroids is not None:
                labels = KMeans(n_clusters=n_days, init=seed_centroids, n_init=1,
                                 random_state=42).fit_predict(coords)
            else:
                labels = KMeans(n_clusters=n_days, random_state=42, n_init=10).fit_predict(coords)
            by_day: List[List[Dict]] = [[] for _ in range(n_days)]
            for i, poi in enumerate(non_meal):
                by_day[labels[i]].append(poi)
            # Post-K-means reassignment: recompute centroids and reassign each POI to
            # nearest centroid — corrects geographic outliers without hardcoded thresholds
            _centroids = []
            for _cluster in by_day:
                if _cluster:
                    _centroids.append((
                        sum(p['lat'] for p in _cluster) / len(_cluster),
                        sum(p['lon'] for p in _cluster) / len(_cluster),
                    ))
                else:
                    _centroids.append((0.0, 0.0))
            _reassigned: List[List[Dict]] = [[] for _ in range(n_days)]
            for _poi in non_meal:
                _best = min(range(n_days),
                            key=lambda _i: self._haversine(
                                _centroids[_i][0], _centroids[_i][1],
                                _poi['lat'], _poi['lon']))
                _reassigned[_best].append(_poi)
            by_day = _reassigned
            # B2: reparacao de cobertura — garante que cada cidade-foco tem pelo
            # menos 1 POI atribuido a algum dia, antes do rebalanceamento
            if all_geos and len(all_geos) > 1:
                by_day = self._repair_city_coverage(by_day, all_geos)
            # 1. Reequilibrar diversidade de categorias (sem restaurantes)
            by_day = self._rebalance_category_diversity(by_day, max_same_cat=2)
            # 2. Balancear tempo entre dias
            by_day = self._balance_time(by_day)
            # 3. Ordenar clusters pela direcção da rota
            by_day = self._sort_clusters_by_direction(by_day, all_geos, route_direction=route_direction)
            # 4. Hard cap por categoria (restaurantes tratados separadamente)
            by_day = self._enforce_category_caps(by_day, {}, default_cap=2)
            # 4a. Nunca repetir o mesmo POI (por nome) no mesmo dia -- excepto
            # praias (repetir a mesma praia em dias diferentes e normal)
            by_day = self._dedupe_same_day_poi(by_day)
            # 4b. Safety net: garantir ≥1 POI de actividade por dia
            self._ensure_min_activities(by_day)
            # 5. Atribuir ≥2 restaurantes por dia por proximidade geográfica
            self._assign_restaurants_to_days(by_day, restaurants)
            # 6. Ordem nearest-neighbour dentro de cada dia
            if self.start_lat:
                by_day = [self._nearest_neighbor_order(d, self.start_lat, self.start_lon)
                          if len(d) > 1 else d for d in by_day]
            return by_day
        except Exception:
            by_day = [[] for _ in range(n_days)]
            for i, poi in enumerate(diurnal):
                by_day[i % n_days].append(poi)
            return by_day

    def _assign_restaurants_to_days(self, by_day: List[List[Dict]],
                                     restaurants: List[Dict], per_day: int = 2) -> None:
        """Atribui até `per_day` restaurantes por dia por proximidade ao centroide do dia.
        Se o restaurante disponível mais próximo estiver a mais de max_move_km do centroide,
        procura o restaurante MAIS PRÓXIMO de todos (usado ou não) dentro de max_move_km
        e copia-o. Se não existir nenhum dentro desse raio, não atribui restaurante.
        Usa self.max_move_km (raio/dias, geo+transporte-aware) em vez de um valor fixo,
        evitando atribuir um restaurante distante do cluster geográfico do dia."""
        if not restaurants:
            return
        import copy as _copy
        MAX_DIST_KM = self.max_move_km
        used: set = set()
        n_days = len(by_day)
        for day_idx in range(n_days):
            non_rest = [p for p in by_day[day_idx] if p.get('category') != 'restaurantes_e_cafes']
            if non_rest:
                clat = sum(p['lat'] for p in non_rest) / len(non_rest)
                clon = sum(p['lon'] for p in non_rest) / len(non_rest)
            else:
                clat = sum(r['lat'] for r in restaurants) / len(restaurants)
                clon = sum(r['lon'] for r in restaurants) / len(restaurants)
            available = [r for r in restaurants if id(r) not in used]
            nearest = sorted(available, key=lambda r: self._haversine(clat, clon, r['lat'], r['lon']))
            assigned = 0
            for r in nearest[:per_day]:
                dist = self._haversine(clat, clon, r['lat'], r['lon'])
                if dist > MAX_DIST_KM:
                    # Procurar o restaurante mais próximo de todos (usado ou não)
                    nearest_any = min(restaurants,
                                      key=lambda x: self._haversine(clat, clon, x['lat'], x['lon']))
                    d_any = self._haversine(clat, clon, nearest_any['lat'], nearest_any['lon'])
                    if d_any <= MAX_DIST_KM:
                        by_day[day_idx].append(_copy.copy(nearest_any))
                        assigned += 1
                    # Se nenhum restaurante está dentro do raio, não atribuir
                    continue
                by_day[day_idx].append(r)
                used.add(id(r))
                assigned += 1
        # Restaurantes restantes (não usados): distribuir round-robin
        for i, r in enumerate(r for r in restaurants if id(r) not in used):
            by_day[i % n_days].append(r)

        # Segundo passe: dias com < per_day restaurantes ficam sem jantar.
        # Copiar o restaurante mais próximo (de qualquer dia) para preencher a lacuna.
        for day_idx in range(n_days):
            rests_in_day = [p for p in by_day[day_idx] if p.get('category') == 'restaurantes_e_cafes']
            missing = per_day - len(rests_in_day)
            if missing <= 0:
                continue
            non_rest = [p for p in by_day[day_idx] if p.get('category') != 'restaurantes_e_cafes']
            if non_rest:
                clat = sum(p['lat'] for p in non_rest) / len(non_rest)
                clon = sum(p['lon'] for p in non_rest) / len(non_rest)
            else:
                clat = sum(r['lat'] for r in restaurants) / len(restaurants)
                clon = sum(r['lon'] for r in restaurants) / len(restaurants)
            # Excluir restaurantes já atribuídos a este dia (por nome)
            existing_names = {p['name'] for p in rests_in_day}
            candidates = sorted(
                [r for r in restaurants if r['name'] not in existing_names],
                key=lambda r: self._haversine(clat, clon, r['lat'], r['lon'])
            )
            for r in candidates[:missing]:
                by_day[day_idx].append(_copy.copy(r))

    def _rebalance_category_diversity(self, by_day: List[List[Dict]], max_same_cat: int = 2) -> List[List[Dict]]:
        """
        Move POIs excedentes da mesma categoria para dias com menos dessa categoria,
        respeitando o budget de tempo por dia. Maximo max_same_cat POIs da mesma
        categoria por dia.
        """
        from collections import defaultdict
        n_days = len(by_day)
        if n_days <= 1:
            return by_day

        for _ in range(n_days * 3):  # iteracoes suficientes para convergir
            moved = False
            for src in range(n_days):
                cat_counts = defaultdict(list)
                for poi in by_day[src]:
                    cat_counts[poi['category']].append(poi)

                for cat, pois_in_cat in cat_counts.items():
                    if len(pois_in_cat) <= max_same_cat:
                        continue
                    # Mover os excedentes (os ultimos da lista - menos prioritarios)
                    for poi in pois_in_cat[max_same_cat:]:
                        # Encontrar o dia destino com menos desta categoria e com espaco
                        best_dst, best_count = None, float('inf')
                        for dst in range(n_days):
                            if dst == src:
                                continue
                            dst_cat_count = sum(1 for p in by_day[dst] if p['category'] == cat)
                            dst_time = sum(p['duration'] for p in by_day[dst])
                            if not (dst_cat_count < best_count and dst_time + poi['duration'] <= self.minutes_per_day):
                                continue
                            # Restrição geográfica: não mover mais do que max_move_km do centroide destino
                            if by_day[dst]:
                                d_lats = [p['lat'] for p in by_day[dst]]
                                d_lons = [p['lon'] for p in by_day[dst]]
                                d_clat = sum(d_lats) / len(d_lats)
                                d_clon = sum(d_lons) / len(d_lons)
                                if self._haversine(d_clat, d_clon, poi['lat'], poi['lon']) > self.max_move_km:
                                    continue
                            best_count = dst_cat_count
                            best_dst = dst
                        if best_dst is not None and best_count < max_same_cat:
                            by_day[src].remove(poi)
                            by_day[best_dst].append(poi)
                            moved = True
                            break  # recalcular cat_counts apos cada movimento
                    if moved:
                        break
            if not moved:
                break

        return by_day

    def _balance_time(self, by_day: List[List[Dict]]) -> List[List[Dict]]:
        """
        Move POIs de dias sobrecarregados para dias subaproveitados.
        Garante que nenhum dia tem menos de 60% do tempo alvo (se houver POIs para redistribuir).
        Apenas move POIs entre dias — não cria nem remove nenhum.
        Dias 0 e 1 (dias 1 e 2 do calendário) nunca recebem POIs: o dia 1 é gerido
        por _day1_max_diurnal em plan_days, e ambos podem ter restrições de chegada.
        """
        n_days = len(by_day)
        if n_days <= 2:
            return by_day
        # 360min = 6h de POIs/dia (480min - 60min refeições - 60min buffer trânsito)
        target = min(self.minutes_per_day, 360)
        # DST_HIGH = target: qualquer dia abaixo de 360min aceita POIs de sobrecarregados
        # DST_LOW  = 60% target: dias muito vazios aceitam de qualquer dia >60%
        DST_HIGH = target
        DST_LOW  = target * 0.60

        changed = True
        max_iters = n_days * 4
        iters = 0
        while changed and iters < max_iters:
            changed = False
            iters += 1
            for dst in range(n_days):
                dst_time = sum(p['duration'] for p in by_day[dst])
                if dst_time >= DST_HIGH:
                    continue
                # Aceita doações de: dias >100% (sempre) ou dias >60% se dst<60%
                src_order = sorted(
                    (i for i in range(n_days) if i != dst),
                    key=lambda i: sum(p['duration'] for p in by_day[i]),
                    reverse=True
                )
                for src in src_order:
                    src_time = sum(p['duration'] for p in by_day[src])
                    src_overloaded = src_time > target          # >100%
                    src_above_low  = src_time > DST_LOW         # >60%
                    dst_very_empty = dst_time < DST_LOW         # <60%
                    if not (src_overloaded or (src_above_low and dst_very_empty)):
                        continue  # não roubar de dias sem excesso
                    if not by_day[src]:
                        continue
                    # POI do src mais próximo do centroide do dst
                    if by_day[dst]:
                        clat = sum(p['lat'] for p in by_day[dst]) / len(by_day[dst])
                        clon = sum(p['lon'] for p in by_day[dst]) / len(by_day[dst])
                        candidate = min(by_day[src],
                                        key=lambda p: self._haversine(clat, clon, p['lat'], p['lon']))
                        # Não transferir POI geograficamente distante do cluster destino
                        if self._haversine(clat, clon, candidate['lat'], candidate['lon']) > self.max_move_km:
                            continue
                    else:
                        candidate = by_day[src][-1]
                    by_day[src].remove(candidate)
                    by_day[dst].append(candidate)
                    changed = True
                    break
        return by_day

    def _enforce_category_caps(self, by_day: List[List[Dict]],
                                caps: dict, default_cap: int = None) -> List[List[Dict]]:
        """
        Hard cap por categoria por dia: remove os excedentes com menor score.
        caps: overrides por categoria. default_cap: aplicado a todas as outras categorias se não None.
        """
        from collections import defaultdict
        for day in by_day:
            cat_groups: dict = defaultdict(list)
            for poi in day:
                cat_groups[poi['category']].append(poi)
            for cat, pois in cat_groups.items():
                cap = caps.get(cat, default_cap)
                if cap is not None and len(pois) > cap:
                    keep = sorted(pois, key=lambda p: -p.get('score', 0))[:cap]
                    for poi in pois:
                        if poi not in keep:
                            day.remove(poi)
        return by_day

    def _dedupe_same_day_poi(self, by_day: List[List[Dict]]) -> List[List[Dict]]:
        """
        Remove ocorrencias repetidas do mesmo POI (por nome) dentro do mesmo
        dia -- excepto praias, onde repetir e uma escolha valida (ex: voltar
        a mesma praia em dias diferentes de uma viagem de sol e praia).
        Mantem a ocorrencia com maior score de cada nome; a(s) restante(s)
        sao descartadas do dia (safety net _ensure_min_activities repoe se
        o dia ficar demasiado vazio).
        """
        for day in by_day:
            best_by_name: dict = {}
            for poi in day:
                if poi.get('category') == 'praias':
                    continue
                name = poi.get('name')
                if name not in best_by_name or poi.get('score', 0) > best_by_name[name].get('score', 0):
                    best_by_name[name] = poi
            day[:] = [poi for poi in day
                      if poi.get('category') == 'praias' or best_by_name.get(poi.get('name')) is poi]
        return by_day

    def _ensure_min_activities(self, by_day: List[List[Dict]], min_per_day: int = 1) -> None:
        """Garante que cada dia tem pelo menos min_per_day POIs de actividade.
        Chamado após _enforce_category_caps quando by_day ainda não tem restaurantes.
        """
        n_days = len(by_day)
        for _ in range(n_days * 2):
            counts = [len(d) for d in by_day]
            empty = [i for i, c in enumerate(counts) if c < min_per_day]
            if not empty:
                break
            richest = max(range(n_days), key=lambda i: counts[i])
            if counts[richest] <= min_per_day:
                break
            dst = empty[0]
            poi = by_day[richest][-1]
            by_day[richest].remove(poi)
            by_day[dst].append(poi)

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

    def _sort_clusters_by_direction(self, by_day: List[List[Dict]],
                                     all_geos: List = None,
                                     route_direction: str = None) -> List[List[Dict]]:
        """
        Ordena clusters por direcção: multi-waypoint projecta no eixo A→Z;
        localização única usa route_direction ('N2S'/'S2N') se fornecida.
        """
        n = len(by_day)
        centroids = []
        for day in by_day:
            if day:
                clat = sum(p['lat'] for p in day) / len(day)
                clon = sum(p['lon'] for p in day) / len(day)
            else:
                clat = clon = 0.0
            centroids.append((clat, clon))

        # Localização única com direcção explícita: ordenar por latitude
        if (not all_geos or len(all_geos) < 2) and route_direction in ("N2S", "S2N"):
            if route_direction == "N2S":
                order = sorted(range(n), key=lambda i: -centroids[i][0])  # decrescente
            else:
                order = sorted(range(n), key=lambda i: centroids[i][0])   # crescente
            return [by_day[i] for i in order]

        if not all_geos or len(all_geos) < 2:
            return by_day

        lat0, lon0 = all_geos[0][0], all_geos[0][1]
        lat1, lon1 = all_geos[-1][0], all_geos[-1][1]
        d = math.sqrt((lat1 - lat0) ** 2 + (lon1 - lon0) ** 2)
        if d <= 0.05:  # waypoints demasiado próximos — não ordenar
            return by_day

        dir_lat, dir_lon = (lat1 - lat0) / d, (lon1 - lon0) / d

        ref_lat = sum(c[0] for c in centroids) / n
        ref_lon = sum(c[1] for c in centroids) / n
        projs = [(c[0] - ref_lat) * dir_lat + (c[1] - ref_lon) * dir_lon
                 for c in centroids]
        order = sorted(range(n), key=lambda i: projs[i])
        return [by_day[i] for i in order]

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        R = 6371
        r = math.radians
        a = math.sin(r(lat2-lat1)/2)**2 + math.cos(r(lat1))*math.cos(r(lat2))*math.sin(r(lon2-lon1)/2)**2
        return R * 2 * math.asin(math.sqrt(a))

    # -- Formatting ----------------------------------------------------------

    def _format_day(self, day_num: int, diurnal: List[Dict], nocturnal: List[Dict],
                    day_start_time: str = None, hotel: Dict = None) -> Dict:

        LUNCH_START  = 12 * 60       # 12:00
        DINNER_START = 19 * 60       # 19:00
        DINNER_END   = 21 * 60 + 30  # 21:30

        restaurants = [p for p in diurnal if p.get('category') == 'restaurantes_e_cafes']
        activities  = [p for p in diurnal if p.get('category') != 'restaurantes_e_cafes']
        lunch_rest  = restaurants[0] if len(restaurants) >= 1 else None
        dinner_rest = restaurants[1] if len(restaurants) >= 2 else None
        _has_restaurant = bool(lunch_rest or dinner_rest)

        schedule       = []
        order          = 1
        scheduled_ids: set       = set()
        morning_immersive: set   = set()
        afternoon_immersive: set = set()

        effective_start = day_start_time if day_start_time else self.start_time
        current  = self._parse_time(effective_start)
        prev_lat = self.start_lat
        prev_lon = self.start_lon

        def _travel_to(poi: Dict) -> int:
            if prev_lat is not None:
                d_km = self._haversine(prev_lat, prev_lon, poi['lat'], poi['lon'])
                mode, _ = self._segment_transport(d_km, prev_lat, prev_lon, poi['lat'], poi['lon'])
                return self._travel_minutes(d_km, mode)
            return 0

        def _sched(poi: Dict, duration: int = None) -> None:
            nonlocal current, order, prev_lat, prev_lon
            dur = duration if duration is not None else poi['duration']
            arr = self._fmt(current)
            dep = self._fmt(current + dur)
            travel_km = None
            travel_mode = None
            travel_note = None
            if prev_lat is not None:
                travel_km = round(self._haversine(prev_lat, prev_lon, poi['lat'], poi['lon']), 2)
                travel_mode, travel_note = self._segment_transport(
                    travel_km, prev_lat, prev_lon, poi['lat'], poi['lon'])
            schedule.append({**poi, "arrival_time": arr, "departure_time": dep,
                              "order": order, "duration": dur,
                              "travel_km": travel_km, "travel_mode": travel_mode,
                              "travel_note": travel_note})
            order    += 1
            current  += dur
            prev_lat  = poi['lat']
            prev_lon  = poi['lon']
            scheduled_ids.add(id(poi))

        # ── MORNING BLOCK (start → 12:00) ─────────────────────────────────────
        _no_restaurant_immersive = False
        for poi in activities:
            cat = poi.get('category', '')
            is_immersive = cat in self.IMMERSIVE_CATEGORIES
            travel  = _travel_to(poi)
            t_start = current + travel

            if is_immersive:
                if cat in morning_immersive:
                    continue
                if not _has_restaurant:
                    # Full-day: praia/parque preenche o dia; utilizador come in situ
                    actual_dur = max(30, min(360, DINNER_START - t_start))
                    current = t_start
                    _sched(poi, actual_dur)
                    morning_immersive.add(cat)
                    _no_restaurant_immersive = True
                else:
                    # Com restaurante: clip à hora do almoço
                    if t_start >= LUNCH_START:
                        continue  # sem espaço de manhã; bloco da tarde trata isto
                    actual_dur = min(LUNCH_START - t_start, poi['duration'])
                    if actual_dur < 30:
                        continue
                    current = t_start
                    _sched(poi, actual_dur)
                    morning_immersive.add(cat)
            else:
                t_end = t_start + poi['duration']
                if t_end > LUNCH_START:
                    continue  # não cabe antes do almoço; bloco da tarde trata isto
                current = t_start
                _sched(poi)

        # ── ALMOÇO ────────────────────────────────────────────────────────────
        if lunch_rest:
            travel   = _travel_to(lunch_rest)
            t_arrive = current + travel
            current  = max(t_arrive, LUNCH_START)
            _sched(lunch_rest)
        elif not _no_restaurant_immersive:
            # Sem restaurante e sem POI imersivo full-day: pausa de 60 min
            current = max(current, LUNCH_START) + 60

        # ── AFTERNOON BLOCK (pós-almoço → 19:00) ──────────────────────────────
        for poi in activities:
            if id(poi) in scheduled_ids:
                continue
            cat = poi.get('category', '')
            is_immersive = cat in self.IMMERSIVE_CATEGORIES

            if is_immersive:
                if not _has_restaurant:
                    continue  # modo full-day: manhã já usou o dia inteiro
                if cat in afternoon_immersive:
                    continue

            travel  = _travel_to(poi)
            t_start = current + travel

            if t_start >= DINNER_START or t_start >= 20 * 60:
                continue

            if is_immersive:
                actual_dur = min(poi['duration'], DINNER_START - t_start)
                if actual_dur < 30:
                    continue
                current = t_start
                _sched(poi, actual_dur)
                afternoon_immersive.add(cat)
            else:
                t_end = t_start + poi['duration']
                if t_end > DINNER_START:
                    continue
                current = t_start
                _sched(poi)

        # ── JANTAR ────────────────────────────────────────────────────────────
        if dinner_rest:
            travel   = _travel_to(dinner_rest)
            t_arrive = current + travel
            current  = max(t_arrive, DINNER_START)
            if current < DINNER_END:
                _sched(dinner_rest)

        # ── BLOCO NOCTURNO (21:00 → 03:00) ────────────────────────────────────
        current = self._parse_time(self.NOCTURNAL_START)
        nocturnal_end_min = 24 * 60 + self._parse_time(self.NOCTURNAL_END)
        PUB_CRAWL_MIN   = 60
        is_pub_crawl    = len(nocturnal) > 1
        nocturnal_count = 0
        for poi in nocturnal:
            if nocturnal_count >= self.MAX_NOCTURNAL_PER_DAY:
                break
            sched_duration = PUB_CRAWL_MIN if is_pub_crawl else poi['duration']
            if current + sched_duration > nocturnal_end_min:
                break
            arr = self._fmt(current)
            dep = self._fmt(current + sched_duration)
            travel_km = None
            travel_mode = None
            travel_note = None
            if prev_lat is not None:
                travel_km = round(self._haversine(prev_lat, prev_lon, poi['lat'], poi['lon']), 2)
                travel_mode, travel_note = self._segment_transport(
                    travel_km, prev_lat, prev_lon, poi['lat'], poi['lon'])
            schedule.append({**poi, "arrival_time": arr, "departure_time": dep,
                              "order": order, "duration": sched_duration,
                              "travel_km": travel_km, "travel_mode": travel_mode,
                              "travel_note": travel_note})
            order   += 1
            current += sched_duration
            prev_lat = poi['lat']
            prev_lon = poi['lon']
            nocturnal_count += 1

        # ── HOTEL ─────────────────────────────────────────────────────────────
        if hotel:
            arr = self._fmt(current)
            dep = self._fmt(current + hotel.get('duration', 30))
            travel_km = None
            travel_mode = None
            travel_note = None
            if prev_lat is not None:
                travel_km = round(self._haversine(prev_lat, prev_lon, hotel['lat'], hotel['lon']), 2)
                travel_mode, travel_note = self._segment_transport(
                    travel_km, prev_lat, prev_lon, hotel['lat'], hotel['lon'])
            schedule.append({**hotel, "arrival_time": arr, "departure_time": dep,
                              "order": order, "is_accommodation": True,
                              "travel_km": travel_km, "travel_mode": travel_mode,
                              "travel_note": travel_note})
            order   += 1
            current += hotel.get('duration', 30)

        end_time   = schedule[-1]['departure_time'] if schedule else effective_start
        total_cost = sum(p['cost'] for p in schedule if not p.get('is_accommodation'))
        total_cost += hotel['cost'] if hotel else 0
        total_time = sum(p['duration'] for p in schedule if not p.get('is_accommodation'))

        return {
            "day": day_num,
            "pois": schedule,
            "total_time": total_time,
            "total_cost": total_cost,
            "n_pois": len(schedule),
            "start_time": effective_start,
            "end_time": end_time,
        }

    # -- Helpers -------------------------------------------------------------

    def _parse_time(self, time_str: str) -> int:
        h, m = map(int, time_str.split(':'))
        return h * 60 + m

    def _fmt(self, minutes: int) -> str:
        rounded = int(math.ceil(minutes / 10) * 10)  # arredondar para cima ao múltiplo de 10
        h = (rounded // 60) % 24
        m = rounded % 60
        return f"{h:02d}:{m:02d}"

    def _generate_summary(self, days: List[Dict]) -> str:
        lines = []
        for d in days:
            lines.append(f"Dia {d['day']}: {d['n_pois']} POIs, {d['total_time']} min "
                         f"({d['total_time']/60:.1f}h), EUR{d['total_cost']:.2f}")
        return "\n".join(lines)

    def print_itinerary(self, day_plan: Dict):
        print(f"\n{'='*70}")
        print(f"ITINERARIO - {day_plan['total_days']} DIAS")
        print(f"{'='*70}\n")
        for day in day_plan['days']:
            print(f"DIA {day['day']} - {day['start_time']} as {day['end_time']}")
            print(f"   {day['n_pois']} POIs | {day['total_time']} min | EUR{day['total_cost']:.2f}\n")
            for poi in day['pois']:
                prefix = "[N]" if poi.get("category") in self.NOCTURNO_CATEGORIES else "   "
                print(f"   {prefix} {poi['order']}. {poi['arrival_time']} - {poi['departure_time']}")
                print(f"         {poi['name']} ({poi['category']})")
                print(f"         Duracao: {poi['duration']} min | Custo: EUR{poi['cost']:.2f}\n")
        print(f"{'='*70}\n")
        print(day_plan['summary'])
        print(f"\n{'='*70}\n")
