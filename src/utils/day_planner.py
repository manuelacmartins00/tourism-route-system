# src/utils/day_planner.py

from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import numpy as np

class DayPlanner:
    """
    Divide rotas turísticas em dias, respeitando:
    - Horários de funcionamento dos POIs
    - Tempo disponível por dia
    - Proximidade geográfica (clustering)
    - Fluxo lógico (manhã → tarde → noite)
    """
    NOCTURNO_CATEGORIES = {"bares_e_discotecas", "casinos"}
    DIURNO_CATEGORIES = {"monumentos", "museus_e_palacios", "espacos_verdes",
                          "parques_e_reservas", "arqueologia", "grutas",
                          "turismo_activo", "praias", "zoos_e_aquarios"}

    def _reorder_by_time_of_day(self, pois: List[Dict]) -> List[Dict]:
        """Reordena POIs dentro de um dia: diurnos primeiro, nocturnos no final."""
        diurnos   = [p for p in pois if p.get("category") in self.DIURNO_CATEGORIES]
        outros    = [p for p in pois if p.get("category") not in self.DIURNO_CATEGORIES
                     and p.get("category") not in self.NOCTURNO_CATEGORIES]
        nocturnos = [p for p in pois if p.get("category") in self.NOCTURNO_CATEGORIES]
        return diurnos + outros + nocturnos
    
    def __init__(self, 
                 hours_per_day: int = 8,
                 start_time: str = "09:00",
                 lunch_break: int = 60):
        """
        Args:
            hours_per_day: Horas úteis de turismo por dia (padrão: 8h)
            start_time: Hora de início diária (padrão: 09:00)
            lunch_break: Minutos de pausa para almoço (padrão: 60)
        """
        self.hours_per_day = hours_per_day
        self.minutes_per_day = hours_per_day * 60
        self.start_time = start_time
        self.lunch_break = lunch_break
    
    def plan_days(self, 
                  route: List[Dict], 
                  distance_matrix: np.ndarray = None,
                  total_days: int = None) -> Dict:
        """
        Divide rota em dias
        
        Args:
            route: Lista de POIs com lat, lon, duration, cost, etc.
            distance_matrix: Matriz de distâncias entre POIs (opcional)
            total_days: Número de dias disponíveis (se None, calcula automaticamente)
        
        Returns:
            Dict com rota organizada por dias
        """
        
        if not route:
            return {"days": [], "total_days": 0}
        
        # Calcular tempo total necessário
        total_time = sum(poi['duration'] for poi in route)
        
        # Se não especificou dias, calcular automaticamente
        if total_days is None:
            total_days = max(1, int(np.ceil(total_time / self.minutes_per_day)))
        
        print(f"\n📅 Planejando {len(route)} POIs em {total_days} dias...")
        print(f"   Tempo total: {total_time} min ({total_time/60:.1f}h)")
        print(f"   Tempo por dia: {self.minutes_per_day} min ({self.hours_per_day}h)\n")
        
        # Estratégia: Clustering geográfico + temporal
        if distance_matrix is not None and len(route) > 3:
            days = self._cluster_by_geography_and_time(route, distance_matrix, total_days)
        else:
            days = self._split_sequential(route, total_days)
        
        # Adicionar informações extras
        result = {
            "days": days,
            "total_days": len(days),
            "total_pois": len(route),
            "summary": self._generate_summary(days)
        }
        
        return result
    def _cluster_by_geography_and_time(self, route: List[Dict],
                                    distance_matrix: np.ndarray,
                                    n_days: int) -> List[Dict]:
        from sklearn.cluster import KMeans
        coords = np.array([[p["lat"], p["lon"]] for p in route])
        n_clusters = min(n_days, len(route))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        clusters = {}
        for i, poi in enumerate(route):
            clusters.setdefault(labels[i], []).append(poi)
        days = []
        day_num = 1
        for cluster_pois in clusters.values():
            current_day = []
            current_time = 0
            for poi in cluster_pois:
                if current_time + poi["duration"] > self.minutes_per_day and current_day:
                    current_day = self._reorder_by_time_of_day(current_day)
                    days.append(self._format_day(day_num, current_day, current_time))
                    current_day = []
                    current_time = 0
                    day_num += 1
                current_day.append(poi)
                current_time += poi["duration"]
            if current_day:
                current_day = self._reorder_by_time_of_day(current_day)
                days.append(self._format_day(day_num, current_day, current_time))
                day_num += 1
        return days

    def _split_sequential(self, route: List[Dict], n_days: int) -> List[Dict]:
        """
        Divisão sequencial simples (sem otimização geográfica)
        """
        
        days = []
        current_day = []
        current_time = 0
        day_num = 1
        
        for poi in route:
            poi_time = poi['duration']
            
            # Se adicionar este POI ultrapassar o tempo do dia, inicia novo dia
            if current_time + poi_time > self.minutes_per_day and current_day:
                current_day = self._reorder_by_time_of_day(current_day)
                days.append(self._format_day(day_num, current_day, current_time))
                current_day = []
                current_time = 0
                day_num += 1
            
            current_day.append(poi)
            current_time += poi_time
        
        # Adicionar último dia
        if current_day:
            current_day = self._reorder_by_time_of_day(current_day)
            days.append(self._format_day(day_num, current_day, current_time))
        
        return days
    
    
    def _format_day(self, day_num: int, pois: List[Dict], total_time: int) -> Dict:
        """Formata informações de um dia"""
        
        total_cost = sum(poi['cost'] for poi in pois)
        
        # Adicionar horários estimados
        current_time = self._parse_time(self.start_time)
        schedule = []
        
        for i, poi in enumerate(pois):
            arrival_time = self._format_time(current_time)
            departure_time = self._format_time(current_time + poi['duration'])
            
            schedule.append({
                **poi,
                "arrival_time": arrival_time,
                "departure_time": departure_time,
                "order": i + 1
            })
            
            current_time += poi['duration']
            
            # Adicionar pausa para almoço (meio-dia)
            if i < len(pois) - 1 and 12 * 60 < current_time < 14 * 60:
                current_time += self.lunch_break
        
        return {
            "day": day_num,
            "pois": schedule,
            "total_time": total_time,
            "total_cost": total_cost,
            "n_pois": len(pois),
            "start_time": self.start_time,
            "end_time": self._format_time(current_time)
        }
    
    def _generate_summary(self, days: List[Dict]) -> str:
        """Gera resumo textual do planejamento"""
        
        summary = []
        
        for day in days:
            day_num = day['day']
            n_pois = day['n_pois']
            time = day['total_time']
            cost = day['total_cost']
            
            summary.append(
                f"Dia {day_num}: {n_pois} POIs, {time} min ({time/60:.1f}h), €{cost:.2f}"
            )
        
        return "\n".join(summary)
    
    def _parse_time(self, time_str: str) -> int:
        """Converte "09:30" para minutos desde meia-noite"""
        h, m = map(int, time_str.split(':'))
        return h * 60 + m
    
    def _format_time(self, minutes: int) -> str:
        """Converte minutos desde meia-noite para "09:30" """
        hours = int(minutes // 60) % 24
        mins = int(minutes % 60)
        return f"{hours:02d}:{mins:02d}"
    
    def print_itinerary(self, day_plan: Dict):
        """Imprime itinerário formatado"""
        
        print(f"\n{'='*70}")
        print(f"📅 ITINERÁRIO - {day_plan['total_days']} DIAS")
        print(f"{'='*70}\n")
        
        for day in day_plan['days']:
            print(f"📆 DIA {day['day']} - {day['start_time']} às {day['end_time']}")
            print(f"   {day['n_pois']} POIs | {day['total_time']} min | €{day['total_cost']:.2f}\n")
            
            for poi in day['pois']:
                print(f"   {poi['order']}. {poi['arrival_time']} - {poi['departure_time']}")
                print(f"      {poi['name']} ({poi['category']})")
                print(f"      Duração: {poi['duration']} min | Custo: €{poi['cost']:.2f}\n")
        
        print(f"{'='*70}\n")
        print(day_plan['summary'])
        print(f"\n{'='*70}\n")