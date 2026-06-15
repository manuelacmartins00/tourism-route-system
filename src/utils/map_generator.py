# src/utils/map_generator.py

import folium
from folium.plugins import Fullscreen
import requests
from pathlib import Path
from typing import List, Dict, Optional
import polyline
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

class RouteMapGenerator:
    """Gera mapas interativos com rotas reais via OSRM"""
    
    def __init__(self):
        self.lisboa_center = [38.7223, -9.1393]
        self.osrm_url = "http://router.project-osrm.org/route/v1"
    
    def get_real_route(self, coordinates: List[List[float]], profile: str = "foot") -> Dict:
        """
        Obtem rota real usando OSRM
        
        Args:
            coordinates: [[lat, lon], [lat, lon], ...]
            profile: 'car', 'bike', 'foot'
        
        Returns:
            Dict com geometria da rota e distancia/duracao
        """
        
        if len(coordinates) < 2:
            return None
        
        # Converter para formato lon,lat (OSRM usa lon,lat!)
        coords_str = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
        
        url = f"{self.osrm_url}/{profile}/{coords_str}"
        params = {
            'overview': 'full',
            'geometries': 'polyline',
            'steps': 'true'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data['code'] == 'Ok':
                route = data['routes'][0]

                # Descodificar polyline
                geometry_encoded = route['geometry']
                geometry_decoded = polyline.decode(geometry_encoded)

                return {
                    'geometry': geometry_decoded,
                    'distance': route['distance'] / 1000,
                    'duration': route['duration'] / 60,
                    'steps': route.get('legs', [])
                }
            else:
                print(f"   AVISO: OSRM erro: {data.get('message', 'Unknown')}")
                return None

        except Exception as e:
            print(f"   AVISO: Erro ao chamar OSRM: {e}")
            return None
    
    def _transit_geometry_safe(self, transit_service, coords_a, coords_b, timeout=4):
        """Chama get_route_geometry com timeout para evitar bloquear o mapa."""
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(transit_service.get_route_geometry, coords_a, coords_b)
            try:
                return future.result(timeout=timeout)
            except (FuturesTimeout, Exception):
                return None

    def _transit_segments_safe(self, transit_service, coords_a, coords_b, timeout=4):
        """Chama get_route_segments com timeout para evitar bloquear o mapa."""
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(transit_service.get_route_segments, coords_a, coords_b)
            try:
                return future.result(timeout=timeout)
            except (FuturesTimeout, Exception):
                return None

    def generate_map(self, route: List[Dict], output_file: str = None, algorithm: str = "",
                     transport_mode: str = "foot", day_plan: dict = None,
                     transit_service=None) -> str:
        """
        Gera mapa interativo com rota REAL via OSRM
        
        Args:
            route: Lista de POIs com lat, lon, name, category
            output_file: Onde guardar o mapa HTML (se None, gera automaticamente com timestamp)
            algorithm: Nome do algoritmo usado (incluido no nome do ficheiro)
        
        Returns:
            Path do ficheiro HTML gerado
        """
        
        if not route:
            print("AVISO: Rota vazia, nao e possivel gerar mapa")
            return None

        # Gerar nome de ficheiro com timestamp se nao especificado
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            algo_suffix = f"_{algorithm}" if algorithm else ""
            output_file = f"outputs/route_map{algo_suffix}_{timestamp}.html"
        
        print(f"\nGerando mapa com OSRM...")

        MODE_LABELS = {
            "foot":             ("Pedonal",              "foot"),
            "car":              ("Carro",                "car"),
            "public_transport": ("Transportes Publicos", "foot"),
            "fastest":          ("Mais Rapido",           "car"),
        }
        mode_label, osrm_profile = MODE_LABELS.get(transport_mode, ("Pedonal", "foot"))
        
        # Criar mapa centrado em Lisboa
        m = folium.Map(
            location=self.lisboa_center,
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        Fullscreen(position='topleft').add_to(m)

        # Paleta de cores por dia (Folium named + CSS hex equivalents)
        DAY_FOLIUM = [
            'red', 'blue', 'green', 'purple', 'orange',
            'darkred', 'cadetblue', 'darkblue', 'darkgreen', 'pink',
        ]
        DAY_CSS = [
            '#e31a1c', '#1f78b4', '#33a02c', '#6a3d9a', '#ff7f00',
            '#b15928', '#74add1', '#313695', '#006837', '#f768a1',
        ]
        
        # Extrair coordenadas dos POIs (para marcadores)
        poi_coordinates = [[poi['lat'], poi['lon']] for poi in route]

        # Para a rota OSRM usar a ordem do day_plan (visita dia-a-dia).
        # O otimizador ordena por fitness global; o day_plan agrupa geograficamente
        # por dia com nearest-neighbour — seguir esta ordem elimina rotas A-B-A e
        # distâncias OSRM absurdas.
        if day_plan and day_plan.get("days"):
            _ordered_pois = [p for day in day_plan["days"] for p in day["pois"]]
            _route_coords = [[p['lat'], p['lon']] for p in _ordered_pois]
        else:
            _ordered_pois = list(route)
            _route_coords = poi_coordinates

        # Construir mapa de (poi_name -> (dia, ordem_no_dia)) a partir do day_plan
        # (necessario antes para atribuir segmentos/marcadores aos FeatureGroups por dia — S4)
        poi_day_label = {}
        if day_plan and day_plan.get("days"):
            for day in day_plan["days"]:
                for p in day["pois"]:
                    poi_day_label[p["name"]] = (day["day"], p["order"])

        # FeatureGroups por dia (S4) — permite ligar/desligar cada dia no mapa
        day_groups: dict = {}
        def _get_day_group(day_num: int):
            if day_num not in day_groups:
                label = f"Dia {day_num}" if day_num > 0 else "Outros"
                day_groups[day_num] = folium.FeatureGroup(name=label, show=True)
            return day_groups[day_num]

        # [OK] OBTER ROTA REAL VIA OSRM
        osrm_route = self.get_real_route(_route_coords, profile=osrm_profile)

        route_group = folium.FeatureGroup(name="Rota completa", show=True)
        if osrm_route and 'geometry' in osrm_route:
            folium.PolyLine(
                osrm_route['geometry'],
                color='#3388ff',
                weight=5,
                opacity=0.8,
                popup=f"""
                    <b>Rota {mode_label}</b><br>
                    Distancia: {osrm_route['distance']:.2f} km<br>
                    Duracao: {osrm_route['duration']:.0f} min
                """,
                tooltip="Rota calculada pelo OpenStreetMap"
            ).add_to(route_group)
            print(f"   [OK] Rota OSRM: {osrm_route['distance']:.2f} km, {osrm_route['duration']:.0f} min")
        else:
            print("   AVISO: OSRM falhou, usando linha reta")
            folium.PolyLine(
                poi_coordinates,
                color='blue',
                weight=3,
                opacity=0.5,
                dash_array='5'
            ).add_to(route_group)
        route_group.add_to(m)

        # Paragens GTFS para transportes públicos — apenas as linhas/segmentos
        # efectivamente usados na rota, agrupados por linha (B5/S6)
        if transport_mode == "public_transport" and transit_service is not None:
            n_transit = 0
            n_legs = 0
            for idx in range(len(_ordered_pois) - 1):
                a = _ordered_pois[idx]
                b = _ordered_pois[idx + 1]
                segments = self._transit_segments_safe(
                    transit_service,
                    (a['lat'], a['lon']),
                    (b['lat'], b['lon'])
                )
                if not segments:
                    continue
                seg_day = poi_day_label[b["name"]][0] if b["name"] in poi_day_label else 0
                seg_group = _get_day_group(seg_day)
                for seg in segments:
                    geom = seg["geometry"]
                    if len(geom) < 2:
                        continue
                    if seg["is_walk"]:
                        # Perna a pé entre paragens/transferências
                        folium.PolyLine(
                            geom,
                            color='#999999',
                            weight=3,
                            opacity=0.7,
                            dash_array='4',
                            tooltip="A pé (transferência)"
                        ).add_to(seg_group)
                    else:
                        line_label = seg["route_id"] or seg["operator"] or "Transportes Públicos"
                        folium.PolyLine(
                            geom,
                            color='#ff6600',
                            weight=4,
                            opacity=0.9,
                            dash_array=None,
                            tooltip=f"Linha {line_label}"
                        ).add_to(seg_group)
                        n_legs += 1
                    # Marcadores de paragem (círculos pequenos), excl. extremos
                    for stop in geom[1:-1]:
                        folium.CircleMarker(
                            location=stop,
                            radius=4,
                            color='#ff6600',
                            fill=True,
                            fill_color='white',
                            fill_opacity=1.0,
                            tooltip="Paragem / Estação"
                        ).add_to(seg_group)
                n_transit += 1
            if n_transit:
                print(f"   [OK] Rotas GTFS desenhadas: {n_transit} trajetos, {n_legs} linhas")

        # [OK] ADICIONAR MARCADORES para cada POI
        for i, poi in enumerate(route, 1):
            lat = poi['lat']
            lon = poi['lon']
            name = poi['name']
            category = poi.get('category', 'attraction')
            duration = poi.get('duration', 60)
            cost = poi.get('cost', 0)

            day_num = poi_day_label[name][0] if name in poi_day_label else 0
            if day_num > 0:
                f_color = DAY_FOLIUM[(day_num - 1) % len(DAY_FOLIUM)]
                css_color = DAY_CSS[(day_num - 1) % len(DAY_CSS)]
            else:
                f_color = 'gray'
                css_color = '#888888'

            label = f"D{day_num}-{poi_day_label[name][1]}" if name in poi_day_label else str(i)
            badge = f"{chr(64+day_num)}{poi_day_label[name][1]}" if name in poi_day_label else str(i)
            poi_group = _get_day_group(day_num)

            # Marcador colorido
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(f"""
                    <div style="font-family: Arial; width: 200px;">
                        <h4 style="margin: 0 0 10px 0; color: {css_color};">
                            {badge}. {name}
                        </h4>
                        <p style="margin: 5px 0;">
                            <b>Categoria:</b> {category.replace('_', ' ').title()}<br>
                            <b>Duracao:</b> {duration} min<br>
                            <b>Custo:</b> EUR{cost:.2f}
                        </p>
                    </div>
                """, max_width=300),
                tooltip=f"{label}. {name}",
                icon=folium.Icon(
                    color=f_color,
                    icon='info-sign',
                    prefix='glyphicon'
                )
            ).add_to(poi_group)

            # Numero do POI sobreposto
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(html=f"""
                    <div style="
                        font-size: 14px;
                        font-weight: bold;
                        color: white;
                        background-color: {css_color};
                        border-radius: 50%;
                        width: 30px;
                        height: 30px;
                        text-align: center;
                        line-height: 30px;
                        border: 3px solid white;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                    ">{badge}</div>
                """)
            ).add_to(poi_group)

        # Adicionar todos os FeatureGroups por dia ao mapa + controlo de camadas (S4)
        for day_num in sorted(day_groups.keys()):
            day_groups[day_num].add_to(m)
        # S4: topleft para nao ficar atras da caixa "Rota Turistica" (topright)
        folium.LayerControl(collapsed=False, position='topleft').add_to(m)

        # [OK] AJUSTAR ZOOM para mostrar toda a rota
        if osrm_route and 'geometry' in osrm_route:
            m.fit_bounds(osrm_route['geometry'])
        elif len(_route_coords) > 1:
            m.fit_bounds(_route_coords)
        
        # [OK] ADICIONAR LEGENDA
        legend_html = f'''
        <div style="
            position: fixed; 
            bottom: 50px; 
            left: 50px; 
            width: 220px; 
            background-color: white; 
            border: 2px solid grey; 
            z-index: 9999; 
            font-size: 13px;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        ">
        <h4 style="margin-top: 0; border-bottom: 2px solid #ddd; padding-bottom: 5px;">
            🔍 Legenda
        </h4>
        <p style="margin: 5px 0;"><b>Total POIs:</b> {len(route)}</p>
        '''
        
        if osrm_route:
            legend_html += f'''
            <p style="margin: 5px 0;"><b>Distancia:</b> {osrm_route['distance']:.1f} km</p>
            <p style="margin: 5px 0;"><b>Deslocacao:</b> {osrm_route['duration']:.0f} min</p>
            '''
        
        legend_html += '<hr style="margin: 10px 0;">'

        # Dias presentes na rota
        if day_plan and day_plan.get("days"):
            for day in day_plan["days"]:
                d = day["day"]
                css_color = DAY_CSS[(d - 1) % len(DAY_CSS)]
                legend_html += f'''
                <p style="margin: 3px 0;">
                    <span style="
                        display: inline-block;
                        width: 14px;
                        height: 14px;
                        background-color: {css_color};
                        border-radius: 50%;
                        margin-right: 5px;
                        vertical-align: middle;
                    "></span>
                    <b>Dia {d}</b> ({len(day["pois"])} paragens) — letra "{chr(64 + d)}"
                </p>
                '''
            legend_html += '''
            <p style="margin: 8px 0 0; font-size: 11px; color: #666;">
                Cada marcador mostra <b>Letra do dia + número da ordem</b> de visita
                (ex: B2 = Dia 2, 2ª paragem)
            </p>
            '''
        else:
            route_categories = set(poi['category'] for poi in route)
            for cat in sorted(route_categories):
                legend_html += f'''
                <p style="margin: 3px 0;">
                    <span style="
                        display: inline-block;
                        width: 12px;
                        height: 12px;
                        background-color: gray;
                        border-radius: 50%;
                        margin-right: 5px;
                    "></span>
                    {cat.replace('_', ' ').title()}
                </p>
                '''

        legend_html += '</div>'
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # [OK] ADICIONAR INFO BOX no topo
        info_html = f'''
        <div style="
            position: fixed;
            top: 10px;
            right: 10px;
            width: 300px;
            background-color: white;
            border: 2px solid #3388ff;
            z-index: 9999;
            font-size: 12px;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        ">
        <h3 style="margin: 0 0 10px 0; color: #3388ff;">
            🗺 Rota Turistica
        </h3>
        <p style="margin: 5px 0;">
            <b>Powered by:</b> OpenStreetMap + OSRM<br>
            <b>Modo:</b> {mode_label}
        </p>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(info_html))
        
        # Guardar
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        m.save(str(output_path))
        
        print(f"   OK Mapa guardado: {output_path}")
        
        return str(output_path)