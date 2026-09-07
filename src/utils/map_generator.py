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

    CATEGORY_LABELS_EN = {
        "restaurantes_e_cafes": "Restaurants & Cafés", "monumentos": "Monuments",
        "turismo_activo": "Active Tourism", "praias": "Beaches", "praia": "Beach",
        "bares_e_discotecas": "Bars & Nightclubs", "museus_e_palacios": "Museums & Palaces",
        "eventos": "Events", "campos": "Fields", "arqueologia": "Archaeology",
        "espacos_verdes": "Green Spaces", "marinas_e_portos": "Marinas & Ports",
        "termas": "Thermal Spas", "parques_e_reservas": "Parks & Reserves",
        "parques_de_diversao": "Amusement Parks", "zoos_e_aquarios": "Zoos & Aquariums",
        "ciencia_e_conhecimento": "Science & Knowledge", "casinos": "Casinos",
        "talassoterapia": "Thalassotherapy", "grutas": "Caves", "academias": "Gyms",
        "barragens": "Dams",
    }

    def generate_map(self, route: List[Dict], output_file: str = None, algorithm: str = "",
                     transport_mode: str = "foot", day_plan: dict = None,
                     transit_service=None, language: str = "pt") -> str:
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

        is_en = (language or "pt").lower().startswith("en")

        # Gerar nome de ficheiro com timestamp se nao especificado
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            algo_suffix = f"_{algorithm}" if algorithm else ""
            output_file = f"outputs/route_map{algo_suffix}_{timestamp}.html"

        print(f"\nGerando mapa com OSRM...")

        MODE_LABELS = {
            "foot":             ("On foot" if is_en else "Pedonal",              "foot"),
            "car":              ("Car" if is_en else "Carro",                   "car"),
            "public_transport": ("Public Transport" if is_en else "Transportes Publicos", "foot"),
            "fastest":          ("Fastest" if is_en else "Mais Rapido",         "car"),
        }
        mode_label, osrm_profile = MODE_LABELS.get(
            transport_mode, ("On foot" if is_en else "Pedonal", "foot"))
        _day_word = "Day" if is_en else "Dia"
        _other_word = "Other" if is_en else "Outros"
        _route_word = "Route" if is_en else "Rota"
        
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
                label = f"{_day_word} {day_num}" if day_num > 0 else _other_word
                day_groups[day_num] = folium.FeatureGroup(name=label, show=True)
            return day_groups[day_num]

        # OSRM por dia — cada polyline vai para o FeatureGroup do seu dia
        # (elimina o grupo "Rota completa" separado dos marcadores)
        all_day_geometries: list = []
        total_distance_km = 0.0
        total_duration_min = 0.0

        if day_plan and day_plan.get("days"):
            for day in day_plan["days"]:
                day_pois = day["pois"]
                d = day["day"]
                day_group = _get_day_group(d)
                css_color = DAY_CSS[(d - 1) % len(DAY_CSS)]
                if len(day_pois) < 2:
                    continue
                day_coords = [[p['lat'], p['lon']] for p in day_pois]
                day_osrm = self.get_real_route(day_coords, profile=osrm_profile)
                if day_osrm and 'geometry' in day_osrm:
                    folium.PolyLine(
                        day_osrm['geometry'],
                        color=css_color,
                        weight=4,
                        opacity=0.75,
                        tooltip=f"{_day_word} {d} · {day_osrm['distance']:.1f} km · {day_osrm['duration']:.0f} min"
                    ).add_to(day_group)
                    all_day_geometries.extend(day_osrm['geometry'])
                    total_distance_km += day_osrm['distance']
                    total_duration_min += day_osrm['duration']
                else:
                    folium.PolyLine(
                        day_coords, color=css_color, weight=3, opacity=0.5, dash_array='5'
                    ).add_to(day_group)
                    all_day_geometries.extend(day_coords)
            if all_day_geometries:
                print(f"   [OK] Rotas OSRM por dia: {total_distance_km:.1f} km total, {total_duration_min:.0f} min")
        else:
            # Sem day_plan: rota única
            osrm_single = self.get_real_route(_route_coords, profile=osrm_profile)
            route_group = folium.FeatureGroup(name=_route_word, show=True)
            if osrm_single and 'geometry' in osrm_single:
                folium.PolyLine(
                    osrm_single['geometry'], color='#3388ff', weight=5, opacity=0.8,
                    tooltip=f"{_route_word} {mode_label}"
                ).add_to(route_group)
                all_day_geometries = osrm_single['geometry']
                total_distance_km = osrm_single['distance']
                total_duration_min = osrm_single['duration']
                print(f"   [OK] Rota OSRM: {total_distance_km:.2f} km, {total_duration_min:.0f} min")
            else:
                print("   AVISO: OSRM falhou, usando linha reta")
                folium.PolyLine(
                    poi_coordinates, color='blue', weight=3, opacity=0.5, dash_array='5'
                ).add_to(route_group)
                all_day_geometries = poi_coordinates
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
                            tooltip=("On foot (transfer)" if is_en else "A pé (transferência)")
                        ).add_to(seg_group)
                    else:
                        _public_transport_word = "Public Transport" if is_en else "Transportes Públicos"
                        line_label = seg["route_id"] or seg["operator"] or _public_transport_word
                        _line_word = "Line" if is_en else "Linha"
                        folium.PolyLine(
                            geom,
                            color='#ff6600',
                            weight=4,
                            opacity=0.9,
                            dash_array=None,
                            tooltip=f"{_line_word} {line_label}"
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
                            tooltip=("Stop / Station" if is_en else "Paragem / Estação")
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

            category_display = (self.CATEGORY_LABELS_EN.get(category, category.replace('_', ' ').title())
                                 if is_en else category.replace('_', ' ').title())
            _cat_word = "Category" if is_en else "Categoria"
            _dur_word = "Duration" if is_en else "Duracao"
            _cost_word = "Cost" if is_en else "Custo"

            # Marcador colorido
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(f"""
                    <div style="font-family: Arial; width: 200px;">
                        <h4 style="margin: 0 0 10px 0; color: {css_color};">
                            {badge}. {name}
                        </h4>
                        <p style="margin: 5px 0;">
                            <b>{_cat_word}:</b> {category_display}<br>
                            <b>{_dur_word}:</b> {duration} min<br>
                            <b>{_cost_word}:</b> EUR{cost:.2f}
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
        if all_day_geometries:
            m.fit_bounds(all_day_geometries)
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
            {"Legend" if is_en else "Legenda"}
        </h4>
        <p style="margin: 5px 0;"><b>{"Total POIs" if is_en else "Total POIs"}:</b> {len(route)}</p>
        '''

        if total_distance_km > 0:
            legend_html += f'''
            <p style="margin: 5px 0;"><b>{"Distance" if is_en else "Distancia"}:</b> {total_distance_km:.1f} km</p>
            <p style="margin: 5px 0;"><b>{"Travel time" if is_en else "Deslocacao"}:</b> {total_duration_min:.0f} min</p>
            '''

        legend_html += '<hr style="margin: 10px 0;">'

        # Dias presentes na rota
        if day_plan and day_plan.get("days"):
            for day in day_plan["days"]:
                d = day["day"]
                css_color = DAY_CSS[(d - 1) % len(DAY_CSS)]
                _stops_word = "stops" if is_en else "paragens"
                _letter_word = "letter" if is_en else "letra"
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
                    <b>{_day_word} {d}</b> ({len(day["pois"])} {_stops_word}) — {_letter_word} "{chr(64 + d)}"
                </p>
                '''
            if is_en:
                legend_html += '''
                <p style="margin: 8px 0 0; font-size: 11px; color: #666;">
                    Each marker shows the <b>day letter + visit order number</b>
                    (e.g. B2 = Day 2, 2nd stop)
                </p>
                '''
            else:
                legend_html += '''
                <p style="margin: 8px 0 0; font-size: 11px; color: #666;">
                    Cada marcador mostra <b>Letra do dia + número da ordem</b> de visita
                    (ex: B2 = Dia 2, 2ª paragem)
                </p>
                '''
        else:
            route_categories = set(poi['category'] for poi in route)
            for cat in sorted(route_categories):
                cat_display = (self.CATEGORY_LABELS_EN.get(cat, cat.replace('_', ' ').title())
                                if is_en else cat.replace('_', ' ').title())
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
                    {cat_display}
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
            {"Tourist Route" if is_en else "Rota Turistica"}
        </h3>
        <p style="margin: 5px 0;">
            <b>Powered by:</b> OpenStreetMap + OSRM<br>
            <b>{"Mode" if is_en else "Modo"}:</b> {mode_label}
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