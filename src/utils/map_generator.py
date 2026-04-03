# src/utils/map_generator.py

import folium
import requests
from pathlib import Path
from typing import List, Dict
import polyline
from datetime import datetime

class RouteMapGenerator:
    """Gera mapas interativos com rotas reais via OSRM"""
    
    def __init__(self):
        self.lisboa_center = [38.7223, -9.1393]
        self.osrm_url = "http://router.project-osrm.org/route/v1"
    
    def get_real_route(self, coordinates: List[List[float]], profile: str = "foot") -> Dict:
        """
        Obtém rota real usando OSRM
        
        Args:
            coordinates: [[lat, lon], [lat, lon], ...]
            profile: 'car', 'bike', 'foot'
        
        Returns:
            Dict com geometria da rota e distância/duração
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
                print(f"   ⚠️ OSRM erro: {data.get('message', 'Unknown')}")
                return None
        
        except Exception as e:
            print(f"   ⚠️ Erro ao chamar OSRM: {e}")
            return None
    
    def generate_map(self, route: List[Dict], output_file: str = None, algorithm: str = "", transport_mode: str = "foot") -> str:
        """
        Gera mapa interativo com rota REAL via OSRM
        
        Args:
            route: Lista de POIs com lat, lon, name, category
            output_file: Onde guardar o mapa HTML (se None, gera automaticamente com timestamp)
            algorithm: Nome do algoritmo usado (incluído no nome do ficheiro)
        
        Returns:
            Path do ficheiro HTML gerado
        """
        
        if not route:
            print("⚠️ Rota vazia, não é possível gerar mapa")
            return None
        
        # ✅ CORREÇÃO 1: Gerar nome de ficheiro com timestamp se não especificado
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            algo_suffix = f"_{algorithm}" if algorithm else ""
            output_file = f"outputs/route_map{algo_suffix}_{timestamp}.html"
        
        print(f"\n🗺️  Gerando mapa com OSRM...")
        
        MODE_LABELS = {
            "foot":             ("Pedonal 🚶",              "foot"),
            "car":              ("Carro 🚗",                "car"),
            "public_transport": ("Transportes Públicos 🚌", "foot"),
            "fastest":          ("Mais Rápido ⚡",           "car"),
        }
        mode_label, osrm_profile = MODE_LABELS.get(transport_mode, ("Pedonal 🚶", "foot"))
        
        # Criar mapa centrado em Lisboa
        m = folium.Map(
            location=self.lisboa_center,
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        
        # Cores por categoria
        colors = {
            'museus_e_palacios': 'blue',
            'monumentos': 'purple',
            'restaurantes_e_cafes': 'red',
            'bares_e_discotecas': 'darkred',
            'parques_e_reservas': 'green',
            'espacos_verdes': 'lightgreen',
            'praias': 'beige',
            'turismo_activo': 'orange',
            'arqueologia': 'cadetblue',
            'grutas': 'darkpurple',
            'zoos_e_aquarios': 'lightblue',
            'eventos': 'pink',
            'casinos': 'black',
            'turismo_espaco_rural': 'darkgreen',
            'hotelaria': 'gray',
            'localidade': 'lightgray',
        }
        
        # Extrair coordenadas dos POIs
        poi_coordinates = [[poi['lat'], poi['lon']] for poi in route]
        
        # ✅ OBTER ROTA REAL VIA OSRM
        osrm_route = self.get_real_route(poi_coordinates, profile=osrm_profile)
        
        if osrm_route and 'geometry' in osrm_route:
            # ✅ Desenhar rota REAL
            folium.PolyLine(
                osrm_route['geometry'],
                color='#3388ff',
                weight=5,
                opacity=0.8,
                popup=f"""
                    <b>Rota {mode_label}</b><br>
                    Distância: {osrm_route['distance']:.2f} km<br>
                    Duração: {osrm_route['duration']:.0f} min
                """,
                tooltip="Rota calculada pelo OpenStreetMap"
            ).add_to(m)
            
            print(f"   ✓ Rota OSRM: {osrm_route['distance']:.2f} km, {osrm_route['duration']:.0f} min")
        else:
            # Fallback: linha reta
            print("   ⚠️ OSRM falhou, usando linha reta")
            folium.PolyLine(
                poi_coordinates,
                color='blue',
                weight=3,
                opacity=0.5,
                dash_array='5'
            ).add_to(m)
        
        # ✅ ADICIONAR MARCADORES para cada POI
        for i, poi in enumerate(route, 1):
            lat = poi['lat']
            lon = poi['lon']
            name = poi['name']
            category = poi.get('category', 'attraction')
            duration = poi.get('duration', 60)
            cost = poi.get('cost', 0)
            
            # Marcador colorido
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(f"""
                    <div style="font-family: Arial; width: 200px;">
                        <h4 style="margin: 0 0 10px 0; color: {colors.get(category, 'gray')};">
                            {i}. {name}
                        </h4>
                        <p style="margin: 5px 0;">
                            <b>Categoria:</b> {category.title()}<br>
                            <b>Duração:</b> {duration} min<br>
                            <b>Custo:</b> €{cost:.2f}
                        </p>
                    </div>
                """, max_width=300),
                tooltip=f"{i}. {name}",
                icon=folium.Icon(
                    color=colors.get(category, 'gray'),
                    icon='info-sign',
                    prefix='glyphicon'
                )
            ).add_to(m)
            
            # Número do POI sobreposto
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(html=f"""
                    <div style="
                        font-size: 16px; 
                        font-weight: bold; 
                        color: white; 
                        background-color: {colors.get(category, 'gray')}; 
                        border-radius: 50%; 
                        width: 30px; 
                        height: 30px; 
                        text-align: center; 
                        line-height: 30px;
                        border: 3px solid white;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                    ">{i}</div>
                """)
            ).add_to(m)
        
        # ✅ AJUSTAR ZOOM para mostrar toda a rota
        if osrm_route and 'geometry' in osrm_route:
            m.fit_bounds(osrm_route['geometry'])
        elif len(poi_coordinates) > 1:
            m.fit_bounds(poi_coordinates)
        
        # ✅ ADICIONAR LEGENDA
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
            <p style="margin: 5px 0;"><b>Distância:</b> {osrm_route['distance']:.1f} km</p>
            <p style="margin: 5px 0;"><b>Deslocação:</b> {osrm_route['duration']:.0f} min</p>
            '''
        
        legend_html += '<hr style="margin: 10px 0;">'
        
        # Categorias presentes na rota
        route_categories = set(poi['category'] for poi in route)
        for cat in sorted(route_categories):
            color = colors.get(cat, 'gray')
            legend_html += f'''
            <p style="margin: 3px 0;">
                <span style="
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    background-color: {color};
                    border-radius: 50%;
                    margin-right: 5px;
                "></span>
                {cat.replace('_', ' ').title()}
            </p>
            '''
        
        legend_html += '</div>'
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # ✅ ADICIONAR INFO BOX no topo
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
            🗺️ Rota Turística
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
        
        print(f"   ✓ Mapa guardado: {output_path}")
        
        return str(output_path)