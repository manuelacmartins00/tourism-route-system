# src/utils/data_loader.py

import json
from typing import List, Dict
from pathlib import Path

def load_pois_from_json(filepath: str = "data/pois_structured_for_rag.json") -> List[Dict]:
    """
    Carrega POIs do ficheiro JSON
    
    Args:
        filepath: Caminho para o ficheiro JSON
    
    Returns:
        Lista de dicionários com POIs
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['pois']

def convert_to_poi_objects(pois_data: List[Dict]):
    """
    Converte dicionários em objetos POI
    
    Args:
        pois_data: Lista de dicionários
    
    Returns:
        Lista de objetos POI
    """
    from src.optimizers.route_evaluator import POI
    
    poi_objects = []
    for poi_dict in pois_data:
        if 'location' in poi_dict:  # Formato estruturado
            poi = POI(
                id=poi_dict['id'],
                name=poi_dict['name'],
                lat=poi_dict['location']['lat'],
                lon=poi_dict['location']['lon'],
                category=poi_dict['category'],
                score=poi_dict['attributes']['score'],
                duration=poi_dict['attributes']['duration_minutes'],
                opening_time=poi_dict['schedule']['opening_time'],
                closing_time=poi_dict['schedule']['closing_time'],
                cost=poi_dict['attributes']['cost_euros']
            )
        else:  # Formato simples
            poi = POI(**poi_dict)
        
        poi_objects.append(poi)
    
    return poi_objects