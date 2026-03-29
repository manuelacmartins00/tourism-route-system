# src/utils/distance_calculator.py

import numpy as np
from typing import List, Dict

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula distância em km entre dois pontos (Haversine)
    
    Args:
        lat1, lon1: Coordenadas do ponto 1
        lat2, lon2: Coordenadas do ponto 2
    
    Returns:
        Distância em quilómetros
    """
    R = 6371  # Raio da Terra em km
    
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = (np.sin(dphi/2)**2 + 
         np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def build_distance_matrix(pois: List[Dict]) -> np.ndarray:
    """
    Cria matriz de distâncias NxN entre POIs
    
    Args:
        pois: Lista de POIs com campos 'lat' e 'lon'
    
    Returns:
        Matriz numpy NxN com distâncias
    """
    n = len(pois)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):  # só metade superior
            if 'location' in pois[i]:
                lat1 = pois[i]['location']['lat']
                lon1 = pois[i]['location']['lon']
                lat2 = pois[j]['location']['lat']
                lon2 = pois[j]['location']['lon']
            else:
                lat1 = pois[i]['lat']
                lon1 = pois[i]['lon']
                lat2 = pois[j]['lat']
                lon2 = pois[j]['lon']
            
            d = haversine(lat1, lon1, lat2, lon2)
            matrix[i][j] = d
            matrix[j][i] = d  # simétrico
    
    return matrix