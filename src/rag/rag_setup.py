# src/rag/rag_setup.py (VERSAO CORRIGIDA)

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import json
from typing import List, Dict
import os

class POI_RAG:
    """Sistema RAG para POIs turisticos"""
    
    def __init__(self, data_file: str = "data/pois_structured_for_rag.json"):
        # Inicializar ChromaDB (persistente) -- v2: sem campos operacionais no embedding
        self.client = chromadb.PersistentClient(
            path="./data/chroma_db2",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Embedding function (multilingue - suporta PT e EN)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Carregar colecao existente ou criar nova
        self.collection = self.client.get_or_create_collection(
            name="portugal_pois_v2",
            embedding_function=self.embedding_fn,
            metadata={"description": "POIs turisticos de Portugal -- so nome/categoria/regiao/descricao"}
        )
        if self.collection.count() == 0:
            self._index_data(data_file)
    
    def _index_data(self, data_file: str):
        """Indexa POIs no ChromaDB"""
        
        print(f"A indexar dados de {data_file}...")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pois = data['pois']
        
        documents = []
        metadatas = []
        ids = []
        
        for i, poi in enumerate(pois):
            # Texto embedded: nome, categoria, regiao, descricao e atividades
            # (sem custo/duracao/horario -- evita poluicao semantica)
            activities = poi.get('original_activities', '')
            doc = f"""
{poi['name']}

Categoria: {poi.get('source', {}).get('bundle', '')}
Regiao: {poi.get('source', {}).get('region', '')}

{poi.get('description', '')}
{f"Atividades: {activities}" if activities else ""}
""".strip()
            
            documents.append(doc)
            
            # Metadata estruturada (filtros)
            metadatas.append({
                "poi_id": str(poi['id']),   # <- usa entity_id real
                "original_id": str(poi['id']),
                "name": poi['name'],
                "category": poi.get('source', {}).get('bundle', 'outros'),
                "region": poi.get('source', {}).get('region', ''),
                "lat": float(poi['location']['lat']),
                "lon": float(poi['location']['lon']),
                "score": float(poi['attributes'].get('score', 0.5)),
                "duration": int(poi['attributes'].get('duration_minutes', 60)),
                "cost": float(poi['attributes'].get('cost_euros', 0.0)),
                "opening_time": poi.get('schedule', {}).get('opening_time', '09:00'),
                "closing_time": poi.get('schedule', {}).get('closing_time', '18:00'),
            })
            
            ids.append(f"poi_{poi['id']}")
        
        # Adicionar ao ChromaDB
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
            print(f"   Indexados {min(i+batch_size, len(documents))}/{len(documents)} POIs...")
        
        print(f"{len(documents)} POIs indexados")
    
    def query(self,
          text: str,
          n_results: int = 20,
          category_filter: List[str] = None,
          category_exclude: List[str] = None,
          max_cost: float = None,
          min_score: float = None,
          lat_min: float = None,
          lat_max: float = None,
          lon_min: float = None,
          lon_max: float = None) -> Dict:
        """
        Query semantica com filtros
        
        [OK] CORRIGIDO: Usa $and para multiplos filtros
        """
        
        # Construir lista de condicoes
        filter_conditions = []
        
        if category_filter:
            filter_conditions.append({"category": {"$in": category_filter}})

        if category_exclude:
            filter_conditions.append({"category": {"$nin": category_exclude}})
        
        if max_cost is not None:
            filter_conditions.append({"cost": {"$lte": max_cost}})
        
        if min_score is not None:
            filter_conditions.append({"score": {"$gte": min_score}})
            
        if lat_min is not None:
            filter_conditions.append({"lat": {"$gte": lat_min}})
            filter_conditions.append({"lat": {"$lte": lat_max}})
            filter_conditions.append({"lon": {"$gte": lon_min}})
            filter_conditions.append({"lon": {"$lte": lon_max}})
        
        # [OK] Combinar filtros com $and se houver multiplos
        where_filter = None
        if len(filter_conditions) > 1:
            where_filter = {"$and": filter_conditions}
        elif len(filter_conditions) == 1:
            where_filter = filter_conditions[0]
        
        # Query
        results = self.collection.query(
            query_texts=[text],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        # Processar resultados
        pois = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            
            poi = {
                "id": metadata['poi_id'],
                "name": metadata['name'],
                "lat": metadata['lat'],
                "lon": metadata['lon'],
                "category": metadata['category'],
                "score": metadata['score'],
                "duration": metadata['duration'],
                "cost": metadata['cost'],
                "opening_time": metadata['opening_time'],
                "closing_time": metadata['closing_time'],
                "distance": results['distances'][0][i],
                "relevance_score": 1.0 - results['distances'][0][i]
            }
            
            pois.append(poi)
        
        return {
            "query": text,
            "n_results": len(pois),
            "pois": pois
        }
    
    def reset(self):
        """Reset da base de dados"""
        self.client.delete_collection("portugal_pois_v2")
        print("[OK] Colecao eliminada")