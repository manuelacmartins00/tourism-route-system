# scripts/setup_rag2.py
# Cria um segundo índice ChromaDB (chroma_db2) usando apenas
# nome + categoria + região + descrição no texto embedded.
# Permite comparar com o índice original (chroma_db) que inclui
# custo/duração/horário no texto. Os metadados são idênticos nos dois.

import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

DATA_FILE   = "data/portugal_todos_pois_final_enriched.json"
CHROMA_PATH = "./data/chroma_db2"
COLLECTION  = "portugal_pois_v2"

def main():
    print(f"🔍 A criar índice RAG v2 em {CHROMA_PATH} ...")
    print(f"   Diferença: sem custo/duração/horário no texto embedded\n")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    try:
        client.delete_collection(COLLECTION)
        print("🔄 Coleção antiga eliminada — a reindexar...")
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=embedding_fn,
        metadata={"description": "POIs turísticos PT — sem campos operacionais no embedding"}
    )

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    pois = data.get("pois", data) if isinstance(data, dict) else data

    documents, metadatas, ids = [], [], []

    for poi in pois:
        # Texto embedded: apenas nome, categoria, região e descrição
        doc = f"""
{poi['name']}

Categoria: {poi.get('source', {}).get('bundle', '')}
Região: {poi.get('source', {}).get('region', '')}

{poi.get('description', '')}
""".strip()

        documents.append(doc)

        # Metadados idênticos ao índice original (para os otimizadores)
        metadatas.append({
            "poi_id":       str(poi['id']),
            "original_id":  str(poi['id']),
            "name":         poi['name'],
            "category":     poi.get('source', {}).get('bundle', 'outros'),
            "region":       poi.get('source', {}).get('region', ''),
            "lat":          float(poi['location']['lat']),
            "lon":          float(poi['location']['lon']),
            "score":        float(poi['attributes'].get('score', 0.5)),
            "duration":     int(poi['attributes'].get('duration_minutes', 60)),
            "cost":         float(poi['attributes'].get('cost_euros', 0.0)),
            "opening_time": poi.get('schedule', {}).get('opening_time', '09:00'),
            "closing_time": poi.get('schedule', {}).get('closing_time', '18:00'),
        })
        ids.append(f"poi_{poi['id']}")

    batch_size = 500
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        print(f"   Indexados {min(i+batch_size, len(documents))}/{len(documents)} POIs...")

    print(f"\n✅ {len(documents)} POIs indexados em {CHROMA_PATH}/{COLLECTION}")
    print("\n🧪 Teste de comparação — query: 'museus de arte em Lisboa'\n")

    # Comparar os dois índices na mesma query
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.rag.rag_setup import POI_RAG

    query_text = "museus de arte em Lisboa"
    n = 5

    print(f"  {'─'*60}")
    print(f"  ÍNDICE ORIGINAL (chroma_db) — com custo/duração/horário")
    print(f"  {'─'*60}")
    rag_original = POI_RAG(data_file=DATA_FILE)
    r1 = rag_original.query(text=query_text, n_results=n)
    for p in r1['pois']:
        print(f"  {p['relevance_score']:.3f}  {p['name']} ({p['category']})")

    print(f"\n  {'─'*60}")
    print(f"  ÍNDICE V2 (chroma_db2) — só nome/categoria/região/descrição")
    print(f"  {'─'*60}")
    r2 = collection.query(query_texts=[query_text], n_results=n)
    for i in range(len(r2['ids'][0])):
        m = r2['metadatas'][0][i]
        score = 1.0 - r2['distances'][0][i]
        print(f"  {score:.3f}  {m['name']} ({m['category']})")

    print(f"\n  Experimenta outras queries editando query_text neste script.")

if __name__ == "__main__":
    main()
