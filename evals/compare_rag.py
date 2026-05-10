"""
Compara os dois indices RAG (v1 e v2) na mesma query.
v1: chroma_db  — embedding inclui custo/duracao/horario
v2: chroma_db2 — embedding apenas nome/categoria/regiao/descricao
Uso: python evals/compare_rag.py "museus de arte em Lisboa"
     python evals/compare_rag.py  (usa query de exemplo)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

DATA_FILE   = "data/portugal_todos_pois_final_enriched.json"
N_RESULTS   = 8

EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

def get_v1(query, n):
    from src.rag.rag_setup import POI_RAG
    rag = POI_RAG(data_file=DATA_FILE)
    r = rag.query(text=query, n_results=n)
    return [(p["name"], p["category"], p["relevance_score"]) for p in r["pois"]]

def get_v2(query, n):
    client = chromadb.PersistentClient(
        path="./data/chroma_db2",
        settings=Settings(anonymized_telemetry=False)
    )
    col = client.get_collection("portugal_pois_v2", embedding_function=EMBED_FN)
    r = col.query(query_texts=[query], n_results=n)
    results = []
    for i in range(len(r["ids"][0])):
        m = r["metadatas"][0][i]
        score = 1.0 - r["distances"][0][i]
        results.append((m["name"], m["category"], round(score, 3)))
    return results

def print_col(title, rows):
    print(f"  {title}")
    print(f"  {'-'*55}")
    for i, (name, cat, score) in enumerate(rows, 1):
        short = name[:35] + "..." if len(name) > 35 else name
        print(f"  {i}. {score:.3f}  {short:<38} ({cat})")

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "museus de arte em Lisboa"
    print(f'\nQuery: "{query}"\n')

    v1 = get_v1(query, N_RESULTS)
    v2 = get_v2(query, N_RESULTS)

    print_col("V1 (chroma_db)  — com custo/duracao/horario no embedding", v1)
    print()
    print_col("V2 (chroma_db2) — so nome/categoria/regiao/descricao", v2)

    # Overlap
    names_v1 = {r[0] for r in v1}
    names_v2 = {r[0] for r in v2}
    common = names_v1 & names_v2
    print(f"\n  Overlap: {len(common)}/{N_RESULTS} POIs em comum")
    if common:
        print(f"  Comuns: {', '.join(list(common)[:4])}{'...' if len(common) > 4 else ''}")
    print()
