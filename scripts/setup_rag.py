# scripts/setup_rag.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.rag_setup import POI_RAG


def main():
    print("🔍 Inicializando sistema RAG...")

    # Criar RAG (vai indexar automaticamente)
    rag = POI_RAG(data_file="data/portugal_todos_pois_limpos_enriched.json")

    # Teste
    print("\n🧪 Teste de query...")
    result = rag.query(
        text="historic monuments and museums",
        n_results=5
    )

    print(f"\n✓ Encontrados {result['n_results']} POIs:")
    for poi in result['pois']:
        print(f"  - {poi['name']} (relevance: {poi['relevance_score']:.2f})")

    print("\n✅ RAG configurado com sucesso!")


if __name__ == "__main__":
    main()