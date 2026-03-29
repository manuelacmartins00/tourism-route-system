# 🗺️ Sistema de Recomendação de Rotas Turísticas

Sistema inteligente que combina **LLMs**, **RAG** e **algoritmos de otimização** para gerar rotas turísticas personalizadas.

## 🎯 Características

- ✅ **Extração automática de preferências** via LLM (Llama 3.1 70B)
- ✅ **RAG semântico** com ChromaDB para recuperar POIs relevantes
- ✅ **3 algoritmos de otimização**: ACO, GA, Greedy
- ✅ **Explicações em linguagem natural** (português/inglês)
- ✅ **Interface CLI interativa** com cores

## 📦 Instalação

### 1. Clonar/Criar estrutura
```bash
mkdir scripts_teste
cd scripts_teste
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Obter API Key (Together AI - GRÁTIS)

1. Vai a: https://api.together.xyz/signup
2. Cria uma conta
3. Copia a API key do dashboard

### 4. Configurar API key
```bash
# Opção A: Variável de ambiente
export TOGETHER_API_KEY="your_key_here"

# Opção B: Ficheiro .env
echo "TOGETHER_API_KEY=your_key_here" > .env
```

### 5. Gerar matriz de distâncias
```bash
python scripts/create_distance_matrix.py
```

## 🚀 Uso

### Modo Interativo (CLI)
```bash
python interactive_cli.py
```

### Modo Script
```python
from main_system import TourismRouteSystem

system = TourismRouteSystem()

result = system.plan_route(
    "quero visitar museus e comer bem, 5 horas, 50 euros"
)

print(result['explanation'])
```

## 📁 Estrutura
```
scripts_teste/
├── data/                       # Dados
│   ├── pois_structured_for_rag.json
│   ├── lisboa_distances.npy
│   └── chroma_db/
├── src/                        # Código fonte
│   ├── rag/
│   ├── llm/
│   ├── optimizers/
│   └── utils/
├── scripts/                    # Scripts auxiliares
├── outputs/                    # Resultados (gerado)
├── main_system.py              # Sistema principal
├── interactive_cli.py          # CLI interativo
└── requirements.txt
```

## 📖 Exemplos de Queries
```
✓ "quero visitar museus e monumentos, tenho 5 horas e 40 euros"
✓ "procuro restaurantes bons e miradouros, 3 horas, 50 euros"
✓ "I want to see historic sites and eat well, 6 hours, 60 euros"
✓ "family-friendly activities with parks, 4 hours, 30 euros"
```

## 🔧 Troubleshooting

### Erro: `TOGETHER_API_KEY não configurada`
- Solução: Configura a API key (ver passo 4 acima)

### Erro: `Matriz de distâncias não encontrada`
- Solução: Executa `python scripts/create_distance_matrix.py`

### Erro: `ChromaDB collection not found`
- Solução: Apaga pasta `data/chroma_db/` e executa novamente

## 📊 Algoritmos

- **ACO**: Ant Colony Optimization (melhor para diversidade)
- **GA**: Genetic Algorithm (melhor para espaços grandes)
- **Greedy**: Baseline rápido (determinístico)

## 👤 Autor

Manuel Martins 