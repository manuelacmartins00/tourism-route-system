---
title: Tourism Route System
emoji: 🗺️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Tourism Route System

Intelligent route recommendation system for Portugal. Takes a natural language query (PT or EN), extracts user preferences via LLM, retrieves relevant Points of Interest using RAG, optimises the route with one of four algorithms, and returns an explained itinerary with an interactive map.

---

## Pipeline

```
User query (natural language)
  └─> LLM Layer (Groq / Llama-3.1-8b)    — extract preferences: time, budget, categories, transport, ...
  └─> RAG Layer (ChromaDB)                — retrieve matching Points of Interest
  └─> Route Optimisation                  — ACO / GA / PSO / Greedy
  └─> Explanation Layer (LLM)             — generate human-readable itinerary
  └─> Map Output (Folium / Leaflet)       — interactive HTML map
```

---

## Setup

```bash
pip install -r requirements.txt

# Required API key (Groq — free tier)
export GROQ_API_KEY="your_key"   # or add to .env

# One-time: build the RAG index
python scripts/setup_rag.py

# Interactive CLI
python interactive_cli.py

# REST API (port 7860)
uvicorn api:app --reload --host 0.0.0.0 --port 7860

# Docker
docker build -t tourism-route-system .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key tourism-route-system
```

> If ChromaDB errors appear, delete `data/chroma_db/` and re-run `setup_rag.py`.

---

## Project Structure

```
TourismRouteSystem/
├── src/
│   ├── llm/                LLM orchestrator (preference extraction + explanation)
│   ├── rag/                ChromaDB vector store + retrieval
│   ├── optimisation/       ACO, GA, PSO, Greedy route solvers
│   ├── map/                Interactive map generation
│   └── utils/
├── data/
│   ├── pois/               Points of Interest dataset (Portugal)
│   └── chroma_db/          Vector index (generated locally, gitignored)
├── scripts/
│   └── setup_rag.py        One-time RAG index builder
├── evals/                  Benchmark prompts and evaluation runs
├── api.py                  FastAPI REST API
├── interactive_cli.py      CLI interface
├── main_system.py          Core system entry point
├── Dockerfile
└── requirements.txt
```

---

## Optimisation Algorithms

| Algorithm | Description |
|---|---|
| ACO | Ant Colony Optimisation — pheromone-based path finding |
| GA | Genetic Algorithm — population-based search |
| PSO | Particle Swarm Optimisation — swarm intelligence |
| Greedy | Fast nearest-neighbour baseline |

---

## Tech Stack

- Python 3.10+
- Groq API (Llama-3.1-8b-instant)
- ChromaDB
- FastAPI
- Folium / Leaflet.js
- Docker
