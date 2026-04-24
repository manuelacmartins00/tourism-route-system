# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This System Does

An intelligent tourism route recommendation system for Portugal. It takes a natural language query (PT or EN), extracts user preferences via LLM, retrieves relevant Points of Interest using RAG (ChromaDB), optimizes the route with one of four algorithms (ACO, GA, PSO, Greedy), and returns an explained route with an interactive map.

## Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Set required API key (Groq — free tier)
export GROQ_API_KEY="your_key"   # or add to .env file

# Interactive CLI
python interactive_cli.py

# REST API (port 7860)
uvicorn api:app --reload --host 0.0.0.0 --port 7860

# Docker
docker build -t tourism-route-system .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key tourism-route-system
```

**One-time setup (RAG index must exist before first use):**
```bash
python scripts/setup_rag.py
```

If ChromaDB errors appear, delete `data/chroma_db/` and rerun setup. There is no automated test suite.

## Architecture Pipeline

Each user query flows through five sequential layers:

1. **LLM Layer** (`src/llm/llm_orchestrator.py`) — Groq API with `llama-3.1-8b-instant` extracts structured preferences: `max_time`, `max_cost`, `preferred_categories`, `location`, `start_time`, `transport_mode`, `mobility_issues`. Returns 422 if required fields are missing (triggers clarification).

2. **RAG Layer** (`src/rag/rag_setup.py`) — ChromaDB with `paraphrase-multilingual-MiniLM-L12-v2` embeddings (384-dim). Semantic search filtered by category, cost ceiling, and geographic bounding box. Returns 25–50 candidate POIs.

3. **Optimization Layer** (`src/optimizers/`) — Algorithms share the same interface and receive candidate POIs + a time/distance matrix. Algorithm is chosen deterministically based on candidate count and time constraint:
   - `TourismACO` — 30 ants, 100 iterations; best diversity
   - `TourismGA` — 50 population, 30 generations; best for large candidate sets
   - `TourismPSOA` — 30 particles, 50 iterations; fast convergence
   - `GreedyPlanner` — deterministic baseline

4. **Evaluation Layer** (`src/optimizers/route_evaluator.py`) — AHP-weighted fitness: time utilization (48.1%), category matching (18.85%), proximity (18.85%), distance efficiency (10.95%), diversity (3.24%). Routes violating hard constraints (time, cost, opening hours) get fitness = 0.

5. **Output Layer** (`src/utils/`) — LLM explanation, Folium map via OSRM routing (`map_generator.py`), day-by-day itinerary clustering into 8-hour days (`day_planner.py`), optional SHAP analysis (`shap_explainer.py`).

**Orchestration** is in `main_system.py` (`TourismRouteSystem` class). `api.py` wraps it in FastAPI; `interactive_cli.py` wraps it in a colorama CLI.

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/query` | Plan a route — body: `{"query": "..."}` |
| `GET` | `/map/{map_id}` | Retrieve generated Folium HTML map |
| `POST` | `/feedback` | Save SUS questionnaire response |
| `GET` | `/admin` | Download feedback CSV (header: `X-Admin-Password`) |
| `GET` | `/health` | Health check |

The `/query` response includes `route`, `explanation`, `map_id`, `day_plan`, `optimization` metrics, and `algorithm_used`.

## Data Files

Large files live in `data/` and are **not committed to git**:
- `portugal_todos_pois_final_enriched.json` — ~15 K POIs (main database)
- `portugal_distances.npy` — pre-computed distance matrix (~1 GB)
- `chroma_db/` — ChromaDB persistent vector index
- `feedback/` — collected SUS responses (CSV-backed)

The transit integration (`src/transit/`) is optional — GTFS data files for Lisbon/Porto metro and buses must be placed in `data/` before `TransitService` will load.

## Deployment

The production app runs on HuggingFace Spaces (`ManuelMartinsTeseISCTE/TourismRouteSystemV1`). **Code changes only take effect after pushing to the HF Space remote** — pushing to GitHub alone is not enough.

```bash
# Push to HuggingFace Space (triggers rebuild)
git push huggingface main

# If push is rejected due to large files (>10MB), migrate them to LFS first:
git lfs migrate import --include="path/to/large/file" --everything --yes
git push huggingface main --force
```

After pushing, the Space rebuilds automatically — wait ~1-2 minutes before testing.

## Key Configuration

- **LLM:** `llama-3.1-8b-instant` via Groq, temperature 0.3, max 600 tokens
- **Algorithm auto-select logic:** in `main_system.py` → `select_algorithm_deterministic()`
- **Day planning defaults:** 8 h/day, 60 min lunch break
- **OSRM profiles:** `foot`, `car`, `bike` (public instance used by default)
- **Admin password** for `/admin`: env var `ADMIN_PASSWORD`, default `"thesis2025"`
- **HuggingFace logging:** enabled when `HF_TOKEN` env var is set (`scripts/log_to_hf.py`)
