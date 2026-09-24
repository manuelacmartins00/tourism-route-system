# Architecture Overview

This document describes how the Tourism Route System is structured and the main design decisions
behind it. It is a high-level overview. Implementation details, parameter values and data
processing specifics are intentionally left out.

---

## 1. The problem

Given a free-text request such as *"3 days in Porto, museums and good food, around 300 euros, on
foot"*, the system must choose **which** places to visit and **in what order**, within the
available time and budget.

This is a variant of the **Orienteering Problem**, known in tourism research as the **Tourist Trip
Design Problem**. Unlike a shortest-path problem, the set of places is not given: the system has to
select a subset and sequence it. The problem is NP-hard, so the system uses approximate
optimisation methods rather than exact solvers.

## 2. Architecture at a glance

The system is a **deterministic pipeline**. A language model sits only at the edges: it interprets
the request at the start and explains the result at the end. All selection and routing decisions
are made by code.

```
Natural-language query (PT / EN)
   │
   ▼
[1] Language understanding (LLM)      → structured preferences
   │
   ▼
[2] Location resolution + retrieval   → candidate Points of Interest
   │
   ▼
[3] Route optimisation                → selected and ordered POIs
   │      ▲
   │      └── [4] Multi-criteria evaluation (scores every candidate route)
   ▼
[5] Day planning · interactive map · natural-language explanation
```

### Why this design

| Option considered | Decision | Main reason |
|---|---|---|
| LLM agent that decides each step with tools | Not adopted | Results must be reproducible and comparable across algorithms; language models are not reliable at satisfying hard constraints such as time, budget and opening hours |
| LLM generates the itinerary directly | Not adopted | No guarantee that places exist or are open, and no measurable quality |
| Classic recommender system | Not adopted | Needs historical user ratings, which do not exist here, and outputs a ranking, not a feasible route |
| Web form instead of natural language | Not adopted | Natural-language interaction is the contribution being evaluated |
| **LLM at the edges + retrieval + optimisation** | **Adopted** | Natural-language interface with verifiable, constraint-respecting routes |

## 3. Data

- About **11,300 Points of Interest** across mainland Portugal, the Azores and Madeira, grouped into
  **43 categories** and **7 tourism regions**.
- The primary source is the official Portuguese tourism POI repository. Some attributes are
  complemented with open geographic data.
- The dataset is a **static, versioned snapshot**. The system does not call external price or
  opening-hours services at query time. This keeps runs reproducible and query latency low.

## 4. Layers

### 4.1 Language understanding

A general-purpose open LLM, served through the Groq API, turns the request into structured
preferences: duration, budget, interests, locations, transport mode, group composition and
accessibility needs. Extraction runs with deterministic settings. If essential information is
missing, the system asks a follow-up question instead of guessing.

### 4.2 Location resolution and retrieval (RAG)

Place names are resolved to geographic areas. Candidate POIs are then retrieved with **hybrid
retrieval**: multilingual semantic search over POI descriptions, combined with structured filters
for category, cost and location. Semantic search captures intent that a plain category filter
cannot, such as "somewhere quiet" or "places linked to maritime history". Numeric constraints are
handled by exact filters, not embeddings.

- Vector store: ChromaDB (local, persistent)
- Embeddings: multilingual Sentence-Transformers model (Portuguese and English)

### 4.3 Route optimisation

Four interchangeable algorithms share the same interface, inputs and evaluator:

| Algorithm | Family | Role |
|---|---|---|
| **Genetic Algorithm** | Evolutionary | **Production algorithm** |
| Ant Colony Optimisation | Swarm intelligence | Comparison |
| Particle Swarm Optimisation | Swarm intelligence | Comparison |
| Greedy | Constructive heuristic | Baseline |

The Genetic Algorithm was selected for production after a large hyperparameter grid search and a
benchmark over several hundred scenarios. It gave the best average route quality, with runtimes
suitable for an interactive service. Exact methods such as integer programming were not used,
because their worst-case runtime is unbounded for a synchronous web request.

### 4.4 Multi-criteria evaluation

Every candidate route gets a single quality score that combines several criteria: use of the
available time, match with the requested interests, geographic coherence, and variety. The
criteria weights were initialised with the **Analytic Hierarchy Process (AHP)**. Routes that break
hard constraints, such as exceeding the time or budget, are treated as infeasible. Context such as
travelling with children or reduced mobility adjusts the score within a limited range.

### 4.5 Output

- **Day planning:** the route is split into days by geographic clustering, then accommodation,
  meals and opening hours are taken into account.
- **Map:** an interactive Folium map with real street geometry from OSRM. Public-transport legs are
  resolved in Lisbon and Porto.
- **Explanation:** the LLM writes a short explanation based on values the system has already
  computed, so it describes the route rather than inventing facts.
- **Follow-up:** after a route is generated, the user can remove places, exclude categories, ask
  questions about the route, or start a new request.

## 5. Evaluation

- **Preference extraction:** field-level accuracy measured on a manually annotated set of queries.
- **Optimisation:** hyperparameter grid search and a multi-profile benchmark comparing the four
  algorithms.
- **Users:** System Usability Scale (SUS) questionnaire linked to each generated route.

## 6. Deployment

- FastAPI service in a Docker container, hosted on HuggingFace Spaces
- Web interface in Portuguese and English
- The design targets free-tier infrastructure (CPU only), which influenced several choices above:
  lightweight embeddings, a bounded optimisation time, and exact routing only after the route is
  fixed

## 7. Known limitations

- Opening hours are considered in day planning, but are not yet a constraint inside the optimiser.
- Public-transport data improves presentation of the route, not the optimisation itself.
- Some POI attributes, such as cost and visit duration, are estimated rather than observed.
