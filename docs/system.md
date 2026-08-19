# TourismRouteSystem — Documentação Técnica Consolidada

> Substitui `architecture.md`, `experimental_decisions.md`, `system_complete_reference.md` e
> `thesis_notas_tecnicas.md` (arquivados). Consolidado e verificado contra o código em 2026-07-25.
> Os quatro documentos anteriores tinham o mesmo objetivo (manter em memória a arquitetura e as
> decisões do sistema) mas divergiam entre si à medida que o código evoluía — este ficheiro é a
> versão única e atual.
>
> Nota de precisão: as secções de arquitetura, camadas, equações principais e parâmetros de
> produção foram todas re-verificadas linha a linha contra o código nesta consolidação. A tabela
> de constantes (§12) e a lista de decisões rejeitadas (§13) vêm da auditoria mais detalhada de
> 12/06/2026 e não foram todas re-confirmadas linha-a-linha — se precisares de citar um número
> exato na tese, confirma primeiro no ficheiro indicado.

---

## 1. Visão Geral

O TourismRouteSystem (TRS) é um sistema de recomendação de rotas turísticas para Portugal.
Recebe uma query em linguagem natural (PT ou EN), extrai preferências via LLM, recupera Pontos
de Interesse (POIs) relevantes via RAG, otimiza a sequência de visita com metaheurísticas, e
devolve uma rota explicada com mapa interativo e itinerário dia-a-dia.

- Query em linguagem natural → rota estruturada (sem formulários)
- Bilingue: Português e Inglês
- 11.395 POIs de Portugal continental, Açores e Madeira
- 4 algoritmos de otimização intercambiáveis (ACO, GA, PSO, Greedy) — GA é o único selecionado automaticamente em produção
- Função fitness AHP com 6 componentes ponderadas + modificador contextual + penalização de elevação opcional
- Planeamento dia-a-dia com alojamento, refeições e vida noturna
- Mapa Folium interativo com rotas reais OSRM
- API REST (FastAPI) + CLI interativo

---

## 2. Pipeline de 5 Camadas

```
INPUT
  interactive_cli.py ──┐
  api.py (POST /query) ┴──► TourismRouteSystem.plan_route()  [main_system.py]
                                │
        ┌───────────────────────▼───────────────────────┐
        │ CAMADA 1 — LLM (extração de preferências)      │
        │ src/llm/llm_orchestrator.py                     │
        │ Groq · openai/gpt-oss-20b                        │
        └───────────────────────┬───────────────────────┘
        ┌───────────────────────▼───────────────────────┐
        │ CAMADA 2 — RAG (recuperação de POIs)            │
        │ src/rag/rag_setup.py                            │
        │ ChromaDB v2 · MiniLM-L12-v2 (384-dim)          │
        └───────────────────────┬───────────────────────┘
        ┌───────────────────────▼───────────────────────┐
        │ CAMADA 3 — Otimização (ordenação da rota)       │
        │ src/optimizers/  — ACO · GA · PSO · Greedy      │
        └───────────────────────┬───────────────────────┘  (avaliação contínua)
        ┌───────────────────────▼───────────────────────┐
        │ CAMADA 4 — Avaliação (fitness AHP)              │
        │ src/optimizers/route_evaluator.py               │
        └───────────────────────┬───────────────────────┘
        ┌───────────────────────▼───────────────────────┐
        │ CAMADA 5 — Output                                │
        │ explicação LLM · mapa Folium/OSRM · day_planner │
        │ · SHAP (opcional)                                │
        └─────────────────────────────────────────────────┘
```

---

## 3. Estrutura do Repositório (após limpeza de 2026-07-25)

```
TourismRouteSystem/
├── main_system.py              # TourismRouteSystem — orquestra o pipeline completo
├── api.py                      # FastAPI REST API
├── interactive_cli.py          # CLI interativo (colorama)
├── src/
│   ├── llm/llm_orchestrator.py         # LlamaOrchestrator + UserPreferences
│   ├── rag/rag_setup.py                # POI_RAG (ChromaDB v2)
│   ├── optimizers/
│   │   ├── route_evaluator.py          # RouteEvaluator — fitness AHP partilhada
│   │   ├── tourism_aco.py / tourism_ga.py / tourism_psoa.py / greedy_planner.py
│   ├── utils/
│   │   ├── location_resolver.py, day_planner.py, map_generator.py,
│   │   │   shap_explainer.py, data_loader.py, distance_calculator.py,
│   │   │   metrics_evaluator.py (usado só pelo CLI)
│   └── transit/
│       ├── gtfs_loader.py, calendar_resolver.py, transit_service.py
├── scripts/                    # scripts de dados/setup pontuais
│   ├── setup_rag.py                    # indexação inicial do ChromaDB (o correto — ver §4)
│   ├── download_gtfs.py, extract_tp_tabs_data.py, check_poi_descriptions.py,
│   │   reintegrate_missing_pois.py, dedupe_final_pois.py, find_duplicate_poi_names.py,
│   │   log_to_hf.py
├── evals/                      # avaliação/benchmark da tese
│   ├── run_extraction_eval.py + extraction_cases.json (100 casos) + extraction_results.json
│   ├── build_fixtures_490.py, fix_budget_prompts.py, prompts_490_v2.txt, fixtures_490.json
│   ├── run_benchmark.py (motor com LLM), run_benchmark_fixtures.py (motor sem LLM)
│   ├── run_grid_search.py, run_algo_benchmark.py
│   ├── analyse_gridsearch.R
│   ├── audit_dataset.py
│   └── outputs/ (gs_aco.csv, gs_ga.csv, gs_pso.csv, gs_analysis/, bench_analysis/)
├── data/                        # não versionado em git (exceto o JSON principal, via Git LFS)
│   ├── portugal_todos_pois_final_enriched.json   # ~11.4K POIs
│   ├── chroma_db2/                                # índice ChromaDB (reconstruído a frio)
│   ├── gtfs/                                      # feeds Lisboa/Porto metro+bus
│   └── feedback/                                  # respostas SUS
├── outputs/                     # mapas gerados, resultados de benchmark (local, não git)
└── docs/
    └── system.md                # este ficheiro
```

---

## 4. Camada 1 — LLM (`src/llm/llm_orchestrator.py`)

**Modelo:** `openai/gpt-oss-20b` via Groq API.

### 4.1 Temperaturas e modos

| Função | Temperatura | Max tokens | Notas |
|---|---|---|---|
| `extract_preferences` | 0.0 | 600 (ou ~400 em modo compacto) | Determinístico; modo compacto ativa com `BENCHMARK_MODE=1` ou `compact=True` |
| `explain_route` | 0.7 | 300 | Explicação natural, variada |
| `_resolve_location` (fallback) | 0.0 | 15 | Restrito a lista fechada de regiões canónicas |
| `interpret_refinement` | 0.1 | 200 | Classifica follow-up: `remove`/`filter_category`/`fresh_query` |

Retry de rate-limit (até 8 tentativas, backoff lendo `"try again in Xs"`) só ativo em
`BENCHMARK_MODE=1`; em produção (API/CLI) é fail-fast em 429.

### 4.2 `UserPreferences` (campos principais)

`max_time`, `max_cost`, `budget_type` (`person`/`group`), `preferred_categories`,
`transport_mode` (`foot`/`car`/`bike`/`public_transport`), `start_time`, `start_date`,
`num_people`, `num_rooms` (default `ceil(num_people/2)`), `locations` (até 4),
`locations_ordered`, `mobility_issues`, `has_children`, `is_elderly`, `include_accommodation`,
`include_meals`, `has_nightlife`/`nightlife_suggested`, `route_direction` (`N2S`/`S2N`/null).

- `has_children`: deteção silenciosa via keywords PT/EN sem acentos (`filho, crianca, kids, bebe, ...`) **ou** sinal do LLM — nunca perguntado ao utilizador.
- `is_elderly`: deteção silenciosa **só** por keywords (`idoso, avo, terceira idade, senior, reformado, elderly, grandparent, ...`) — ao contrário de `has_children`, não tem sinal correspondente do lado do LLM. Confirmado implementado (`is_elderly` usado em `llm_orchestrator.py` e `route_evaluator.py`).
- `nightlife_suggested`: heurística pós-hoc — sem crianças, `num_people≥2`, `max_time≥960min` (~2 dias) → sugere 1 noite de vida noturna, mesmo sem pedido explícito.

### 4.3 Precisão de extração (validado, 100 casos — `evals/extraction_cases.json`)

- Global: 92,4% de campos corretos (342/370); `missing_fields` correto em 89/100 casos.
- 100% em `start_time`, `last_day_end_time`, `has_children`, `mobility_issues`.
- Ponto mais fraco: `locations_ordered` (ordem explícita de visita em query multi-cidade) — 62,0% (31/50).

---

## 5. Camada 2 — RAG (`src/rag/rag_setup.py`)

- **Vector store:** ChromaDB persistente, `./data/chroma_db2`, coleção `portugal_pois_v2` (confirmado no código — `rag_setup.py:16,30`).
- **Embedding:** `paraphrase-multilingual-MiniLM-L12-v2`, 384-dim.
- **Texto embebido:** nome + categoria/bundle + região + descrição + atividades (quando existem). Custo/duração/horário são **excluídos** deliberadamente do texto (evita poluição semântica).
- **Setup:** `scripts/setup_rag.py` é o script correto — chama a classe `POI_RAG`, que já indexa no formato "v2" acima. (`scripts/setup_rag2.py` reimplementava manualmente a mesma lógica antes de ela ser absorvida na classe; redundante hoje, mantido de lado até confirmação final.)
- **Auto-reconstrução:** `POI_RAG.__init__` só indexa se `collection.count()==0`; como `data/chroma_db2/` não vai para git, o HF Space reconstrói o índice do zero a cada cold start a partir do JSON (Git LFS) — por isso nunca é preciso enviar o índice manualmente.

### 5.1 Fluxo de retrieval (`main_system.py`)

1. Query semântica principal: `n_results=40`, filtros de categoria/custo/bbox.
2. Query suplementar semântica (+20, diversidade).
3. Rebalanceamento por categoria: garante `max(3, 25 // n_categorias)` candidatos por categoria preferida, com re-query dirigida se insuficiente.
4. Fallback de pool insuficiente: se tempo total estimado dos candidatos < 60% de `max_time`, re-query sem filtro de categoria, `n_results=80`.
5. Filtro geográfico circular pós-RAG (a bbox do RAG é retangular; aplica-se um filtro Haversine pelo raio resolvido).
6. `NEVER_INCLUDE_CATEGORIES` — reaplicado defensivamente em Python (`eventos, postos_de_turismo, agencias_de_viagem, localidade, servicos_de_turismo, outros, rentacar`).
7. Alojamento e restaurantes: queries RAG separadas ([Meals-Pre] garante `total_days × 2` restaurantes; alojamento com custo mínimo 60€).

---

## 6. Camada 3 — Otimização (`src/optimizers/`)

### 6.1 Seleção do algoritmo

`select_algorithm_deterministic(n_candidates, max_time)` em `src/llm/llm_orchestrator.py:1231`
**devolve sempre `"GA"`** (verificado no código). Razão documentada: GA venceu 9/16 queries num
benchmark inicial com menor tempo de execução; o fallback PSO→GA duplicava o tempo sem ganho de
qualidade. A assinatura da função é mantida por compatibilidade, mas os argumentos são ignorados.
Override manual: parâmetro `force_algorithm` em `plan_route()`.

### 6.2 Parâmetros — defaults de classe vs. produção (verificado, `main_system.py:928-941`)

| Algoritmo | Defaults da classe | **Parâmetros usados em produção** |
|---|---|---|
| ACO | `n_ants=30, n_iterations=100, α=1.0, β=2.0, evap=0.5, q0=0.9` | iguais aos defaults — `force_algorithm="ACO"` |
| GA | `population_size=50, n_generations=30, cx=0.7, mut=0.2` | **`population_size=100, n_generations=50, crossover_prob=0.6, mutation_prob=0.1, mutation_dynamic=True, mutation_patience=5`** (config `GA_DYN_046`, vencedora do grid search — é a que corre sempre, dado §6.1) |
| PSO | `n_particles=30, n_iterations=50, w=0.7, c1=c2=1.5` | **`n_particles=40, n_iterations=50, w=0.7, c1=c2=0.5, w_decay=False`** (config `PSO_043`, vencedora do grid search PSO 54 configs, 2026-07-25 — atualizado de `n_particles=20, n_iterations=30`) — `force_algorithm="PSO"`, com fallback automático para GA se estagnar |
| Greedy | sem hiperparâmetros | `force_algorithm="GREEDY"` — baseline determinístico |

**Nota importante:** os defaults de classe (usados se instanciares os otimizadores diretamente,
fora do `main_system.py`) **não são** os parâmetros de produção — só a chamada em
`main_system.py:928-941` usa os valores corretos. Documentos anteriores confundiam os dois.

### 6.3 Mecanismos por algoritmo

- **GA** — representação por permutação; crossover OX (Ordered Crossover, Davis 1985); mutação por swap (nunca a posição 0), revertida se tornar a rota inviável; seleção por torneio (tamanho 3). Mutação dinâmica: após `mutation_patience` gerações sem melhoria, `mutation_prob` sobe para 0.9 (hipermutação binária e permanente, Cobb 1990 / Şehab & Turan 2024).
- **ACO** — adaptado ao Orienteering Problem; `prob(j) = τ^α · η^β · (score_j+0.5)`; com prob. `q0=0.9` escolhe argmax (exploitation), senão amostragem proporcional; feromona clipada [0.01, 10.0].
- **PSO** — partícula = sequência de POIs; velocidade = lista de operações {swap, insert, remove}; fallback automático para GA se `best_history[-1] ≤ best_history[0] × 1.005` (melhoria total <0.5%).
- **Greedy** — 3 estratégias (`score`/`nearest`/`hybrid`, default `hybrid = score/(dist+0.001)`), construção gulosa até esgotar candidatos viáveis.

### 6.4 Matriz de distâncias/tempos

Haversine + tabela estática por modo (não OSRM na otimização — evitaria timeout no HF Space):

| Modo | Referências |
|---|---|
| `foot` | 1km→12min, 2km→25min, 5km→60min |
| `car`/`fastest` | 2km→4min, 5km→8min, 15km→15min, 50km→38min |
| `public_transport` | 1km→10min, 5km→30min, 15km→47min, 50km→94min |
| `bike` (só day planner) | 2km→8min, 5km→18min, 15km→50min |

Se a distância entre dois POIs consecutivos é <500m, usa-se sempre "a pé", independentemente do modo declarado.

---

## 7. Camada 4 — Avaliação (`src/optimizers/route_evaluator.py`)

### 7.1 Função fitness (pesos verificados em `route_evaluator.py:36-45`)

```
fitness_bruto = w_time·time_efficiency + w_proximity·proximity_component
              + w_cat_indata·cat_indata_comp + w_distance·distance_penalty
              + w_diversity·diversity_component + w_cat_general·cat_general_comp
fitness = min(100, fitness_bruto × contextual_modifier)
```

| Componente | Peso | Interpretação |
|---|---|---|
| `w_time` | 0.3934 | Utilização do tempo disponível |
| `w_proximity` | 0.1885 | POIs dentro da área geográfica pedida |
| `w_cat_indata` | 0.1686 | O otimizador usou os POIs preferidos disponíveis no pool? |
| `w_distance` | 0.1095 | Penaliza POIs fora da área declarada |
| `w_diversity` | 0.0800 | Variedade de categorias na rota |
| `w_cat_general` | 0.0600 | Categorias pedidas representadas (dados + modelo) |

AHP 5×5, Consistency Ratio CR=0,0081. Pesos derivados da literatura do Orienteering Problem
(Vansteenwegen et al. 2011; Gunawan et al. 2016), que define o "profit" do POI (~`cat_indata`)
como objetivo primário e a diversidade como critério secundário.

### 7.2 Definições

```
time_efficiency = util × 0.55  se util<50%,  senão = util      (util = min(100, tempo_usado/max_time×100))
proximity_component = média(ps(ratio_i))×100     ps(r)=1 se r≤1, senão max(0,1-(r-1)²)
cat_indata_comp  = min(n_preferidos_na_rota / max_achievable, 1.0) × 100
distance_penalty = média(dp(ratio_i))×100        dp(r)=1 se r≤1, senão max(0,2-r)
diversity_component = min(unique_categories / cap, 1.0)×100   cap = max(1, min(n_local_cats, max(6, min(n_preferidos,8))))
cat_general_comp = |categorias_pedidas ∩ categorias_rota| / |categorias_pedidas| × 100
```

`geo_ratio(poi)` = distância Haversine mínima a qualquer cidade-alvo, normalizada pelo raio
resolvido para essa cidade (0 = centro, 1 = fronteira, >1 = fora — penalizado).

### 7.3 Hard constraints (fitness = 0 se violado)

- Tempo total > `max_time`
- Custo total > `max_cost` (alojamento dividido por `num_people/num_rooms`)
- `include_accommodation=True` mas nenhum hotel em itinerários ≥2 dias
- POIs noturnos > `6 × noites_disponíveis`

### 7.4 Modificador contextual (multiplicativo, clamp **[0.8, 1.2]**)

Ativo quando `has_children`, `mobility_issues`, ou `is_elderly`:

| Contexto | Penaliza (−0.15/POI) | Bonifica (+0.10/POI) |
|---|---|---|
| `has_children` | bares_e_discotecas, casinos, turismo_activo | espacos_verdes, parques_e_reservas, parques_de_diversao, zoos_e_aquarios, ciencia_e_conhecimento |
| `mobility_issues` | turismo_activo, campos, parques_e_reservas, parques_de_diversao, grutas | restaurantes_e_cafes, monumentos, museus_e_palacios, espacos_verdes, termas, ciencia_e_conhecimento, talassoterapia |
| `is_elderly` | turismo_activo, bares_e_discotecas, campos | museus_e_palacios, monumentos, termas, espacos_verdes, talassoterapia, ciencia_e_conhecimento, restaurantes_e_cafes |

(`casinos` foi removido do conjunto de penalização de `is_elderly` por decisão explícita — não são considerados inapropriados para idosos.)

### 7.5 Penalização de elevação (opcional, `w_elevation=0.15`, os outros 6 pesos reescalados)

Ativa quando `mobility_issues` **ou** `is_elderly`. Calculada só pós-otimização, limitada aos
primeiros 20 POIs da rota escolhida (via OpenTopoData SRTM30m — evita milhares de chamadas HTTP
pré-otimização). Score 100 até 50m de ganho acumulado, decai linearmente até 0 aos 200m.

---

## 8. Camada 5 — Output

### 8.1 Planeamento diário (`src/utils/day_planner.py`)

1. **Clustering:** K-means por dia; sementes geográficas interpoladas entre cidades quando há ≥2 cidades-foco (`_seed_centroids`), não `random_state` puro.
2. **Reparação de cobertura:** garante ≥1 POI por cidade-foco num raio de 20km.
3. **Alojamento:** mais próximo do centróide do dia; nunca volta a um hotel abandonado; mantém o hotel de ontem se <30km do melhor de hoje; sem hotel no dia de partida.
4. **Restaurantes:** ≤2/dia por proximidade; se o mais próximo está a >35km, reutiliza deliberadamente um restaurante já usado noutro dia (decisão consciente: preferível a um restaurante isolado distante).
5. **Balanceamento de tempo:** redistribui POIs entre dias sobrecarregados (>360min) e subcarregados (<216min), com limite de deslocação `move_cap_km = max(15, max_radius/n_dias)`.
6. **Fill-D:** se um dia tem <360min de atividades, RAG dirigido ao centróide do dia; se as categorias preferidas já estão esgotadas, recorre a categorias-irmãs do mesmo grupo temático.
7. **Vida noturna:** janela 21:00–03:00; nunca nas últimas 2 noites; preferência por 6ª/sábado quando a data é conhecida; ≤3 bares/noite (60min/bar em modo pub crawl).

### 8.2 Mapa (`src/utils/map_generator.py`)

Folium + OSRM (`foot`/`car`/`bike`) para a rota real; cores por dia; a linha desenha-se pela
ordem dia-a-dia (nearest-neighbour), não pela ordem bruta do otimizador — evita zigzags A-B-A no
mapa renderizado.

### 8.3 Explicação e interpretabilidade

`explain_route()` gera texto PT/EN alimentado pela rota, fitness, top-3 componentes SHAP e
resumo do day plan. `shap_explainer.py` usa `KernelExplainer` ao nível do POI (custo, duração,
diversidade de bundle, proximidade geográfica) para fundamentar a explicação em evidência
computada em vez de afirmações generativas não verificáveis.

---

## 9. Transit / GTFS (`src/transit/`)

5 operadores integrados (Metro Lisboa, Metro Porto, STCP, Carris Metropolitana, CP), grafo
NetworkX unificado (~15.8K nós, 577K arestas, ~31MB, cache). **Importante:** o cálculo de custo
via GTFS (`build_cost_matrix`, Dijkstra) está implementado mas **desativado**
(`_use_transit=False`, hardcoded) — a otimização nunca usa dados GTFS reais para decidir quais
POIs visitar ou por que ordem, só a tabela estática Haversine (§6.4). O GTFS é usado só depois da
rota estar decidida: para decompor um troço em explicação (1ª/última milha + perna de transporte)
e para desenhar os segmentos reais no mapa.

---

## 10. API REST (`api.py`) e CLI

| Método | Path | Propósito |
|---|---|---|
| `POST` | `/query` | Planear rota — `{"query": "...", "session_id"?, ...}` |
| `GET` | `/map/{map_id}` | Mapa Folium HTML |
| `POST` | `/feedback` | Guardar resposta SUS |
| `GET` | `/admin` | CSV de feedback (header `X-Admin-Password`, default `"thesis2025"`) |
| `GET` | `/health` | Health check (GET+HEAD, para UptimeRobot) |

`interactive_cli.py`: menu com nova rota, comparação dos 4 algoritmos na mesma query, histórico,
dashboard de métricas (`MetricsEvaluator`, exclusivo do CLI).

`BENCHMARK_MODE=1`: prompts compactos + retry de rate-limit; produção é fail-fast sem retry.

---

## 11. Resultados Validados

### 11.1 Grid search de hiperparâmetros (108.810 runs)

186 configurações (54 ACO, 54 GA estático, 54 GA dinâmico, 24 PSO) × 195 cenários × 3 seeds.
Cenários cobrem todo o continente + Açores + Madeira, incluindo 70 cenários multi-cidade.
Executado num servidor Hetzner dedicado (~€3,95 total, ~35h).

**Melhor config por família** (média sobre 585 runs):

| | ACO_036 | GA_050 (estático) | **GA_DYN_046 (produção)** | PSO_020 |
|---|---|---|---|---|
| Hiperparâmetros | ants=50, α=1.0, β=0.5, ρ=0.1 | pop=100, gen=50, cx=0.75, mut=0.20 | pop=100, gen=50, cx=0.60, mut=0.10, patience=5 | particles=40, w=0.9 (decay), c1=c2=0.5 |
| Fitness (média) | 81,35 | 82,97 | **82,97** | 81,96 |

- GA dinâmico venceu o GA estático em 90,7% de 54 configs pareadas (Wilcoxon p<0,0001).
- Tempo de execução (mediana): PSO 0,16s < GA_estático 0,20s < GA_DYN 0,21s « ACO 1,93s (~12× mais lento, sem vantagem de qualidade correspondente).
- Config de produção do ACO (n_ants=30, β=2.0) ficou em 27º/54; PSO de produção (particles=20, w=0.7) ficou em 19º/24 no grid original — ambos longe do ótimo do grid search, mas continuam a ser os únicos alcançáveis via `force_algorithm` (GA é sempre o escolhido automaticamente, §6.1).

### 11.1.1 Extensão do grid search do PSO — 24→54 configs (2026-07-25)

O grid original do PSO tinha só 24 configs (contra 54 dos outros três algoritmos) e nunca variava
`n_iterations` (fixo em 50) — limitação já documentada. Corrido um grid PSO expandido em paridade
com ACO/GA: `n_particles{20,30,40} × w{0.4,0.7,0.9} × c1=c2{0.5,1.0,1.5} × n_iterations{30,50} = 54
configs × 195 fixtures × 3 seeds = 31.590 runs` (`evals/outputs/gs_pso_54.csv`, 0 erros). Resultados
do grid original (`gs_pso.csv`, 24 configs, 14.040 runs) preservados à parte, não sobrescritos.

**Melhor config do grid expandido:** `PSO_043` — `n_particles=40, w=0.7, c1=c2=0.5, n_iterations=50,
w_decay=False` — fitness média 81,91 (sd=27,09) (muito próxima do `PSO_020` do grid original, 81,96 —
mesma vizinhança de hiperparâmetros, resultado replicado independentemente). **`main_system.py` e
`evals/run_algo_benchmark.py` foram atualizados em 2026-07-25 para usar `PSO_043` como config de
produção** (substituindo a anterior).

**A config de produção anterior (`n_particles=20, n_iterations=30, w=0.7, c1=c2=1.5`, `PSO_010` no
grid expandido) ficava em último lugar, 54º/54** (fitness média 81,45, sd=27,07 — uma diferença
pequena em termos absolutos face à melhor, 0,46 pontos / 0,56%, mas era a pior combinação testada).
Em compensação era também a **mais rápida das 54** (mediana 0,118s vs. 0,467s da nova produção,
`PSO_043`) — a troca ganha ~0,46 pontos de fitness ao custo de ~4× mais tempo de execução do PSO
(o algoritmo continua a ser mais rápido que ACO em qualquer dos casos, e só é alcançado via
`force_algorithm="PSO"` ou fallback de estagnação, nunca por seleção automática — §6.1).

**Sensibilidade por parâmetro** (média de fitness por valor, 18-27 configs por grupo):

| Parâmetro | Valores (fitness média) | Conclusão |
|---|---|---|
| `n_particles` | 20→81,46, 30→81,74, 40→81,88 | dominante, efeito monótono — mais partículas ajuda sempre |
| `w` (inércia) | 0,4→81,69, 0,7→81,70, 0,9→81,70 | insensível |
| `c1=c2` | 0,5→81,71, 1,0→81,69, 1,5→81,68 | quase insensível, 0,5 marginalmente melhor |
| `n_iterations` | 30→81,69, 50→81,70 | **insensível — resolve a limitação antes documentada**: mais iterações não ajuda neste domínio |

### 11.2 Benchmark de 490 prompts

7 perfis (natureza/aventura, cultura/história, gastronomia, praia, compras/urbano, família,
bem-estar), 1–21 dias, 1–10 pessoas, cobrindo as 24 capitais de distrito + principais cidades
insulares.

- 483/490 (98,6%) rotas válidas; 7/490 pediram clarificação; 0 erros de sistema.
- Fitness médio: 54,6 (proximidade e eficiência de distância perto do teto ~99,5%; utilização do
  tempo 40,8% — maior margem de melhoria e componente com maior peso).
- Comparação dos 4 algoritmos no mesmo corpus (488 cenários válidos): **GA 83,09** > PSO ~82,0 >
  ACO ~80,0 > Greedy 77,08 — confirma o ranking do grid search, agora incluindo o Greedy.

### 11.3 Precisão de extração LLM (100 casos) — ver §4.3

### 11.4 Questionário SUS (instrumento, resultados por reportar)

`feedback.html`, 18 perguntas / 4 blocos: P1–P10 SUS padrão (PT, Likert 5 pontos), P11–P15
qualidade da rota, P16 feedback aberto, P17–P18 demografia opcional. Cada rota tem `run_id`
ligado ao feedback no dataset `TourismRunsLog` (HuggingFace).

---

## 12. Constantes de Configuração (auditoria 12/06/2026 — confirmar linha exata antes de citar)

| Constante | Valor |
|---|---|
| `max_radius_km` | `30 + total_days × 10` (1d→40km, 7d→100km, 14d→170km) |
| `move_cap_km` | `max(15, max_radius/total_days)` |
| `CORRIDOR_BUFFER_DEG` / `CORRIDOR_KM` | 0,45° (~50km) / 25,0km |
| `ACCOMMODATION_MIN_COST` | 60,0€ |
| `MEALS_PER_DAY` / `DEFAULT_MEAL_COST` | 2 / 12,0€ |
| `meal_budget_reserve` cap | ≤50% de `max_cost` |
| Pool RAG inicial | `n_results=40` (principal) + 20 (suplementar) |
| `min_per_cat` (rebalance) | `max(3, 25 // n_categorias)` |
| Threshold fallback pool insuficiente | tempo candidatos < 60% de `max_time` |
| `hours_per_day` / `lunch_break` | 8h / 60min |
| `NIGHT_WINDOW` / `PUB_CRAWL_DURATION` | 21:00–03:00 / 60min |
| `MAX_NOCTURNAL_PER_DAY` | 3 |
| `MAX_DIST_KM` (restaurantes) | 35,0km |
| `_balance_time` target/DST_HIGH/DST_LOW | 360min / 360min / 216min (60%) |
| `_repair_city_coverage` raio | 20,0km |
| `max_time` cap | 20.160min (42 dias) |
| `nightlife_suggested` threshold | `num_people≥2`, `max_time≥960min`, sem crianças |
| `RADIUS_MUNICIPIO` / `RADIUS_NOMINATIM` | 25,0 / 50,0km |
| `_COUNTRY_REGION_OVERRIDES` | Portugal=350km, Algarve=120km, Alentejo=150km, Açores=200km, ... |
| Nominatim throttle | ≥1,1s entre chamadas |

---

## 13. Decisões Rejeitadas/Revertidas (contexto para a tese)

- **Dedup de restaurante no mesmo dia** — implementado e revertido: "dois restaurantes idênticos no mesmo dia é preferível a um vindo de longe — limitação dos dados na região, não um bug."
- **Cap rígido de 40% de diversidade por categoria** — implementado como hard constraint e revertido no mesmo dia (tornava rotas infeasible em corredores mono-categoria); substituído por penalização suave via `w_diversity`.
- **`max_same_cat`/`default_cap` = 3** — testado e revertido de volta a 2 no mesmo dia.
- **`w_score`** (componente de score do POI no fitness) — removido; o score do POI influencia agora só indiretamente via ranking RAG.
- **Matriz de tempos via GTFS/OSRM real (Dijkstra N²)** — abandonada para o otimizador (timeout no HF Space); mantida só para visualização (§9).
- **Cálculo de elevação pré-otimização para todos os candidatos** — abandonado (milhares de chamadas HTTP); movido para pós-otimização, só top-20 POIs da rota final.
- **Cap artificial de 10 POIs por rota (GA/PSO)** — removido, era uma limitação herdada sem justificação atual.

---

## 14. Deployment

**Produção:** HuggingFace Spaces (`ManuelMartinsTeseISCTE/TourismRouteSystemV1`), Docker + FastAPI, porta 7860.

```bash
git push huggingface main     # aciona rebuild ~1-2min
git lfs migrate import --include="path/to/large/file" --everything --yes   # ficheiros >10MB antes do push
```

- `data/portugal_todos_pois_final_enriched.json` é a única exceção rastreada em `data/` (via Git LFS).
- `data/chroma_db2/` reconstrói-se do zero em cada cold start (§5) — cold start do índice completo demora 20-30min em CPU.
- Free tier HF Spaces suspende após 48h de inatividade; UptimeRobot faz ping a `/health` a cada 5min.
- Variáveis de ambiente: `GROQ_API_KEY` (obrigatória), `ADMIN_PASSWORD` (default `"thesis2025"`), `HF_TOKEN` (opcional, logging), `BENCHMARK_MODE` (default `0`).

---

## 15. Ficheiros-chave para citação na tese

- `main_system.py` — orquestração, geo-filtering, TSP multi-localização, Fill/Fill2/Fill3, budget de refeições, vida noturna, THEMATIC_GROUPS.
- `src/optimizers/route_evaluator.py` — fitness AHP, hard constraints, modificador contextual, penalização de elevação.
- `src/optimizers/tourism_aco.py`, `tourism_ga.py`, `tourism_psoa.py`, `greedy_planner.py`.
- `src/llm/llm_orchestrator.py` — extração de preferências, `select_algorithm_deterministic`.
- `src/utils/day_planner.py` — K-means com sementes, hotéis, restaurantes, bares.
- `src/utils/location_resolver.py` — overrides regionais, GeoJSON, Nominatim.
- `src/rag/rag_setup.py` — ChromaDB v2, embeddings, filtros.
- `evals/run_grid_search.py` + `evals/analyse_gridsearch.R` — grid search e análise estatística.
- `evals/run_benchmark_fixtures.py` + `evals/fixtures_490.json` — benchmark de 490 prompts sem LLM.
- `evals/run_extraction_eval.py` + `evals/extraction_cases.json` — precisão de extração LLM (100 casos).
