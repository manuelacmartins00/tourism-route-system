# Relatorio Completo do Dataset de POIs

## 1. Visao Geral

| Metrica | Valor |
|---------|-------|
| Total de POIs | 11.395 |
| Fonte original | Portal Turismo de Portugal (visitportugal.com) |
| Enriquecimento | OSM (OpenStreetMap) + Google Places API |
| Ficheiro activo | portugal_todos_pois_final_enriched.json |

---

## 2. Distribuicao por Bundle (Categoria)

| Bundle | N | % | Custo medio |
|--------|---|---|-------------|
| restaurantes_e_cafes | 2512 | 22.0% | EUR22.9 |
| hotelaria | 1483 | 13.0% | EUR6.1 (*) |
| turismo_espaco_rural | 905 | 7.9% | EUR5.6 (*) |
| monumentos | 862 | 7.6% | EUR5.0 (*) |
| agencias_de_viagem | 804 | 7.1% | EUR5.0 |
| turismo_activo | 630 | 5.5% | EUR25.0 |
| rentacar | 458 | 4.0% | EUR5.0 |
| servicos_de_turismo | 423 | 3.7% | EUR5.0 |
| bares_e_discotecas | 404 | 3.5% | EUR14.2 |
| postos_de_turismo | 385 | 3.4% | EUR4.9 |
| praias | 364 | 3.2% | EUR0.1 |
| museus_e_palacios | 302 | 2.7% | EUR8.0 |
| eventos | 271 | 2.4% | EUR10.1 |
| turismo_habitacao | 244 | 2.1% | EUR5.6 (*) |
| localidade | 238 | 2.1% | EUR5.0 |
| apartamento_turistico | 190 | 1.7% | EUR5.4 (*) |
| parques_de_campismo | 172 | 1.5% | EUR4.9 (*) |
| alojamento_local | 106 | 0.9% | EUR6.0 (*) |
| campos (golf) | 89 | 0.8% | EUR15.0 |
| arqueologia | 52 | 0.5% | EUR5.0 |
| pousadas_da_juventude | 50 | 0.4% | EUR5.0 (*) |
| aldeamento_turistico | 48 | 0.4% | EUR6.4 (*) |
| espacos_verdes | 46 | 0.4% | EUR0.1 |
| marinas_e_portos | 42 | 0.4% | EUR5.0 |
| termas | 39 | 0.3% | EUR35.0 |
| parques_e_reservas | 37 | 0.3% | EUR0.0 |
| parques_de_diversao | 29 | 0.3% | EUR20.0 |
| zoos_e_aquarios | 14 | 0.1% | EUR12.0 |
| casinos | 11 | 0.1% | EUR20.0 |
| talassoterapia | 11 | 0.1% | EUR40.0 |
| grutas | 10 | 0.1% | EUR8.0 |
| ciencia_e_conhecimento | 12 | 0.1% | EUR8.0 |
| academias | 8 | 0.1% | EUR20.0 |
| sugestoes | 26 | 0.2% | EUR5.0 |
| organismos_e_associacoes | 26 | 0.2% | EUR5.0 |
| outros | 64 | 0.6% | EUR5.1 |

(*) Custo medio inflacionado por valor default de EUR5.0

---

## 3. Distribuicao Geografica

| Regiao | N | % |
|--------|---|---|
| Porto e Norte | 3025 | 26.5% |
| Lisboa e Regiao | 2319 | 20.4% |
| Centro de Portugal | 2240 | 19.7% |
| Algarve | 1321 | 11.6% |
| Alentejo | 1259 | 11.0% |
| Acores | 640 | 5.6% |
| Madeira | 591 | 5.2% |

---

## 4. Proveniencia e Completude dos Dados

### 4.1 Descricao
- Com descricao: 8.250 / 11.395 (72.4%)
- Sem descricao: 3.145 / 11.395 (27.6%)
- Das com descricao:
  - Descricao geral (sem horario/custo): 55.9%
  - Horario + descricao: 14.0%
  - Custo + descricao: 0.7%
  - Horario + custo: 0.4%
  - Muito curta (<50 chars): 1.7%

### 4.2 Custo (cost_euros)
| Fonte | N | % |
|-------|---|---|
| default (EUR5.0 arbitrario) | 9.879 | 86.7% |
| google_price_level | 1.290 | 11.3% |
| osm | 226 | 2.0% |

- 57.0% dos POIs tem custo exactamente EUR5.0 (valor default suspeito)
- 4.8% tem custo EUR0.0
- Custo real disponivel apenas para 13.3% dos POIs

### 4.3 Horario (opening_time / closing_time)
| Fonte | N | % |
|-------|---|---|
| default (09:00-18:00) | 6.645 | 58.3% |
| osm | 3.020 | 26.5% |
| google | 1.730 | 15.2% |

### 4.4 Score de relevancia
- score_source = "heuristic" para 100% dos POIs (valor fixo 0.7)
- O score nao tem valor diferencial entre POIs
- Nao existe qualquer fonte de popularidade ou relevancia real

### 4.5 Matching com fontes externas
| Fonte | N matchados | % |
|-------|-------------|---|
| Google Places | 3.804 | 33.4% |
| OpenStreetMap | 3.913 | 34.3% |
| Ambos | ~2.500 (estimativa) | ~22% |
| Nenhum | ~5.178 | ~45% |

---

## 5. Problemas de Qualidade Conhecidos

### 5.1 Custos
- 86.7% dos POIs tem custo "default" (EUR5.0) — sem valor real
- Hoteis a EUR5.0/noite e EUR0.0/noite — claramente errado
- O default devia ter sido EUR80 para alojamento, nao EUR5
- 3.161 POIs de alojamento com custo < EUR10

### 5.2 Bundle/Categoria
- 278 POIs com nome que sugere bundle diferente do atribuido:
  - "Museu do Pao" -> restaurantes (devia ser museus)
  - "Cyber Snack Museu" -> bares (devia ser museus)
  - Hoteis-casino classificados como hotelaria (devia ser casinos)
  - Hoteis rurais classificados como turismo_espaco_rural vs hotelaria
- Bundles sem utilidade para routing: agencias_de_viagem, rentacar, postos_de_turismo, organismos_e_associacoes, aeroportos, estacoes_ferroviarias/rodoviarias
- 115 POIs foram dropados do pipeline de enriquecimento (ja reintegrados)

### 5.3 Descricoes
- 27.6% sem qualquer descricao
- Das com descricao, 55.9% sao apenas texto geral sem informacao operacional
- Codificacao: alguns POIs tem caracteres corrompidos (artefactos de encoding cp1252 vs utf-8)

### 5.4 Horarios
- 58.3% usa o default 09:00-18:00 (sem dados reais)
- Barras e discotecas com horario 09:00-18:00 (claramente errado)
- Eventos com horarios genericos

### 5.5 Bundles sem relevancia para routing
Estes bundles existem nos dados mas nao devem ser incluidos em rotas:
- agencias_de_viagem (804 POIs)
- rentacar (458 POIs)
- servicos_de_turismo (423 POIs)
- postos_de_turismo (385 POIs)
- localidade (238 POIs) — descricoes de cidades, nao POIs
- organismos_e_associacoes (26 POIs)
- aeroportos, estacoes_ferroviarias/rodoviarias, rodoviarios

---

## 6. Bundles Excluidos do RAG (Filtro Ativo)

- eventos: 271 POIs (excluidos via $nin no RAG)

---

## 7. Ficheiros Derivados

| Ficheiro | Conteudo |
|---------|---------|
| data/pois_horario_descricao.json | 1.574 POIs com horario + descricao |
| data/pois_horario_custo.json | 40 POIs com horario + custo |
| data/pois_custo_descricao.json | 82 POIs com custo + descricao |
| data/alojamento_custo_baixo.json | 3.161 POIs de alojamento com custo < EUR10 |
| data/bundle_mismatches.json | 278 POIs com bundle potencialmente errado |

---

## 8. Recomendacoes

1. **Custos de alojamento**: usar Google Places API (google_place_id disponivel para 3.804 POIs) para obter price_level real; aplicar mapeamento bundle->preco para os restantes com default correcto (hotelaria: EUR80, hostel: EUR25, rural: EUR60, campismo: EUR15)
2. **Descricoes em falta**: gerar com LLM (nome + categoria + regiao) para os 3.145 POIs sem descricao — melhora qualidade do RAG
3. **Horarios**: para barras/discotecas/restaurantes sem horario Google/OSM, aplicar defaults por bundle (barras: 21:00-04:00, restaurantes: 12:00-23:00)
4. **Bundle mismatch**: corrigir os 278 casos mais criticos (especialmente museus em restaurantes e casinos em hotelaria)
5. **Bundles irrelevantes**: excluir do RAG via $nin: agencias_de_viagem, rentacar, postos_de_turismo, organismos_e_associacoes, localidade, aeroportos, estacoes
6. **Score**: substituir o heuristic fixo (0.7) por um score calculado: num_fontes_matched * 0.3 + has_description * 0.3 + has_real_cost * 0.2 + has_real_hours * 0.2
