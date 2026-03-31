# src/llm/llm_orchestrator.py (VERSÃO COM GROQ)

from groq import Groq
import json
import re
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class UserPreferences:
    max_time: int
    max_cost: float
    preferred_categories: List[str]
    category_weights: Dict[str, float]
    start_time: str
    interests: List[str]
    secondary_tags: List[str] = None
    location: str = None
    missing_fields: List[str] = None
    transport_mode: str = "foot"

class LlamaOrchestrator:
    """
    Orquestrador LLM com mapeamento semântico inteligente
    """
    
    VALID_TAGS = [
        # Bundles principais
        "restaurantes_e_cafes", "monumentos", "turismo_activo",
        "praias", "bares_e_discotecas", "museus_e_palacios",
        "eventos", "campos", "arqueologia", "espacos_verdes",
        "marinas_e_portos", "termas", "parques_e_reservas",
        "parques_de_diversao", "zoos_e_aquarios", "ciencia_e_conhecimento",
        "casinos", "talassoterapia", "grutas", "academias", "barragens",
        # Tags semânticas
        "historico", "natureza", "familia", "romantico",
        "gratuito", "fotografia", "noturno", "aventura",
    ]

    TAG_TO_MAIN_CATEGORY = {
        # Bundles mapeiam para si próprios
        "restaurantes_e_cafes": "restaurantes_e_cafes",
        "monumentos": "monumentos",
        "turismo_activo": "turismo_activo",
        "praias": "praias",
        "bares_e_discotecas": "bares_e_discotecas",
        "museus_e_palacios": "museus_e_palacios",
        "eventos": "eventos",
        "campos": "campos",
        "arqueologia": "arqueologia",
        "espacos_verdes": "espacos_verdes",
        "marinas_e_portos": "marinas_e_portos",
        "termas": "termas",
        "parques_e_reservas": "parques_e_reservas",
        "parques_de_diversao": "parques_de_diversao",
        "zoos_e_aquarios": "zoos_e_aquarios",
        "ciencia_e_conhecimento": "ciencia_e_conhecimento",
        "casinos": "casinos",
        "talassoterapia": "talassoterapia",
        "grutas": "grutas",
        "academias": "academias",
        "barragens": "barragens",
        # Tags semânticas mapeiam para o bundle mais relevante
        "historico": "monumentos",
        "natureza": "espacos_verdes",
        "familia": "parques_de_diversao",
        "romantico": "restaurantes_e_cafes",
        "noturno": "bares_e_discotecas",
        "aventura": "turismo_activo",
        "gratuito": None,
        "fotografia": None,
    }
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant" 
    
    def _call_llm(self, prompt: str, max_tokens: int = 600, temperature: float = 0.3) -> str:
        """Chama o modelo via Groq API"""
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def extract_preferences(self, user_query: str) -> UserPreferences:
        """
        Extrai preferências com mapeamento semântico de tags
        """
        
        valid_tags_str = ", ".join(self.VALID_TAGS)
        
        prompt = f"""Tu és um assistente de turismo inteligente. Analisa a query do utilizador e extrai preferências.

TAGS VÁLIDAS DO SISTEMA:
{valid_tags_str}

QUERY DO UTILIZADOR:
"{user_query}"

TAREFA:
1. Identifica os interesses e necessidades do utilizador
2. Mapeia SEMANTICAMENTE para as TAGS VÁLIDAS acima
3. Pensa em sinónimos e contexto:
   - "música ao vivo" → music_venue, concert, live_music
   - "natureza" → park, nature, outdoor, garden
   - "comida tradicional" → restaurant, traditional, portuguese_culture, food
   - "desporto" → sports, outdoor, hiking, bike_friendly, surf
   - "crianças" → family, children, educational, interactive
   - "jantar romântico" → restaurant, romantic, food, sunset
   - "história" → monument, historic, museum, architecture
   - "praia" → beach, outdoor, nature, surf

4. Extrai também:
   - Tempo disponível (IMPORTANTE - converter corretamente):
     * Se mencionar "horas": converter para minutos (ex: 5 horas = 300)
     * Se mencionar "dias": assumir 8 horas úteis por dia em minutos
       · 1 dia = 480 minutos (8 horas)
       · 2 dias = 960 minutos (16 horas)
       · 3 dias = 1440 minutos (24 horas)
       · 1 semana = 3360 minutos (56 horas)
   
   - Orçamento (em euros):
     * Se mencionar "por dia" ou "por pessoa": multiplicar pelo número de dias/pessoas
     * Ex: "60 euros por dia, 3 dias" = 180 euros total
     * Ex: "50 euros por pessoa, 4 pessoas" = 200 euros total
   
   - Hora de início (padrão 09:00)

EXEMPLOS DE CONVERSÃO:
- "3 dias" → max_time: 1440
- "5 horas" → max_time: 300
- "meio dia" → max_time: 240
- "1 semana" → max_time: 3360
- "60 euros por dia, 3 dias" → max_cost: 180
- "40 euros por pessoa, 2 pessoas" → max_cost: 80

Devolve APENAS JSON (sem texto adicional):
{{
  "max_time": 300,
  "max_cost": 50.0,
  "tags": ["tag1", "tag2", "tag3"],
  "interests": ["interest1", "interest2"],
  "start_time": "09:00",
  "location": "Lisboa",
  "transport_mode": "foot",
  "missing_fields": ["group_size"]
}}

REGRAS PARA transport_mode:
- "a pé", "a andar", "walking" → "foot"
- "de carro", "carro", "car", "driving" → "car"
- "transportes públicos", "metro", "autocarro", "comboio" → "public_transport"
- Se não mencionado → incluir "transport_mode" em missing_fields

CAMPOS A VERIFICAR PARA missing_fields:
- "location": se não foi mencionada nenhuma localização em Portugal
- "max_time": se o utilizador não mencionou duração nem número de dias
- "max_cost": se o utilizador não mencionou orçamento nem preço
- "group_size": se não foi mencionado número de pessoas
- "transport_mode": sempre incluir se não mencionado (foot, car, public_transport)
- "has_children": SÓ incluir se não foi mencionado de todo. Se a query contiver "sem crianças", "viajamos sem crianças", "não temos crianças", "grupo de amigos", "casal" → NÃO incluir, assume false
- "mobility_issues": SÓ incluir se não foi mencionado de todo. Se a query contiver "sem problemas de mobilidade", "mobilidade normal", "grupo de amigos" → NÃO incluir, assume false

REGRAS:
- Só inclui em missing_fields campos que realmente faltam e são relevantes para a query
- Se a query for muito curta ou vaga, inclui mais campos
- Se a query for detalhada, missing_fields pode ser []

REGRAS PARA location:
- "quero visitar museus em Lisboa" → "Lisboa"
- "praia no Algarve" → "Algarve"
- "casal em Sintra, 6 horas" → "Sintra"
- "Serra da Estrela" → "Serra da Estrela"
- Se não mencionar localização → null

REGRAS IMPORTANTES:
- Usa APENAS tags da lista VÁLIDA acima
- CUIDADO com conversão de tempo: dias × 480 minutos
- CUIDADO com orçamento: multiplicar por dias/pessoas se mencionado
- Seleciona NO MÁXIMO 5 tags — as mais relevantes para a query
- NÃO uses tags genéricas se a query for específica
- Pensa semanticamente (sinónimos, contexto, relacionados)

Responde APENAS com o JSON, sem explicações."""

        content = ""
        try:
            content = self._call_llm(prompt, max_tokens=600, temperature=0.3)
            content = re.sub(r'```json\s*|\s*```', '', content).strip()
            
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                content = content[start:end]
            
            data = json.loads(content)
            
            extracted_tags = data.get("tags", [])
            valid_extracted_tags = [tag for tag in extracted_tags if tag in self.VALID_TAGS]
            
            if len(valid_extracted_tags) < len(extracted_tags):
                print(f"   ⚠️ Algumas tags inválidas foram removidas")
                print(f"      Original: {extracted_tags}")
                print(f"      Válidas: {valid_extracted_tags}")
            
            main_categories = []
            secondary_tags = []
            
            for tag in valid_extracted_tags:
                mapped_category = self.TAG_TO_MAIN_CATEGORY.get(tag)
                
                if mapped_category:
                    if mapped_category not in main_categories:
                        main_categories.append(mapped_category)
                else:
                    secondary_tags.append(tag)
            
            if not main_categories:
                main_categories = None
            
            category_weights = {}
            if main_categories:
                for category in main_categories:
                    category_weights[category] = 0.8
            
            # ✅ FIX: Validar tempo extraído
            extracted_time = data.get("max_time", 300)
            if extracted_time > 5000:
                print(f"   ⚠️ Tempo extraído parece errado: {extracted_time} min")
                print(f"      Limitando a 1440 min (1 dia útil)")
                extracted_time = 1440
            
            # ✅ FIX: Validar custo extraído
            # ✅ FIX: Validar custo extraído
            extracted_cost = float(data.get("max_cost", 50.0))
            if extracted_cost > 1000:
                print(f"   ⚠️ Orçamento extraído parece alto: €{extracted_cost}")

            # Extrair localização
            extracted_location = data.get("location", None)
            if extracted_location and not isinstance(extracted_location, str):
                extracted_location = None
            if extracted_location:
                print(f"   📍 Localização extraída: '{extracted_location}'")

            # Extrair modo de transporte
            transport_mode = data.get("transport_mode", None)
            if not transport_mode or transport_mode not in ["foot", "car", "public_transport"]:
                transport_mode = None
                if "transport_mode" not in data.get("missing_fields", []):
                    data.setdefault("missing_fields", []).append("transport_mode")
            if transport_mode:
                print(f"   🚗 Modo de transporte: '{transport_mode}'")

            # Extrair campos em falta
            missing_fields = data.get("missing_fields", [])
            if missing_fields:
                print(f"   ❓ Campos em falta: {missing_fields}")

            print(f"   🔄 Tags extraídas pelo LLM: {valid_extracted_tags}")
            print(f"   🎯 Categorias principais (filtro): {main_categories}")
            print(f"   🏷️  Tags secundárias (semântica): {secondary_tags}")

            return UserPreferences(
                max_time=extracted_time,
                max_cost=extracted_cost,
                preferred_categories=main_categories,
                category_weights=category_weights,
                start_time=data.get("start_time", "09:00"),
                interests=data.get("interests", []),
                secondary_tags=secondary_tags,
                location=extracted_location,
                missing_fields=missing_fields,
                transport_mode=transport_mode or "foot"
            )
        
        except Exception as e:
            print(f"⚠️ Erro ao extrair preferências: {e}")
            print(f"   Resposta LLM: {content if content else 'N/A'}")
            
            return UserPreferences(
                max_time=480,
                max_cost=50.0,
                preferred_categories=["museum", "monument", "park"],
                category_weights={"museum": 0.8, "monument": 0.8, "park": 0.8},
                start_time="09:00",
                interests=["culture", "history"],
                secondary_tags=[]
            )
    
    def generate_rag_query(self, preferences: UserPreferences, user_history=None) -> str:
        """Gera query otimizada para o RAG"""
        
        query_parts = []
        
        if preferences.preferred_categories:
            query_parts.extend(preferences.preferred_categories)
        
        if preferences.secondary_tags:
            query_parts.extend(preferences.secondary_tags)
        
        if preferences.interests:
            query_parts.extend(preferences.interests)
        
        query = " ".join(query_parts)
        return query
    
    def explain_route(self, route: List[Dict], preferences: UserPreferences,
                     algorithm_used: str, optimization_metadata: Dict) -> str:
        """Gera explicação em português sobre a rota gerada"""
        
        route_summary = []
        total_cost = 0
        total_duration = 0
        
        for poi in route:
            route_summary.append(f"{poi['name']} ({poi['category']})")
            total_cost += poi['cost']
            total_duration += poi['duration']
        
        route_str = ", ".join(route_summary)
        
        prompt = f"""Gera uma explicação CURTA e AMIGÁVEL em português sobre esta rota turística.

ROTA GERADA:
{route_str}

DETALHES:
- Algoritmo usado: {algorithm_used}
- Fitness score: {optimization_metadata.get('fitness', 0):.2f}
- POIs selecionados: {len(route)}
- Duração total: {total_duration} minutos
- Custo total: €{total_cost:.2f}
- Preferências utilizador: {', '.join(preferences.interests)}

TAREFA:
Escreve um parágrafo curto (3-4 frases) explicando:
1. Por que esta rota é adequada para o utilizador
2. Destaca 1-2 POIs principais
3. Menciona a diversidade ou características especiais

TOM: Amigável, informativo, português de Portugal
TAMANHO: Máximo 4 frases

Responde APENAS com o texto da explicação, sem introduções."""

        try:
            explanation = self._call_llm(prompt, max_tokens=300, temperature=0.7).strip()
            return explanation
        
        except Exception as e:
            print(f"⚠️ Erro ao gerar explicação: {e}")
            return f"Esta rota foi otimizada com o algoritmo {algorithm_used} para incluir {len(route)} POIs que correspondem aos teus interesses em {', '.join(preferences.interests)}. O percurso tem uma duração total de {total_duration} minutos e custa €{total_cost:.2f}."
        
        
def select_algorithm_deterministic(n_candidates: int, max_time: int) -> str:
    """
    Thresholds derivados empiricamente do benchmark 16 queries × 4 algoritmos.

    Resultados:
    - GA venceu 9/16 queries, dominante em max_time ≤ 1920 min (até ~4 dias)
    - PSO venceu 7/16 queries, dominante em max_time ≥ 2400 min (5+ dias)
    - ACO: 0 vitórias — excluído da selecção automática
    - GREEDY: 0 vitórias — excluído da selecção automática

    Limiar de transição GA→PSO: entre 1920 e 2400 min → corte em 2400.
    n_candidates não revelou padrão discriminativo — max_time é o eixo relevante.
    """
    if max_time < 2400:
        return "GA"
    else:
        return "PSO" 