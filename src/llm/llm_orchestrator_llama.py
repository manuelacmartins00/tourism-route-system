# src/llm/llm_orchestrator.py (VERSAO COM GROQ)

from groq import Groq
import json
import re
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class UserPreferences:
    """Preferencias do utilizador extraidas pelo LLM"""
    max_time: int
    max_cost: float
    preferred_categories: List[str]
    category_weights: Dict[str, float]
    start_time: str
    interests: List[str]
    secondary_tags: List[str] = None

class LlamaOrchestrator:
    """
    Orquestrador LLM com mapeamento semantico inteligente
    """
    
    VALID_TAGS = [
        # Categorias Principais
        "monument", "museum", "park", "viewpoint", "beach",
        "restaurant", "cafe", "music_venue", "nightlife",
        "sports", "family", "attraction", "shopping",
        "religious", "educational", "wellness", "transport",
        
        # Tags Secundarias
        "historic", "architecture", "art", "nature", "outdoor",
        "food", "culture", "photography", "sunset", "panorama",
        "children", "interactive", "traditional", "modern",
        "waterfront", "unesco", "concert", "theater", "surf",
        "hiking", "bike_friendly", "free", "trendy", "romantic",
        "fado", "portuguese_culture", "contemporary", "medieval",
        "garden", "aquarium", "animals", "science", "fitness",
        "rooftop", "dance", "electronic", "seafood", "pastry",
        "iconic", "industrial", "scenic", "educational", "spa",
        "relaxation", "massage", "market", "fresh_produce",
        "football", "stadium", "tour", "multimedia", "engineering",
        "walking", "roleplay", "dinner_show", "live_music"
    ]
    
    TAG_TO_MAIN_CATEGORY = {
        "monument": "monument",
        "museum": "museum",
        "park": "park",
        "viewpoint": "viewpoint",
        "beach": "beach",
        "restaurant": "restaurant",
        "cafe": "cafe",
        "music_venue": "music_venue",
        "nightlife": "nightlife",
        "sports": "sports",
        "attraction": "attraction",
        "shopping": "shopping",
        "religious": "religious",
        "educational": "educational",
        "wellness": "wellness",
        "transport": "transport",
        
        "family": "attraction",
        "children": "attraction",
        "interactive": "attraction",
        "animals": "attraction",
        "aquarium": "attraction",
        
        "historic": "monument",
        "architecture": "monument",
        "unesco": "monument",
        "medieval": "monument",
        
        "nature": "park",
        "outdoor": "park",
        "garden": "park",
        "hiking": "park",
        "bike_friendly": "park",
        
        "food": "restaurant",
        "seafood": "restaurant",
        "traditional": "restaurant",
        "pastry": "cafe",
        
        "concert": "music_venue",
        "live_music": "music_venue",
        "fado": "music_venue",
        "theater": "music_venue",
        "dinner_show": "music_venue",
        
        "art": "museum",
        "contemporary": "museum",
        "science": "museum",
        "multimedia": "museum",
        
        "sunset": "viewpoint",
        "panorama": "viewpoint",
        "photography": "viewpoint",
        "rooftop": "viewpoint",
        
        "surf": "beach",
        "waterfront": "beach",
        
        "spa": "wellness",
        "massage": "wellness",
        "fitness": "wellness",
        "relaxation": "wellness",
        
        "market": "shopping",
        "fresh_produce": "shopping",
        
        "football": "sports",
        "stadium": "sports",
        
        "tour": "educational",
        "engineering": "educational",
        
        "free": None,
        "trendy": None,
        "romantic": None,
        "iconic": None,
        "scenic": None,
        "modern": None,
        "industrial": None,
        "portuguese_culture": None,
        "culture": None,
        "walking": None,
        "roleplay": None,
        "dance": None,
        "electronic": None,
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
        Extrai preferencias com mapeamento semantico de tags
        """
        
        valid_tags_str = ", ".join(self.VALID_TAGS)
        
        prompt = f"""Tu es um assistente de turismo inteligente. Analisa a query do utilizador e extrai preferencias.

TAGS VALIDAS DO SISTEMA:
{valid_tags_str}

QUERY DO UTILIZADOR:
"{user_query}"

TAREFA:
1. Identifica os interesses e necessidades do utilizador
2. Mapeia SEMANTICAMENTE para as TAGS VALIDAS acima
3. Pensa em sinonimos e contexto:
   - "musica ao vivo" -> music_venue, concert, live_music
   - "natureza" -> park, nature, outdoor, garden
   - "comida tradicional" -> restaurant, traditional, portuguese_culture, food
   - "desporto" -> sports, outdoor, hiking, bike_friendly, surf
   - "criancas" -> family, children, educational, interactive
   - "jantar romantico" -> restaurant, romantic, food, sunset
   - "historia" -> monument, historic, museum, architecture
   - "praia" -> beach, outdoor, nature, surf

4. Extrai tambem:
   - Tempo disponivel (IMPORTANTE - converter corretamente):
     * Se mencionar "horas": converter para minutos (ex: 5 horas = 300)
     * Se mencionar "dias": assumir 8 horas uteis por dia em minutos
       - 1 dia = 480 minutos (8 horas)
       - 2 dias = 960 minutos (16 horas)
       - 3 dias = 1440 minutos (24 horas)
       - 1 semana = 3360 minutos (56 horas)

   - Orcamento (em euros):
     * Se mencionar "por dia" ou "por pessoa": multiplicar pelo numero de dias/pessoas
     * Ex: "60 euros por dia, 3 dias" = 180 euros total
     * Ex: "50 euros por pessoa, 4 pessoas" = 200 euros total

   - Hora de inicio (padrao 09:00)

EXEMPLOS DE CONVERSAO:
- "3 dias" -> max_time: 1440
- "5 horas" -> max_time: 300
- "meio dia" -> max_time: 240
- "1 semana" -> max_time: 3360
- "60 euros por dia, 3 dias" -> max_cost: 180
- "40 euros por pessoa, 2 pessoas" -> max_cost: 80

Devolve APENAS JSON (sem texto adicional):
{{
  "max_time": 300,
  "max_cost": 50.0,
  "tags": ["tag1", "tag2", "tag3"],
  "interests": ["interest1", "interest2"],
  "start_time": "09:00"
}}

REGRAS IMPORTANTES:
- Usa APENAS tags da lista VALIDA acima
- CUIDADO com conversao de tempo: dias x 480 minutos
- CUIDADO com orcamento: multiplicar por dias/pessoas se mencionado
- Multiplas tags sao permitidas e encorajadas
- Pensa semanticamente (sinonimos, contexto, relacionados)

Responde APENAS com o JSON, sem explicacoes."""

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
                print(f"   [WARN] Algumas tags invalidas foram removidas")
                print(f"      Original: {extracted_tags}")
                print(f"      Validas: {valid_extracted_tags}")
            
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
            
            # FIX: Validar tempo extraido
            extracted_time = data.get("max_time", 300)
            if extracted_time > 5000:
                print(f"   [WARN] Tempo extraido parece errado: {extracted_time} min")
                print(f"      Limitando a 1440 min (1 dia util)")
                extracted_time = 1440

            # FIX: Validar custo extraido
            extracted_cost = float(data.get("max_cost", 50.0))
            if extracted_cost > 1000:
                print(f"   [WARN] Orcamento extraido parece alto: EUR{extracted_cost}")

            print(f"   [INFO] Tags extraidas pelo LLM: {valid_extracted_tags}")
            print(f"   [INFO] Categorias principais (filtro): {main_categories}")
            print(f"   [INFO] Tags secundarias (semantica): {secondary_tags}")
            
            return UserPreferences(
                max_time=extracted_time,
                max_cost=extracted_cost,
                preferred_categories=main_categories,
                category_weights=category_weights,
                start_time=data.get("start_time", "09:00"),
                interests=data.get("interests", []),
                secondary_tags=secondary_tags
            )
        
        except Exception as e:
            print(f"[WARN] Erro ao extrair preferencias: {e}")
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
    
    def generate_rag_query(self, preferences: UserPreferences) -> str:
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
    
    def select_algorithm(self, preferences: UserPreferences, n_candidates: int) -> str:
        """Seleciona algoritmo de otimizacao"""

        prompt = f"""Seleciona o MELHOR algoritmo de otimizacao para este problema de roteamento turistico.

CONTEXTO:
- Numero de POIs candidatos: {n_candidates}
- Tempo disponivel: {preferences.max_time} minutos
- Orcamento: EUR{preferences.max_cost}
- Categorias desejadas: {len(preferences.preferred_categories) if preferences.preferred_categories else 0}

ALGORITMOS DISPONIVEIS:
1. ACO (Ant Colony Optimization)
   - Melhor para: problemas multi-objetivo, exploracao de multiplas rotas
   - Recomendado quando: n_candidates > 15, multiplas categorias

2. GA (Genetic Algorithm)
   - Melhor para: grandes espacos de busca, diversidade de solucoes
   - Recomendado quando: n_candidates > 12, variedade importante

3. PSO (Particle Swarm Optimization)
   - Melhor para: convergencia rapida
   - Recomendado quando: n_candidates 8-15, tempo limitado

4. GREEDY (Algoritmo Guloso)
   - Melhor para: solucoes rapidas, poucos candidatos
   - Recomendado quando: n_candidates < 10

DECISAO:
Escolhe o algoritmo MAIS APROPRIADO e responde APENAS com uma palavra: ACO, GA, PSO ou GREEDY."""

        try:
            algo = self._call_llm(prompt, max_tokens=50, temperature=0.1).strip().upper()
            
            if algo in ["ACO", "GA", "PSO", "GREEDY"]:
                return algo
            else:
                print(f"[WARN] Algoritmo invalido '{algo}', usando ACO por padrao")
                return "ACO"

        except Exception as e:
            print(f"[WARN] Erro ao selecionar algoritmo: {e}, usando ACO por padrao")
            return "ACO"
    
    def explain_route(self, route: List[Dict], preferences: UserPreferences,
                     algorithm_used: str, optimization_metadata: Dict) -> str:
        """Gera explicacao em portugues sobre a rota gerada"""
        
        route_summary = []
        total_cost = 0
        total_duration = 0
        
        for poi in route:
            route_summary.append(f"{poi['name']} ({poi['category']})")
            total_cost += poi['cost']
            total_duration += poi['duration']
        
        route_str = ", ".join(route_summary)
        
        prompt = f"""Gera uma explicacao CURTA e AMIGAVEL em portugues sobre esta rota turistica.

ROTA GERADA:
{route_str}

DETALHES:
- Algoritmo usado: {algorithm_used}
- Fitness score: {optimization_metadata.get('fitness', 0):.2f}
- POIs selecionados: {len(route)}
- Duracao total: {total_duration} minutos
- Custo total: EUR{total_cost:.2f}
- Preferencias utilizador: {', '.join(preferences.interests)}

TAREFA:
Escreve um paragrafo curto (3-4 frases) explicando:
1. Por que esta rota e adequada para o utilizador
2. Destaca 1-2 POIs principais
3. Menciona a diversidade ou caracteristicas especiais

TOM: Amigavel, informativo, portugues de Portugal
TAMANHO: Maximo 4 frases

Responde APENAS com o texto da explicacao, sem introducoes."""

        try:
            explanation = self._call_llm(prompt, max_tokens=300, temperature=0.7).strip()
            return explanation
        
        except Exception as e:
            print(f"[WARN] Erro ao gerar explicacao: {e}")
            return f"Esta rota foi otimizada com o algoritmo {algorithm_used} para incluir {len(route)} POIs que correspondem aos teus interesses em {', '.join(preferences.interests)}. O percurso tem uma duracao total de {total_duration} minutos e custa EUR{total_cost:.2f}."