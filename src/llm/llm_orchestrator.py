# src/llm/llm_orchestrator.py (VERSAO COM GROQ)

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
    start_location: str = None
    mobility_issues: bool = False
    num_people: int = 1
    has_children: bool = False
    last_day_end_time: str = None
    end_location: str = None

class LlamaOrchestrator:
    """
    Orquestrador LLM com mapeamento semantico inteligente
    """
    
    VALID_TAGS = [
        # Bundles principais
        "restaurantes_e_cafes", "monumentos", "turismo_activo",
        "praias", "bares_e_discotecas", "museus_e_palacios",
        "eventos", "campos", "arqueologia", "espacos_verdes",
        "marinas_e_portos", "termas", "parques_e_reservas",
        "parques_de_diversao", "zoos_e_aquarios", "ciencia_e_conhecimento",
        "casinos", "talassoterapia", "grutas", "academias", "barragens",
        # Tags semanticas
        "historico", "natureza", "familia", "romantico",
        "gratuito", "fotografia", "noturno", "aventura",
    ]

    TAG_TO_MAIN_CATEGORY = {
        # Bundles mapeiam para si proprios
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
        # Tags semanticas mapeiam para o bundle mais relevante
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
        Extrai preferencias com mapeamento semantico de tags
        """
        # Pre-processar: substituir ordinais de dia-da-semana
        # Aceita: 6ª, 6ª feira, 6a feira, 6a tarde/manha/noite, 6a a tarde (informal)
        import re as _pre
        import unicodedata as _ud
        _ordinal_map = {'2': 'segunda-feira', '3': 'terca-feira', '4': 'quarta-feira',
                        '5': 'quinta-feira',  '6': 'sexta-feira'}
        def _replace_ordinal(m):
            return _ordinal_map.get(m.group(1), m.group(0))
        user_query = _pre.sub(
            r'\b([2-6])(?:[\xaa\xba]\s*(?:-?\s*feira)?'
            r'|a\s+a\s+(?:tarde|manha|noite)'
            r'|a\s*(?:-?\s*feira|tarde|manha|noite|de\s+tarde|de\s+manha)'
            r'|\s+-?\s*feira)\b',
            _replace_ordinal,
            user_query,
            flags=_pre.IGNORECASE
        )

        # Pre-processar: "fim de semana" = 2 dias fixo
        _fim_semana = bool(_pre.search(r'\bfim\s+de\s+semana\b', user_query, _pre.IGNORECASE))

        # Pre-processar: palavras de periodo do dia -> start_time + duracao implicita
        _TIME_OF_DAY = [
            (r'\bmanha\s+cedo\b',    '08:00', 240),
            (r'\bde\s+manha\b',      '09:00', 240),
            (r'\bpela\s+manha\b',    '09:00', 240),
            (r'\bde\s+manhã\b',      '09:00', 240),
            (r'\bmanha\b',           '09:00', 240),
            (r'\bao\s+almoco\b',     '12:00', 120),
            (r'\bmeio[- ]dia\b',     '12:00', 120),
            (r'\bfinal\s+da\s+tarde\b', '17:00', 120),
            (r'\bao\s+fim\s+da\s+tarde\b', '17:00', 120),
            (r'\bda\s+tarde\b',      '14:00', 240),
            (r'\bde\s+tarde\b',      '14:00', 240),
            (r'\bpela\s+tarde\b',    '14:00', 240),
            (r'\ba\s+tarde\b',       '14:00', 240),
            (r'\btarde\b',           '14:00', 240),
            (r'\bao\s+jantar\b',     '19:00', 120),
            (r'\ba\s+noite\b',       '20:00', None),
            (r'\bnoite\b',           '20:00', None),
        ]
        _inferred_start_time = None
        _inferred_duration_min = None
        _q_lower = user_query.lower()
        for _pat, _t, _d in _TIME_OF_DAY:
            if _pre.search(_pat, _q_lower):
                _inferred_start_time = _t
                _inferred_duration_min = _d
                break

        # Pre-processar: detetar intervalos de dias da semana implicitos
        _DAY_NUM = {
            'segunda': 1, 'terca': 2, 'quarta': 3, 'quinta': 4,
            'sexta': 5, 'sabado': 6, 'domingo': 7,
        }
        def _norm(s):
            return ''.join(c for c in _ud.normalize('NFKD', s.lower()) if not _ud.combining(c))
        def _day_n(name):
            return _DAY_NUM.get(_norm(name.replace('-feira', '').strip()))

        _implicit_days = None
        if _fim_semana:
            _implicit_days = 2
        else:
            # Padrao: de [dia1] ... (a|ate) ... [dia2]
            _range_m = _pre.search(
                r'\bde\s+(segunda|terca|quarta|quinta|sexta|sabado|domingo)(?:-feira)?\b'
                r'.{0,30}?\b(?:a|ate)\b.{0,15}?\b(segunda|terca|quarta|quinta|sexta|sabado|domingo)(?:-feira)?\b',
                _norm(user_query), _pre.IGNORECASE
            )
            if _range_m:
                d1, d2 = _day_n(_range_m.group(1)), _day_n(_range_m.group(2))
                if d1 and d2:
                    diff = (d2 - d1) % 7
                    _implicit_days = max(1, diff + 1)
            # Padrao: [dia1] e [dia2] (ex: "sexta e sabado")
            if _implicit_days is None:
                _enum = _pre.findall(
                    r'\b(segunda|terca|quarta|quinta|sexta|sabado|domingo)(?:-feira)?\b',
                    _norm(user_query), _pre.IGNORECASE
                )
                if len(_enum) >= 2:
                    nums = sorted({_day_n(d) for d in _enum if _day_n(d)})
                    if nums:
                        _implicit_days = (max(nums) - min(nums)) + 1

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
   
   - Orcamento - extrair DOIS campos separados:
     * "budget_value": o valor numerico mencionado (nunca calcular, nunca multiplicar)
     * "budget_type": o tipo do orcamento:
       - "per_person"         -> "por pessoa", "each", "per person", "p/pessoa"
       - "per_day"            -> "por dia" (para o grupo todo), "daily"
       - "per_person_per_day" -> "por pessoa por dia", "per person per day"
       - "total"              -> "no total", "para todos", "para o grupo", sem especificar tipo
     * Se so foi dito "budget baixo"  -> budget_value: 25,  budget_type: "per_person"
     * Se so foi dito "budget medio"  -> budget_value: 70,  budget_type: "per_person"
     * Se so foi dito "budget alto"   -> budget_value: 150, budget_type: "per_person"
     * Se foi dado valor sem tipo claro -> incluir "budget_type" em missing_fields

   - Numero de pessoas:
     * "num_people": extrair de "5 amigos", "somos 3", "familia de 4", "eu e a minha namorada" (2), etc.
     * Se nao mencionado -> num_people: 1 (padrao, nao perguntar)
     * APENAS incluir "num_people" em missing_fields se budget_type for "total" ou "per_day"
       E num_people nao puder ser determinado E houver indicio de grupo (plural, "amigos", "familia", "nos")

   - Hora de inicio (apenas para o PRIMEIRO dia, padrao "09:00"):
     * "de manha" / "de manha cedo" -> "09:00"
     * "ao meio-dia" -> "12:00"
     * "a tarde" / "da tarde" -> "16:00"
     * "ao final da tarde" -> "18:00"
     * "a noite" -> "20:00"

   - Hora de fim do ULTIMO dia - campo "last_day_end_time" (null se nao especificado):
     * "ate de manha" / "termina de manha" -> "12:00"
     * "ate ao meio-dia" -> "12:00"
     * "ate a tarde" / "ate domingo a tarde" -> "17:00"
     * "ate ao final da tarde" -> "18:00"
     * "ate a noite" -> "21:00"
     * Se nao especificado -> null

EXEMPLOS DE CONVERSAO:
- "3 dias" -> max_time: 1440
- "5 horas" -> max_time: 300
- "meio dia" -> max_time: 240
- "1 semana" -> max_time: 3360
- "60 euros por dia" -> budget_value: 60, budget_type: "per_day"
- "40 euros por pessoa" -> budget_value: 40, budget_type: "per_person"
- "1000 euros para o grupo" -> budget_value: 1000, budget_type: "total"
- "50EUR por pessoa por dia" -> budget_value: 50, budget_type: "per_person_per_day"
- "5 amigos" -> num_people: 5
- "somos 3" -> num_people: 3
- "eu e a minha namorada" -> num_people: 2
- "familia com 2 criancas" -> num_people: 4 (2 adultos + 2 criancas)
- "casal com 1 filho" -> num_people: 3
- "de 6a a tarde ate domingo a tarde" -> start_time: "16:00", last_day_end_time: "17:00", max_time: 1440
- "de sabado de manha ate domingo ao meio-dia" -> start_time: "09:00", last_day_end_time: "12:00", max_time: 960
- "fim de semana" -> max_time: 960 (sabado + domingo = 2 dias)
- "familia com 2 criancas" -> num_people: 4 (2 adultos + 2 criancas)

REGRA CRITICA PARA missing_fields:
- Se max_time nao foi mencionado: max_time deve ser null E "max_time" em missing_fields
- Se budget nao foi mencionado: budget_value deve ser null E "max_cost" em missing_fields
- NUNCA inventar valores para estes campos — null obrigatorio se nao mencionados
- "carro", "de carro", "a carro", "carro proprio", "carro alugado" -> transport_mode: "car"

Devolve APENAS JSON (sem texto adicional):
{{
  "max_time": null,
  "budget_value": null,
  "budget_type": "per_person",
  "num_people": 1,
  "tags": ["tag1", "tag2", "tag3"],
  "interests": ["interest1", "interest2"],
  "start_time": "09:00",
  "last_day_end_time": null,
  "location": "Lisboa",
  "end_location": null,
  "transport_mode": "foot",
  "start_location": null,
  "mobility_issues": false,
  "missing_fields": []
}}

REGRAS PARA transport_mode:
- "a pe", "a andar", "walking" -> "foot"
- "de carro", "carro", "car", "driving" -> "car"
- "transportes publicos", "metro", "autocarro", "comboio" -> "public_transport"
- "mais rapido", "qualquer meio", "o que for mais rapido", "sem preferencia" -> "fastest"
- Se NAO for mencionado EXPLICITAMENTE -> OBRIGATORIO colocar "transport_mode" em missing_fields e deixar o campo como null
- NUNCA assumir um modo de transporte por defeito - e SEMPRE obrigatorio perguntar

CAMPOS A VERIFICAR PARA missing_fields:
- "location": se nao foi mencionada nenhuma localizacao em Portugal
- "max_time": se o utilizador nao mencionou duracao nem numero de dias
- "max_cost": se o utilizador nao mencionou orcamento nem preco de forma alguma (nem vago)
- "budget_type": se foi dado um valor de orcamento mas o tipo nao e claro (nao disse "por pessoa", "por dia", "total", etc.)
- "max_time": OBRIGATORIO incluir se nao foi mencionada duracao nem numero de dias. ESPECIALMENTE obrigatorio se budget_type for "per_day" ou "per_person_per_day" - sem saber os dias, o orcamento total nao pode ser calculado.
- "transport_mode": sempre incluir se nao mencionado (foot, car, public_transport)
- "has_children": SO incluir se nao foi mencionado de todo. Se a query contiver "sem criancas", "viajamos sem criancas", "nao temos criancas", "grupo de amigos", "casal" -> NAO incluir, assume false
- "mobility_issues": NUNCA incluir em missing_fields. Extrair como campo booleano separado:
  * true se o utilizador mencionar cadeira de rodas, mobilidade reduzida, dificuldades a andar, problemas de locomocao, idoso com mobilidade limitada, bengala, andarilho
  * false em todos os outros casos (incluindo "sem problemas de mobilidade", "mobilidade normal", "grupo de amigos")
- "start_location": extrair se o utilizador mencionar onde esta hospedado, o hotel, a residencia ou o ponto de partida diario (ex: "estou hospedado no centro do Porto", "hotel em Alfama"). NUNCA obrigatorio - NUNCA incluir em missing_fields. Se nao mencionado, devolver null.
  
REGRAS:
- So inclui em missing_fields campos que realmente faltam e sao relevantes para a query
- Se a query for muito curta ou vaga, inclui mais campos
- Se a query for detalhada, missing_fields pode ser []

REGRAS PARA location e end_location:
- "quero visitar museus em Lisboa" -> location: "Lisboa", end_location: null
- "praia no Algarve" -> location: "Algarve", end_location: null
- "de Lisboa ao Porto" -> location: "Lisboa", end_location: "Porto"
- "de Porto a Vila Real" -> location: "Porto", end_location: "Vila Real"
- "entre Coimbra e Aveiro" -> location: "Coimbra", end_location: "Aveiro"
- "rota de Faro a Lisboa" -> location: "Faro", end_location: "Lisboa"
- Se mencionar apenas uma localizacao -> end_location: null
- Se mencionar duas localidades ligadas por "a", "ate", "para", "e" em contexto de rota -> extrair as duas

REGRAS IMPORTANTES:
- Usa APENAS tags da lista VALIDA acima
- CUIDADO com conversao de tempo: dias x 480 minutos
- CUIDADO com orcamento: multiplicar por dias/pessoas se mencionado
- Seleciona NO MAXIMO 5 tags - as mais relevantes para a query
- NAO uses tags genericas se a query for especifica
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
                print(f"   AVISO: Algumas tags invalidas foram removidas")
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
                main_categories = []

            # Garantir categorias obvias que o LLM as vezes nao extrai
            _q = user_query.lower()
            _keyword_cats = {
                "bares_e_discotecas": ["vida noturna", "noturno", "bar ", "bares", "discoteca", "night", "beber copos", "sair a noite"],
                "restaurantes_e_cafes": ["comer", "jantar", "restaurante", "gastronomia", "almoco"],
                "museus_e_palacios": ["museu", "museus", "palacio", "palacios"],
            }
            for cat, hints in _keyword_cats.items():
                if any(h in _q for h in hints) and cat not in main_categories:
                    main_categories.append(cat)
                    print(f"   [KEY] Categoria '{cat}' adicionada por keyword")

            if not main_categories:
                main_categories = None

            category_weights = {}
            if main_categories:
                for category in main_categories:
                    category_weights[category] = 0.8
            
            # Validar tempo extraido — null significa nao mencionado
            extracted_time = data.get("max_time")
            if extracted_time is None:
                extracted_time = 300  # placeholder; missing_fields tratara isto
            else:
                extracted_time = int(extracted_time)
                if extracted_time > 5000:
                    print(f"   AVISO: Tempo extraido parece errado: {extracted_time} min - limitando a 1440 min")
                    extracted_time = 1440

            # Numero de pessoas
            num_people = max(1, int(data.get("num_people", 1) or 1))

            # Calcular orcamento per-person total com base no tipo declarado
            import math as _math
            num_days = max(1, _math.ceil(extracted_time / 480))
            raw_budget = data.get("budget_value") or data.get("max_cost")
            budget_value = float(raw_budget) if raw_budget is not None else None
            budget_type  = data.get("budget_type", "per_person") or "per_person"

            # Corrigir budget_type: se LLM retornou "per_person" mas a query diz "por dia"
            _q = user_query.lower()
            _per_day_hints = ["por dia", "per day", "diario", "/dia", "daily", "each day"]
            if budget_type == "per_person" and any(h in _q for h in _per_day_hints):
                budget_type = "per_person_per_day"
                print(f"   [INFO] budget_type corrigido: 'por dia' detectado na query")

            if budget_value is None:
                extracted_cost = 50.0  # placeholder; missing_fields tratara isto
            elif budget_type == "per_person":
                extracted_cost = budget_value
            elif budget_type == "per_person_per_day":
                extracted_cost = budget_value * num_days
            elif budget_type == "per_day":
                extracted_cost = (budget_value * num_days) / num_people
            else:  # "total"
                extracted_cost = budget_value / num_people

            if budget_value is not None:
                if extracted_cost > 1000:
                    print(f"   AVISO: Orcamento por pessoa calculado parece alto: EUR{extracted_cost:.0f}")
                print(f"   Budget: EUR{budget_value} ({budget_type}) x {num_days}d / {num_people}p -> EUR{extracted_cost:.2f}/pessoa")

            # Extrair localizacao
            extracted_location = data.get("location", None)
            if extracted_location and not isinstance(extracted_location, str):
                extracted_location = None
            if extracted_location:
                print(f"   Localizacao extraida: '{extracted_location}'")

            # Extrair modo de transporte
            transport_mode = data.get("transport_mode", None)
            # Verificar se ha keyword explicita de transporte na query
            _transport_kws = [
                "a pe", "a andar", "walking", "foot",
                "carro", "de carro", "a carro", "car", "driving", "automovel",
                "transportes publicos", "transporte publico", "metro", "autocarro",
                "comboio", "autobus", "bus", "public transport",
                "bicicleta", "bike", "cycling", "mais rapido", "fastest",
            ]
            _has_transport_kw = any(kw in user_query.lower() for kw in _transport_kws)

            if not transport_mode or transport_mode not in ["foot", "car", "public_transport", "fastest"]:
                transport_mode = None
            elif not _has_transport_kw:
                # LLM adivinhou — anular e perguntar
                transport_mode = None

            if transport_mode is None:
                if "transport_mode" not in data.get("missing_fields", []):
                    data.setdefault("missing_fields", []).append("transport_mode")
                print(f"   Modo de transporte nao identificado - sera pedido ao utilizador")
            else:
                print(f"   Modo de transporte: '{transport_mode}'")

            # Extrair problemas de mobilidade
            mobility_issues = bool(data.get("mobility_issues", False))
            if mobility_issues:
                print(f"   Mobilidade reduzida identificada - pipeline de elevacao activado")

            # Detetar criancas silenciosamente por keywords (sem perguntar ao utilizador)
            _children_hints = ["filho", "filha", "filhos", "filhas", "crianca", "criancas",
                                "kids", "children", "child", "bebe", "bebes", "bebe", "bebes",
                                "miudo", "miudos", "family with kids", "com criancas"]
            has_children = bool(data.get("has_children", False)) or any(h in _q for h in _children_hints)
            if has_children:
                print(f"   Criancas detectadas - regras contextuais activadas")

            # Extrair hora de fim do ultimo dia (opcional)
            last_day_end_time = data.get("last_day_end_time", None)
            if last_day_end_time and isinstance(last_day_end_time, str):
                import re as _ret
                if _ret.match(r'^\d{2}:\d{2}$', last_day_end_time):
                    print(f"   Hora de fim do ultimo dia: '{last_day_end_time}'")
                else:
                    last_day_end_time = None
            else:
                last_day_end_time = None

            # Extrair ponto de partida (opcional)
            start_location = data.get("start_location", None)
            if start_location and not isinstance(start_location, str):
                start_location = None
            if start_location:
                print(f"   Ponto de partida: '{start_location}'")

            # Extrair localizacao de destino (rota A->B)
            end_location = data.get("end_location", None)
            if end_location and not isinstance(end_location, str):
                end_location = None
            # Fallback regex: detetar "de X a Y" / "entre X e Y" se LLM nao extraiu
            if not end_location:
                _route_patterns = [
                    r'\bde\s+([A-Za-zÀ-ÿ\s]+?)\s+(?:a|ate|para|->)\s+([A-Za-zÀ-ÿ\s]+?)(?:\s*,|\s*$|\s+\d)',
                    r'\bentre\s+([A-Za-zÀ-ÿ\s]+?)\s+e\s+([A-Za-zÀ-ÿ\s]+?)(?:\s*,|\s*$|\s+\d)',
                    r'\brota\s+(?:de\s+)?([A-Za-zÀ-ÿ\s]+?)\s+(?:ao?|ate)\s+([A-Za-zÀ-ÿ\s]+?)(?:\s*,|\s*$)',
                ]
                import re as _re2
                _DAY_NAMES_SET = set(_DAY_NUM.keys()) | {'sabado', 'domingo'}
                for pat in _route_patterns:
                    m = _re2.search(pat, user_query, _re2.IGNORECASE)
                    if m:
                        candidate_start = m.group(1).strip()
                        candidate_end   = m.group(2).strip()
                        # Ignorar se os candidatos sao nomes de dias da semana
                        cs_norm = _norm(candidate_start.split()[0])
                        ce_norm = _norm(candidate_end.split()[0])
                        if cs_norm in _DAY_NAMES_SET or ce_norm in _DAY_NAMES_SET:
                            continue
                        if len(candidate_start) > 2 and len(candidate_end) > 2:
                            if not extracted_location:
                                extracted_location = candidate_start
                            end_location = candidate_end
                            print(f"   Rota A->B detectada por regex: '{extracted_location}' -> '{end_location}'")
                            break
            if end_location:
                print(f"   Destino final: '{end_location}'")
            
            # Extrair campos em falta
            missing_fields = data.get("missing_fields", [])

            # 1. Remover campos nunca obrigatorios
            missing_fields = [f for f in missing_fields if f not in
                              ("has_children", "mobility_issues", "group_size", "num_people", "start_location")]

            # 2. Remover campos que foram extraidos com sucesso pelo LLM
            if transport_mode:
                missing_fields = [f for f in missing_fields if f != "transport_mode"]
            if extracted_location:
                missing_fields = [f for f in missing_fields if f != "location"]
            if budget_value is not None:
                missing_fields = [f for f in missing_fields if f not in ("max_cost", "budget_value")]
            if data.get("budget_type"):
                missing_fields = [f for f in missing_fields if f != "budget_type"]
            if data.get("max_time") is not None:
                missing_fields = [f for f in missing_fields if f != "max_time"]

            # 3. Se LLM devolveu null para max_time mas a duração está explícita na query, extrair
            import re as _re
            _WORD_TO_NUM = {"um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3,
                            "quatro": 4, "cinco": 5, "seis": 6, "sete": 7,
                            "oito": 8, "nove": 9, "dez": 10, "meio": 0.5}
            _DUR_PATTERN = r'\b(\d+|um|uma|dois|duas|tr[ee]s|quatro|cinco|seis|sete|oito|nove|dez|meio)\s*(dias?|horas?|semanas?|days?|hours?|weeks?|noites?|nights?|fin\s+de\s+semana|weekend)\b'
            _dur_match = _re.search(_DUR_PATTERN, user_query.lower())
            _has_explicit_duration = bool(_dur_match)

            if _has_explicit_duration and data.get("max_time") is None:
                # Parse valor da duracao diretamente da query
                _n_raw, _unit = _dur_match.group(1), _dur_match.group(2).lower()
                _n = float(_n_raw) if _n_raw.isdigit() else _WORD_TO_NUM.get(_n_raw, 1)
                if "hora" in _unit or "hour" in _unit:
                    extracted_time = int(_n * 60)
                elif "semana" in _unit or "week" in _unit:
                    extracted_time = int(_n * 7 * 480)
                else:
                    extracted_time = int(_n * 480)
                missing_fields = [f for f in missing_fields if f != "max_time"]
                print(f"   max_time extraido por regex: {extracted_time} min ({_n} {_unit})")
            elif _implicit_days and data.get("max_time") is None:
                # Duracao implicita por intervalo de dias da semana
                extracted_time = _implicit_days * 480
                missing_fields = [f for f in missing_fields if f != "max_time"]
                print(f"   max_time por intervalo de dias: {extracted_time} min ({_implicit_days} dias)")
            elif _inferred_duration_min and data.get("max_time") is None:
                # Duracao implicita por periodo do dia (manha / tarde)
                extracted_time = _inferred_duration_min
                missing_fields = [f for f in missing_fields if f != "max_time"]
                print(f"   max_time por periodo do dia: {extracted_time} min")
            elif not _has_explicit_duration and not _implicit_days and data.get("max_time") is None:
                if "max_time" not in missing_fields:
                    missing_fields.append("max_time")
                    print("   max_time adicionado: duracao nao mencionada na query")

            if budget_value is None:
                if "max_cost" not in missing_fields:
                    missing_fields.append("max_cost")
                    print("   max_cost adicionado: orcamento nao mencionado na query")

            # num_rooms: perguntar sempre que num_people > 2 e alojamento provavel
            _accom_hints = ["hotel", "hostel", "alojamento", "quarto", "noite", "noites",
                            "dormir", "ficar", "hospedado", "campismo", "airbnb"]
            _needs_accom = (num_people > 2 and
                            any(h in user_query.lower() for h in _accom_hints))
            if _needs_accom and "num_rooms" not in missing_fields:
                missing_fields.append("num_rooms")
                print(f"   num_rooms adicionado: {num_people} pessoas, alojamento provavel")

            # 4. Ordenar por prioridade
            FIELD_PRIORITY = ["location", "max_time", "max_cost", "budget_type", "transport_mode"]
            missing_fields.sort(key=lambda f: FIELD_PRIORITY.index(f) if f in FIELD_PRIORITY else 99)
            if missing_fields:
                print(f"   Campos em falta: {missing_fields}")

            print(f"   Tags extraidas pelo LLM: {valid_extracted_tags}")
            print(f"   Categorias principais (filtro): {main_categories}")
            print(f"   Tags secundarias (semantica): {secondary_tags}")

            # start_time: LLM > periodo do dia hardcoded > default 09:00
            llm_start_time = data.get("start_time", "09:00") or "09:00"
            resolved_start_time = llm_start_time if llm_start_time != "09:00" else (_inferred_start_time or "09:00")

            # end_location: default para location se nao detectado (rota de ponto unico)
            if not end_location and extracted_location:
                end_location = extracted_location

            return UserPreferences(
                max_time=extracted_time,
                max_cost=extracted_cost,
                preferred_categories=main_categories,
                category_weights=category_weights,
                start_time=resolved_start_time,
                interests=data.get("interests", []),
                secondary_tags=secondary_tags,
                location=extracted_location,
                missing_fields=missing_fields,
                transport_mode=transport_mode or "foot",
                start_location=start_location,
                mobility_issues=mobility_issues,
                num_people=num_people,
                has_children=has_children,
                last_day_end_time=last_day_end_time,
                end_location=end_location,
            )
        
        except Exception as e:
            print(f"AVISO: Erro ao extrair preferencias: {e}")
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
            print(f"AVISO: Erro ao gerar explicacao: {e}")
            return f"Esta rota foi otimizada com o algoritmo {algorithm_used} para incluir {len(route)} POIs que correspondem aos teus interesses em {', '.join(preferences.interests)}. O percurso tem uma duracao total de {total_duration} minutos e custa EUR{total_cost:.2f}."

    def interpret_refinement(self, instruction: str, current_route: List[Dict]) -> Dict:
        """
        Interpreta uma instrucao de refinamento sobre a rota existente.
        Devolve um dict com o tipo de operacao a aplicar:
          {"type": "remove",          "poi_names": [...]}
          {"type": "filter_category", "exclude_categories": [...]}
          {"type": "fresh_query"}
        """
        poi_list = "\n".join(
            f"- {p['name']} ({p.get('category', '?')})" for p in current_route
        )

        prompt = f"""Tens uma rota turistica com os seguintes POIs:
{poi_list}

O utilizador diz: "{instruction}"

Classifica a instrucao e devolve APENAS JSON (sem texto adicional):

Se o utilizador quer REMOVER um ou mais POIs especificos:
{{"type": "remove", "poi_names": ["nome exacto do POI"]}}

Se o utilizador quer EXCLUIR uma categoria inteira (ex: sem restaurantes, sem museus):
{{"type": "filter_category", "exclude_categories": ["categoria"]}}
Categorias validas: restaurantes_e_cafes, museus_e_palacios, monumentos, espacos_verdes, praias, turismo_activo, bares_e_discotecas, parques_e_reservas, arqueologia, eventos

Se a instrucao e complexa demais para modificacao directa (nova zona, novo tema, regenerar tudo):
{{"type": "fresh_query"}}

Responde APENAS com o JSON."""

        try:
            content = self._call_llm(prompt, max_tokens=200, temperature=0.1)
            content = re.sub(r'```json\s*|\s*```', '', content).strip()
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                content = content[start:end]
            return json.loads(content)
        except Exception as e:
            print(f"AVISO: Erro ao interpretar refinamento: {e}")
            return {"type": "fresh_query"}


def select_algorithm_deterministic(n_candidates: int, max_time: int) -> str:
    """
    Thresholds derivados empiricamente do benchmark 16 queries x 4 algoritmos.

    Resultados:
    - GA venceu 9/16 queries, dominante em max_time <= 1920 min (ate ~4 dias)
    - PSO venceu 7/16 queries, dominante em max_time >= 2400 min (5+ dias)
    - ACO: 0 vitorias - excluido da seleccao automatica
    - GREEDY: 0 vitorias - excluido da seleccao automatica

    Limiar de transicao GA->PSO: entre 1920 e 2400 min -> corte em 2400.
    n_candidates nao revelou padrao discriminativo - max_time e o eixo relevante.
    """
    if max_time < 2400:
        return "GA"
    else:
        return "PSO" 