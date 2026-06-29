"""
Extrai dados oficiais do campo "tabs" (Turismo de Portugal / visitportugal.com)
que existiam na fonte bruta (data/original data/*.json) mas foram removidos do
pipeline de enriquecimento sem extracao previa.

Adiciona campos novos por POI em data/portugal_todos_pois_final_enriched.json,
so quando ha conteudo real (nao cria strings vazias):
  - original_opening_hours_text
  - original_payments
  - original_accessibility
  - original_other_information
  - original_activities          (tab "Activities" -- surf/bodyboard/etc)
  - closed_days                  (lista de int, 0=Segunda...6=Domingo,
                                   extraido de "Closed: <dia>" dentro do
                                   texto de horario; excecao: POIs de
                                   alojamento sao sempre considerados abertos)

Extracao pura para os campos de horario -- nao toca em schedule.opening_time/
closing_time nem em enrichment.hours_source (esses campos nao sao usados pelo
otimizador hoje, ver route_evaluator.py; fica para uma tarefa futura).
closed_days e um campo novo, tambem ainda nao ligado ao otimizador.

Cria backup antes de escrever: data/portugal_todos_pois_final_enriched.json.bak

Uso: python scripts/extract_tp_tabs_data.py
"""

import glob
import json
import re
import shutil
from pathlib import Path

FINAL_FILE = "data/portugal_todos_pois_final_enriched.json"
ORIGINAL_GLOB = "data/original data/*.json"

# --- Normalizacao de tabname (multi-idioma) para categorias canonicas ---
TABNAME_MAP = {
    # Horario
    "Timetable": "hours",
    "Timetable and reservations": "hours",
    "時刻表": "hours",
    "時刻表と予約": "hours",
    # Pagamentos
    "Payments": "payments",
    "支払い": "payments",
    # Acessibilidade
    "Accessibility": "accessibility",
    "バリアフリー案内": "accessibility",
    # Outras informacoes
    "Other informations": "other_info",
    "その他の情報": "other_info",
    # Atividades (surf, bodyboard, etc. -- praias e turismo_activo)
    "Activities": "activities",
    "アクティビティー": "activities",
    "体験できる場所": "activities",
}

FIELD_BY_CATEGORY = {
    "hours": "original_opening_hours_text",
    "payments": "original_payments",
    "accessibility": "original_accessibility",
    "other_info": "original_other_information",
    "activities": "original_activities",
}

# Categorias de alojamento -- "visitar" nao se aplica, sao a base de
# pernoita, nao uma parada com horario. Nunca recebem closed_days.
ALWAYS_OPEN_CATEGORIES = {
    "hotelaria",
    "turismo_habitacao",
    "alojamento_local",
    "apartamento_turistico",
    "aldeamento_turistico",
    "pousadas_da_juventude",
}

WEEKDAY_NAMES = {
    "monday": 0, "mondays": 0, "segunda": 0, "segunda-feira": 0,
    "tuesday": 1, "tuesdays": 1, "terca": 1, "terça": 1,
    "terca-feira": 1, "terça-feira": 1,
    "wednesday": 2, "wednesdays": 2, "quarta": 2, "quarta-feira": 2,
    "thursday": 3, "thursdays": 3, "quinta": 3, "quinta-feira": 3,
    "friday": 4, "fridays": 4, "sexta": 4, "sexta-feira": 4,
    "saturday": 5, "saturdays": 5, "sabado": 5, "sábado": 5,
    "sabados": 5, "sábados": 5,
    "sunday": 6, "sundays": 6, "domingo": 6, "domingos": 6,
}
CLOSED_TRIGGER_RE = re.compile(r"closed|closing day|encerrad|fecha", re.IGNORECASE)
_WEEKDAY_ALTS = "|".join(sorted(WEEKDAY_NAMES.keys(), key=len, reverse=True))
WEEKDAY_RE = re.compile(r"\b(" + _WEEKDAY_ALTS + r")\b", re.IGNORECASE)

# Feriados com nome de dia da semana (data variavel, NAO e fecho semanal
# recorrente) -- remover antes de procurar dias da semana, para nao
# confundir "Easter Sunday"/"Domingo de Pascoa" com um fecho ao domingo.
HOLIDAY_EXCLUDE_RE = re.compile(
    r"easter\s+sunday|good\s+friday|palm\s+sunday|whit\s+sunday|"
    r"domingo\s+de\s+p[áa]scoa|domingo\s+de\s+ramos|sexta-feira\s+santa",
    re.IGNORECASE,
)

HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(value):
    text = HTML_TAG_RE.sub(" ", value or "")
    text = " ".join(text.split())
    return text.strip()


def extract_closed_days(text):
    """Procura um gatilho ("Closed"/"Closing Day(s)"/"Encerrado"/"Fecha") e
    devolve os dias da semana mencionados DEPOIS dele (0=Seg...6=Dom).
    Conservador: ignora dias mencionados antes do gatilho (normalmente
    descrevem dias ABERTOS, ex: "(From Tuesday to Sunday) Closed: Mondays").
    Sem gatilho -> [] (assume aberto todos os dias)."""
    if not text:
        return []
    trigger = CLOSED_TRIGGER_RE.search(text)
    if not trigger:
        return []
    after = HOLIDAY_EXCLUDE_RE.sub(" ", text[trigger.end():])
    days = {WEEKDAY_NAMES[m.group(1).lower()] for m in WEEKDAY_RE.finditer(after)}
    return sorted(days)


def load_official_tabs():
    """Carrega tabs de todos os ficheiros oficiais, indexados por entity_id.
    Junta listas de tabs quando o mesmo entity_id aparece em mais de um
    ficheiro (41 duplicados observados entre os 7 bundles)."""
    by_entity = {}
    for path in glob.glob(ORIGINAL_GLOB):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for doc in data.get("response", {}).get("docs", []):
            eid = doc.get("entity_id")
            if eid is None:
                continue
            by_entity.setdefault(int(eid), []).extend(doc.get("tabs", []) or [])
    return by_entity


def extract_categories(tabs):
    """Agrupa valores nao-vazios por categoria canonica, unindo duplicados."""
    by_cat = {}
    unmapped = set()
    for tab in tabs:
        name = tab.get("tabname", "")
        value = clean_text(tab.get("value", ""))
        if not value:
            continue
        cat = TABNAME_MAP.get(name)
        if cat is None:
            unmapped.add(name)
            continue
        by_cat.setdefault(cat, [])
        if value not in by_cat[cat]:
            by_cat[cat].append(value)
    return by_cat, unmapped


def main():
    print("A carregar tabs da fonte oficial...")
    official_tabs = load_official_tabs()
    print(f"  entity_id com tabs: {len(official_tabs)}")

    print(f"A carregar {FINAL_FILE}...")
    data = json.loads(Path(FINAL_FILE).read_text(encoding="utf-8"))
    pois = data.get("pois", data) if isinstance(data, dict) else data
    print(f"  POIs: {len(pois)}")

    backup_path = FINAL_FILE + ".bak"
    shutil.copy(FINAL_FILE, backup_path)
    print(f"  Backup criado em {backup_path}")

    field_counts = {f: 0 for f in FIELD_BY_CATEGORY.values()}
    pois_with_any_field = 0
    pois_with_closed_days = 0
    pois_skipped_alojamento = 0
    all_unmapped = set()

    for poi in pois:
        eid = poi.get("id")
        if eid is None:
            continue
        tabs = official_tabs.get(int(eid))
        if not tabs:
            continue

        by_cat, unmapped = extract_categories(tabs)
        all_unmapped.update(unmapped)

        added_any = False
        for cat, field in FIELD_BY_CATEGORY.items():
            values = by_cat.get(cat)
            if not values:
                continue
            poi[field] = " | ".join(values)
            field_counts[field] += 1
            added_any = True

        if added_any:
            pois_with_any_field += 1

        category = poi.get("category")
        if category in ALWAYS_OPEN_CATEGORIES:
            pois_skipped_alojamento += 1
        else:
            hours_text = poi.get("original_opening_hours_text", "")
            closed = extract_closed_days(hours_text)
            if closed:
                poi["closed_days"] = closed
                pois_with_closed_days += 1

    if isinstance(data, dict):
        data["pois"] = pois
        out = data
    else:
        out = pois

    Path(FINAL_FILE).write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Relatorio ===")
    for field, count in field_counts.items():
        print(f"  {field}: {count} POIs")
    print(f"  POIs com pelo menos um campo original_* novo: {pois_with_any_field}")
    print(f"  closed_days extraido: {pois_with_closed_days} POIs")
    print(f"  POIs de alojamento (excecao, nunca recebem closed_days): {pois_skipped_alojamento}")
    if all_unmapped:
        print(f"  tabnames nao reconhecidos (ignorados): {sorted(all_unmapped)}")

    print(f"\n[OK] {FINAL_FILE} actualizado.")


if __name__ == "__main__":
    main()
