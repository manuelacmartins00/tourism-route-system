"""
Reintegra os 89 POIs reais que foram dropados do final_enriched durante
o pipeline de processamento. Os POIs ja tinham sido enriched com OSM e Google
(campos geos_geocoords, schedule_google, price_level_google, google_place_id)
mas nao foram convertidos para o formato final_enriched.

Exclui os 26 POIs do bundle 'sugestoes' (conteudo editorial, nao sao POIs reais).

Uso: python scripts/reintegrate_missing_pois.py
"""

import json
from datetime import datetime
from pathlib import Path

FINAL_FILE  = "data/portugal_todos_pois_final_enriched.json"
SOURCE_FILE = "data/portugal_todos_pois_limpos_osm_and_google_enriched.json"

PRICE_MAP = {
    "restaurantes_e_cafes":  {0: 0,  1: 12,  2: 25,  3: 45,  4: 80},
    "bares_e_discotecas":    {0: 0,  1: 8,   2: 15,  3: 30,  4: 60},
    "hotelaria":             {0: 0,  1: 50,  2: 90,  3: 150, 4: 250},
    "turismo_espaco_rural":  {0: 0,  1: 40,  2: 80,  3: 130, 4: 200},
    "turismo_activo":        {0: 0,  1: 20,  2: 40,  3: 70,  4: 120},
    "turismo_habitacao":     {0: 0,  1: 40,  2: 80,  3: 130, 4: 200},
    "alojamento_local":      {0: 0,  1: 40,  2: 80,  3: 130, 4: 200},
    "aldeamento_turistico":  {0: 0,  1: 40,  2: 80,  3: 130, 4: 200},
    "apartamento_turistico": {0: 0,  1: 40,  2: 80,  3: 130, 4: 200},
    "museus_e_palacios":     {0: 0,  1: 5,   2: 10,  3: 20,  4: 35},
    "praias":                {0: 0,  1: 0,   2: 5,   3: 15,  4: 30},
    "parques_de_diversao":   {0: 0,  1: 10,  2: 20,  3: 35,  4: 60},
    "default":               {0: 0,  1: 10,  2: 25,  3: 50,  4: 100},
}

DURATION_DEFAULTS = {
    "restaurantes_e_cafes":   60,
    "hotelaria":              480,
    "turismo_espaco_rural":   480,
    "monumentos":             45,
    "turismo_activo":         120,
    "praias":                 120,
    "bares_e_discotecas":     120,
    "museus_e_palacios":      90,
    "turismo_habitacao":      480,
    "alojamento_local":       480,
    "apartamento_turistico":  480,
    "parques_de_campismo":    480,
    "aldeamento_turistico":   480,
    "espacos_verdes":         60,
    "parques_e_reservas":     90,
    "arqueologia":            45,
    "marinas_e_portos":       45,
    "termas":                 120,
    "parques_de_diversao":    180,
    "zoos_e_aquarios":        120,
    "ciencia_e_conhecimento": 90,
    "casinos":                120,
    "campos":                 180,
    "eventos":                120,
}

def parse_coords(p):
    gc = p.get("geos_geocoords", "")
    if gc and gc != "None":
        parts = gc.split(",")
        if len(parts) == 2:
            try:
                return float(parts[0].strip()), float(parts[1].strip())
            except ValueError:
                pass
    c = p.get("coordenadas", {})
    if isinstance(c, dict) and "coordinates" in c:
        lon, lat = c["coordinates"]
        return float(lat), float(lon)
    return None, None

def convert(raw, existing_ids):
    entity_id = int(raw.get("entity_id", 0))
    if entity_id in existing_ids:
        return None

    lat, lon = parse_coords(raw)
    if lat is None:
        return None

    bundle = raw.get("bundle", "outros")
    name   = raw.get("ts_poi_nome", "").strip()
    desc   = raw.get("ts_poi_descritivo", "").strip()
    region = raw.get("regiao_origem", "")
    url    = raw.get("url", "")

    # Schedule
    sched_raw = raw.get("schedule_google")
    if sched_raw and sched_raw != "None" and isinstance(sched_raw, dict):
        opening = sched_raw.get("opening_time", "09:00")
        closing = sched_raw.get("closing_time", "18:00")
        hours_source = "google"
    else:
        opening, closing = "09:00", "18:00"
        hours_source = "default"

    # Price level
    pl = raw.get("price_level_google")
    pl_int = None
    if pl is not None and pl != "None":
        try:
            pl_int = int(pl)
        except (ValueError, TypeError):
            pass

    if pl_int is not None:
        mapping  = PRICE_MAP.get(bundle, PRICE_MAP["default"])
        cost     = float(mapping.get(pl_int, 10))
        cost_src = "google_price_level"
    else:
        cost     = 5.0
        cost_src = "default"

    duration     = DURATION_DEFAULTS.get(bundle, 60)
    google_pid   = raw.get("google_place_id")
    if google_pid == "None":
        google_pid = None
    google_matched = google_pid is not None

    return {
        "id":          entity_id,
        "name":        name,
        "category":    bundle,
        "description": desc,
        "location": {"lat": lat, "lon": lon},
        "attributes": {
            "score":            0.7,
            "cost_euros":       cost,
            "duration_minutes": duration,
        },
        "schedule": {
            "opening_time": opening,
            "closing_time": closing,
        },
        "source": {
            "bundle": bundle,
            "region": region,
            "url":    url,
        },
        "enrichment": {
            "timestamp":     datetime.utcnow().isoformat(),
            "osm_matched":   False,
            "google_matched": google_matched,
            "hours_source":  hours_source,
            "cost_source":   cost_src,
            "score_source":  "heuristic",
            "google_place_id": google_pid,
        },
        "price_level_google": pl_int,
    }


def main():
    print(f"A carregar {FINAL_FILE}...")
    data = json.loads(Path(FINAL_FILE).read_text(encoding="utf-8"))
    pois = data.get("pois", data) if isinstance(data, dict) else data
    existing_ids = {int(p["id"]) for p in pois}
    print(f"  POIs actuais: {len(pois)}")

    print(f"A carregar {SOURCE_FILE}...")
    raw_list = json.loads(Path(SOURCE_FILE).read_text(encoding="utf-8"))
    if isinstance(raw_list, dict):
        raw_list = raw_list.get("pois", raw_list)

    ids_in_final = {str(p["id"]) for p in pois}
    candidates   = [p for p in raw_list
                    if str(p.get("entity_id","")) not in ids_in_final]
    print(f"  Candidatos a reintegrar: {len(candidates)}")

    converted = []
    skipped   = 0
    for raw in candidates:
        poi = convert(raw, existing_ids)
        if poi:
            converted.append(poi)
        else:
            skipped += 1

    print(f"  Convertidos com sucesso: {len(converted)}")
    print(f"  Saltados (sem coords ou ID duplicado): {skipped}")

    pois.extend(converted)

    if isinstance(data, dict):
        data["pois"] = pois
        out = data
    else:
        out = pois

    Path(FINAL_FILE).write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n[OK] {FINAL_FILE} actualizado: {len(pois)} POIs total (+{len(converted)} reintegrados)")


if __name__ == "__main__":
    main()
