# apply_google_price_levels.py
#
# Converte price_level_google (0-4) em cost_euros para os POIs que tem
# este campo preenchido, usando tabelas de mapeamento por bundle.
# So actualiza POIs com price_level_google nao-nulo.
# Guarda o resultado no mesmo ficheiro (ou num ficheiro novo com --output).
#
# Uso:
#   python apply_google_price_levels.py
#   python apply_google_price_levels.py --input data/portugal_todos_pois_final_enriched.json
#   python apply_google_price_levels.py --output data/portugal_todos_pois_final_enriched_v2.json

import json, argparse
from pathlib import Path

# -- Mapeamento price_level_google (0-4) -> cost_euros por bundle ------
# 0 = gratuito, 1 = barato, 2 = moderado, 3 = caro, 4 = muito caro
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
    "eventos":               {0: 0,  1: 10,  2: 25,  3: 50,  4: 100},
    "default":               {0: 0,  1: 10,  2: 25,  3: 50,  4: 100},
}

def get_bundle(poi: dict) -> str:
    return (
        poi.get("category") or
        poi.get("source", {}).get("bundle") or
        "default"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/portugal_todos_pois_final_enriched.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path  = args.input
    output_path = args.output or input_path  # sobrescreve por defeito

    print(f"\nA carregar {input_path}...")
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    pois = data if isinstance(data, list) else data.get("pois", [])

    updated = skipped = no_map = 0

    for poi in pois:
        pl = poi.get("price_level_google")
        if pl is None:
            skipped += 1
            continue

        bundle  = get_bundle(poi)
        mapping = PRICE_MAP.get(bundle, PRICE_MAP["default"])
        cost    = mapping.get(int(pl))

        if cost is None:
            no_map += 1
            continue

        # Actualizar cost_euros dentro de attributes (formato normalizado)
        if "attributes" in poi:
            poi["attributes"]["cost_euros"] = float(cost)
            poi["attributes"]["cost_source"] = "google_price_level"
        else:
            # Formato MongoDB raw - campo directo
            poi["cost_euros"] = float(cost)

        # Registar fonte no enrichment se existir
        if "enrichment" in poi:
            poi["enrichment"]["cost_source"] = "google_price_level"

        updated += 1

    # Guardar
    if isinstance(data, dict):
        data["pois"] = pois
    else:
        data = pois

    Path(output_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"[OK] CONCLUIDO")
    print(f"   Actualizados: {updated}")
    print(f"   Sem price_level (saltados): {skipped}")
    print(f"   Sem mapeamento: {no_map}")
    print(f"   Output: {output_path}\n")

if __name__ == "__main__":
    main()