import csv
import re

INPUT_FILE = "prompts_amostra_50.csv"
OUTPUT_FILE = "prompts_geradas_50.txt"

TRANSPORTE_MAP = {
    "car": "carro",
    "foot": "a pé",
    "public_transport": "transporte público",
    "public_transport_train": "comboio",
    "public_transport_bus": "autocarro",
}

PERFIL_MAP = {
    "A - Natureza & Aventura": "natureza e aventura",
    "B - Cultura & História": "cultura e história",
    "C - Gastronomia & Vinho": "gastronomia e vinho",
    "D - Sol & Praia": "sol e praia",
    "E - City Break": "city break urbano",
    "F - Família & Crianças": "família e crianças",
    "G - Luxo & Bem-estar": "luxo e bem-estar",
}

ORCAMENTO_MAP = {
    "0-20": "um orçamento muito reduzido (0€ a 20€)",
    "20-50": "um orçamento reduzido (20€ a 50€)",
    "50-100": "um orçamento baixo (50€ a 100€)",
    "100-250": "um orçamento moderado (100€ a 250€)",
    "250-500": "um orçamento razoável (250€ a 500€)",
    "500-750": "um orçamento médio (500€ a 750€)",
    "750-1000": "um orçamento médio-alto (750€ a 1000€)",
    "1000-1200": "um orçamento alto (1000€ a 1200€)",
    "1200-1400": "um orçamento alto (1200€ a 1400€)",
    "1400-1600": "um orçamento alto (1400€ a 1600€)",
    "1600-1800": "um orçamento alto (1600€ a 1800€)",
    "1800-2000": "um orçamento alto (1800€ a 2000€)",
    "2000-2500": "um orçamento muito alto (2000€ a 2500€)",
    "2500-3000": "um orçamento muito alto (2500€ a 3000€)",
    "3000-4000": "um orçamento premium (3000€ a 4000€)",
    "4000-5000": "um orçamento premium (4000€ a 5000€)",
    ">5000": "um orçamento ilimitado (mais de 5000€)",
}


def build_prompt(row):
    n = int(row["n_pessoas"])
    cidade_inicial = row["cidade_inicial"].strip()
    perfil_raw = row["perfil"].strip()
    duracao = int(row["duracao_dias"])
    orcamento_raw = row["orcamento_total"].strip()
    transporte_raw = row["modo_transporte"].strip()
    tem_criancas = row["tem_criancas"].strip().lower() == "sim"
    mobilidade = row["mobilidade_reduzida"].strip().lower() == "sim"

    # Cidades adicionais
    cidades_adicionais = []
    for key in ["cidade_adicional_1", "cidade_adicional_2", "cidade_adicional_3", "cidade_adicional_4", "cidade_adicional_5"]:
        val = row.get(key, "").strip()
        if val and val.lower() != "nenhuma":
            cidades_adicionais.append(val)

    # Mapeamentos
    transporte = TRANSPORTE_MAP.get(transporte_raw, transporte_raw)
    perfil = PERFIL_MAP.get(perfil_raw, perfil_raw)
    orcamento = ORCAMENTO_MAP.get(orcamento_raw, f"um orçamento de {orcamento_raw}€")

    # Grupo
    if n == 1:
        grupo = "Estou a viajar sozinho"
    elif n == 2:
        grupo = "Somos 2 pessoas"
    else:
        grupo = f"Somos um grupo de {n} pessoas"

    # Duração
    if duracao == 1:
        duracao_str = "1 dia"
    else:
        duracao_str = f"{duracao} dias"

    # Locais
    if cidades_adicionais:
        locais_str = f"começando em {cidade_inicial} e passando também por {', '.join(cidades_adicionais)}"
    else:
        locais_str = f"em {cidade_inicial}"

    # Condições especiais
    condicoes = []
    if tem_criancas:
        condicoes.append("viajamos com crianças")
    if mobilidade:
        condicoes.append("temos pessoas com mobilidade reduzida no grupo")

    condicoes_str = ""
    if condicoes:
        condicoes_str = f" De notar que {' e '.join(condicoes)}."

    prompt = (
        f"{grupo} e queremos fazer uma viagem de {duracao_str} {locais_str}, "
        f"com interesse em {perfil}. "
        f"Vamos deslocar-nos de {transporte} e temos {orcamento} para o grupo todo.{condicoes_str} "
        f"Podes sugerir-nos uma rota turística?"
    )

    return prompt


def main():
    with open(INPUT_FILE, newline="", encoding="latin-1") as infile:
        reader = csv.DictReader(infile, delimiter=";")
        rows = list(reader)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for row in rows:
            prompt_id = row["prompt_id"]
            perfil = row["perfil"].strip()
            prompt = build_prompt(row)
            outfile.write(f"{prompt_id}|{perfil}|{prompt}\n")

    print(f"✅ {len(rows)} prompts geradas em '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()