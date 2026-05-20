"""
evals/generate_prompts.py
Gera 500 prompts de linguagem natural a partir das combinações de features do Excel.

Fluxo:
  1. Lê Folha1 do Excel (2000 combinações)
  2. Amostra estratificada: 71-72 por perfil x 7 perfis ≈ 500
  3. Para cada combinação, chama Groq (llama-3.1-8b-instant) para gerar um prompt
  4. Guarda em evals/prompts_500_llm.txt (formato: prompt_id|perfil|prompt)

Uso:
  python evals/generate_prompts.py
  python evals/generate_prompts.py --output evals/prompts_500_llm.txt --throttle 2.5
"""

import os, sys, time, argparse, random
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import openpyxl
except ImportError:
    print("ERRO: pip install openpyxl")
    sys.exit(1)

from groq import Groq

EXCEL_PATH = "data/Prompts_teste_standard.xlsx"

PERFIL_INTERESSES = {
    "A - Natureza & Aventura":   "natureza, aventura, desporto ao ar livre, trilhos, paisagens naturais",
    "B - Vida Noturna":          "vida noturna, bares, música, entretenimento, restaurantes noturnos",
    "C - Cultural & Histórico":  "cultura, história, monumentos, museus, arqueologia, património",
    "D - Família":               "família, atividades para crianças, parques, diversão, segurança",
    "E - Bem-estar":             "bem-estar, spa, termas, relaxamento, natureza, saúde",
    "F - Gastronomia":           "gastronomia, restaurantes, vinhos, culinária local, mercados",
    "G - Road Trip":             "road trip, paisagens, múltiplos destinos, fotografia, liberdade",
}

TRANSPORTE_PT = {
    "foot":                    "a pé",
    "car":                     "de carro",
    "public_transport":        "em transportes públicos",
    "public_transport_train":  "de comboio",
}

ORCAMENTO_DESC = {
    "0-20":       "muito reduzido (0€ a 20€)",
    "20-50":      "muito baixo (20€ a 50€)",
    "50-100":     "baixo (50€ a 100€)",
    "100-250":    "baixo-médio (100€ a 250€)",
    "250-500":    "médio (250€ a 500€)",
    "500-750":    "médio (500€ a 750€)",
    "750-1000":   "médio-alto (750€ a 1000€)",
    "1000-1200":  "alto (1000€ a 1200€)",
    "1200-1400":  "alto (1200€ a 1400€)",
    "1400-1600":  "alto (1400€ a 1600€)",
    "1600-1800":  "alto (1600€ a 1800€)",
    "1800-2000":  "muito alto (1800€ a 2000€)",
    "2000-2500":  "muito alto (2000€ a 2500€)",
    "2500-3000":  "muito alto (2500€ a 3000€)",
    "3000-4000":  "premium (3000€ a 4000€)",
    "4000-5000":  "premium (4000€ a 5000€)",
    ">5000":      "sem limite (mais de 5000€)",
}


def load_combinations(excel_path: str) -> list:
    wb = openpyxl.load_workbook(excel_path)
    ws = wb["Folha1"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    combos = []
    for row in rows:
        if not row[0]:
            continue
        cidades_extra = [str(c).strip() for c in row[3:8] if c and str(c).strip() not in ("Nenhuma", "None", "")]
        combos.append({
            "prompt_id":              str(row[0]),
            "n_pessoas":              row[1],
            "cidade_inicial":         str(row[2]).strip(),
            "cidades_extra":          cidades_extra,
            "perfil":                 str(row[8]).strip(),
            "duracao_dias":           row[9],
            "orcamento":              str(row[10]).strip() if row[10] else "250-500",
            "transporte":             str(row[11]).strip() if row[11] else "car",
            "tem_criancas":           str(row[12]).strip() if row[12] else "Não",
            "mobilidade":             str(row[13]).strip() if row[13] else "Não",
            "include_accommodation":  bool(row[14]) if row[14] is not None else True,
            "include_meals":          bool(row[15]) if row[15] is not None else True,
        })
    return combos


def stratified_sample(combos: list, n_per_profile: int = 72, seed: int = 123) -> list:
    random.seed(seed)
    by_profile = defaultdict(list)
    for c in combos:
        by_profile[c["perfil"]].append(c)
    sample = []
    for perfil, items in sorted(by_profile.items()):
        chosen = random.sample(items, min(n_per_profile, len(items)))
        sample.extend(chosen)
    return sample


def adjust_budget(orcamento_key: str, include_accommodation: bool, include_meals: bool) -> str:
    """
    Ajusta o orçamento conforme as features:
    - include_accommodation=False → x0.4 (alojamento não incluído, ~60% do budget seria alojamento)
    - include_meals=True          → x0.9 (refeições incluídas consomem ~10% do budget)
    """
    # Extrair valor médio do range
    key = orcamento_key.strip()
    if key.startswith(">"):
        mid = 6000.0
    else:
        parts = key.split("-")
        try:
            mid = (float(parts[0]) + float(parts[1])) / 2
        except Exception:
            mid = 500.0

    if not include_accommodation:
        mid *= 0.4   # sem alojamento: só ~40% do budget vai para atividades
    if not include_meals:
        mid *= 0.9   # sem refeições: poupa ~10% do budget

    # Formatar de volta como descrição legível
    if mid < 30:    return f"muito reduzido (até {int(mid)}€)"
    if mid < 75:    return f"baixo (até {int(mid)}€)"
    if mid < 200:   return f"baixo-médio (até {int(mid)}€)"
    if mid < 400:   return f"médio (até {int(mid)}€)"
    if mid < 700:   return f"médio-alto (até {int(mid)}€)"
    if mid < 1200:  return f"alto (até {int(mid)}€)"
    if mid < 3000:  return f"muito alto (até {int(mid)}€)"
    return f"premium (até {int(mid)}€)"


def build_llm_prompt(combo: dict) -> str:
    interesses = PERFIL_INTERESSES.get(combo["perfil"], combo["perfil"])
    transporte  = TRANSPORTE_PT.get(combo["transporte"], combo["transporte"])
    incl_aloj  = combo.get("include_accommodation", True)
    incl_refei = combo.get("include_meals", True)
    orcamento   = adjust_budget(combo["orcamento"], incl_aloj, incl_refei)
    n = combo["n_pessoas"]
    dias = combo["duracao_dias"]
    cidade = combo["cidade_inicial"]
    extras = combo["cidades_extra"]
    criancas = combo["tem_criancas"].lower() in ("sim", "yes")
    mobilidade = combo["mobilidade"].lower() in ("sim", "yes")

    if extras:
        destinos = f"começando em {cidade} e passando também por {', '.join(extras)}"
    else:
        destinos = f"em {cidade}"

    contexto_extra = []
    if criancas:
        contexto_extra.append("viajamos com crianças")
    if mobilidade:
        contexto_extra.append("temos pessoas com mobilidade reduzida no grupo")
    if not incl_aloj:
        contexto_extra.append("tratamos o alojamento por conta própria")
    if not incl_refei:
        contexto_extra.append("não precisamos de sugestões de refeições")

    contexto_str = (". De notar que " + " e ".join(contexto_extra)) if contexto_extra else ""

    n_str = "sozinho" if n == 1 else f"um grupo de {n} pessoas"

    return f"""Gera UMA query turística em português de Portugal, escrita como se fosse um utilizador real a pedir sugestões de rota.

DADOS DA VIAGEM:
- Viajante: {n_str}
- Destino: {destinos}
- Duração: {dias} dias
- Interesses: {interesses}
- Transporte: {transporte}
- Orçamento total do grupo: {orcamento}
{('- Contexto especial: ' + '; '.join(contexto_extra)) if contexto_extra else ''}

REGRAS OBRIGATÓRIAS:
- Escreve em português de Portugal, tom natural e conversacional
- 3-5 frases, entre 80 e 150 palavras
- OBRIGATÓRIO mencionar explicitamente o número de dias (ex: "viagem de {dias} dias", "durante {dias} dias")
- OBRIGATÓRIO mencionar o destino, o orçamento aproximado e o modo de transporte
- Menciona os interesses/atividades pretendidas de forma natural
- Não uses linguagem técnica nem menciones algoritmos ou sistemas
- Varia o fraseado (não uses sempre "Somos um grupo de X pessoas")
- Termina com um pedido de sugestão de rota{contexto_str}

Responde APENAS com o texto da query, sem introduções ou explicações."""


def generate_prompt_llm(client: Groq, combo: dict) -> str:
    llm_prompt = build_llm_prompt(combo)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": llm_prompt}],
        model="llama-3.1-8b-instant",
        max_tokens=150,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default="evals/prompts_500_llm.txt")
    parser.add_argument("--throttle", type=float, default=2.5, help="Segundos entre calls Groq")
    parser.add_argument("--per-profile", type=int, default=72, help="Amostras por perfil (72x7=504)")
    parser.add_argument("--seed",     type=int, default=123)
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERRO: GROQ_API_KEY não definida no .env")
        sys.exit(1)

    client = Groq(api_key=api_key)

    print("A carregar combinações do Excel...")
    combos = load_combinations(EXCEL_PATH)
    print(f"  {len(combos)} combinações carregadas")

    sample = stratified_sample(combos, n_per_profile=args.per_profile, seed=args.seed)
    print(f"  {len(sample)} amostras estratificadas ({args.per_profile} por perfil)")

    from collections import Counter
    for perfil, count in sorted(Counter(c["perfil"] for c in sample).items()):
        print(f"    {count}  {perfil}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Retomar se ficheiro já existe parcialmente
    done_ids = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    parts = line.split("|", 2)
                    if parts:
                        done_ids.add(parts[0].strip())
        print(f"\n  Resume: {len(done_ids)} prompts já gerados")

    total = len(sample)
    errors = 0

    print(f"\nA gerar {total - len(done_ids)} prompts via Groq (throttle={args.throttle}s)...\n")

    with open(output_path, "a", encoding="utf-8") as out:
        if not done_ids:
            out.write("# Prompts gerados por LLM a partir de combinações estratificadas\n")
            out.write(f"# {total} amostras | {args.per_profile} por perfil | seed={args.seed}\n")
            out.write("# formato: prompt_id|perfil|include_accommodation|include_meals|prompt\n")

        for i, combo in enumerate(sample):
            pid = combo["prompt_id"]
            if pid in done_ids:
                continue

            perfil     = combo["perfil"]
            incl_aloj  = combo.get("include_accommodation", True)
            incl_meals = combo.get("include_meals", True)
            print(f"[{i+1}/{total}] P{pid.zfill(4)} | {perfil[:30]}", end="  ", flush=True)

            try:
                prompt_text = generate_prompt_llm(client, combo)
                prompt_text = prompt_text.replace("\n", " ").replace("|", "/")
                out.write(f"{pid}|{perfil}|{incl_aloj}|{incl_meals}|{prompt_text}\n")
                out.flush()
                print(f"OK ({len(prompt_text)} chars)")
            except Exception as e:
                errors += 1
                print(f"ERRO: {e}")

            time.sleep(args.throttle)

    print(f"\nConcluído: {total - errors} prompts gerados")
    print(f"Ficheiro: {output_path}")
    if errors:
        print(f"Erros: {errors} (podes re-correr para retomar)")


if __name__ == "__main__":
    main()
