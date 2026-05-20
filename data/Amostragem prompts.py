import pandas as pd
import numpy as np
from itertools import product

np.random.seed(123)

n_pessoas = [1, 2, 3, 4, 5, 6, 8, 10]

cidades = ["Aveiro", "Beja", "Braga", "Bragança", "Castelo Branco", "Coimbra",
           "Évora", "Faro", "Guarda", "Leiria", "Lisboa", "Portalegre", "Porto",
           "Santarém", "Setúbal", "Viana do Castelo", "Vila Real", "Viseu",
           "Ponta Delgada", "Angra do Heroísmo", "Horta", "Funchal", "Porto Santo", "São Miguel"]

cidades_adicional = cidades + ["Nenhuma"]

perfis = ["A - Natureza & Aventura", "B - Vida Noturna", "C - Cultural & Histórico",
          "D - Família", "E - Bem-estar", "F - Gastronomia", "G - Road Trip"]

duracoes = [1, 2, 3, 4, 5, 7, 9, 10, 12, 14, 21]

orcamentos = ["0-20", "20-50", "50-100", "100-250", "250-500", "500-750",
              "750-1000", "1000-1200", "1200-1400", "1400-1600", "1600-1800",
              "1800-2000", "2000-2500", "2500-3000", "3000-4000", "4000-5000", ">5000"]

modos = ["foot", "car", "public_transport_train", "public_transport_bus", "public_transport"]

tem_criancas = ["Sim", "Não"]

mobilidade_reduzida = ["Sim", "Não"]

N = 50
rows = []

# Estratificação: garantir cobertura uniforme das variáveis principais
# Cada perfil tem ~7 entradas (50/7)
n_por_perfil = N // len(perfis)

for perfil in perfis:
    for _ in range(n_por_perfil):
        cidade_ini = np.random.choice(cidades)
        # Cidades adicionais — maioria "Nenhuma" (realista)
        def pick_cidade_adicional(excluir):
            if np.random.random() < 0.6:
                return "Nenhuma"
            opcoes = [c for c in cidades if c != excluir]
            return np.random.choice(opcoes)

        ca1 = pick_cidade_adicional(cidade_ini)
        ca2 = pick_cidade_adicional(cidade_ini) if ca1 != "Nenhuma" and np.random.random() < 0.4 else "Nenhuma"
        ca3 = pick_cidade_adicional(cidade_ini) if ca2 != "Nenhuma" and np.random.random() < 0.2 else "Nenhuma"
        ca4 = pick_cidade_adicional(cidade_ini) if ca3 != "Nenhuma" and np.random.random() < 0.1 else "Nenhuma"
        ca5 = pick_cidade_adicional(cidade_ini) if ca4 != "Nenhuma" and np.random.random() < 0.05 else "Nenhuma"

        rows.append({
            "prompt_id": None,
            "n_pessoas": np.random.choice(n_pessoas),
            "cidade_inicial": cidade_ini,
            "cidade_adicional_1": ca1,
            "cidade_adicional_2": ca2,
            "cidade_adicional_3": ca3,
            "cidade_adicional_4": ca4,
            "cidade_adicional_5": ca5,
            "perfil": perfil,
            "duracao_dias": np.random.choice(duracoes),
            "orcamento_total": np.random.choice(orcamentos),
            "modo_transporte": np.random.choice(modos),
            "tem_criancas": np.random.choice(tem_criancas),
            "mobilidade_reduzida": np.random.choice(mobilidade_reduzida),
        })

# Completar até 50 com amostragem aleatória pura
while len(rows) < N:
    cidade_ini = np.random.choice(cidades)
    rows.append({
        "prompt_id": None,
        "n_pessoas": np.random.choice(n_pessoas),
        "cidade_inicial": cidade_ini,
        "cidade_adicional_1": np.random.choice(cidades_adicional),
        "cidade_adicional_2": "Nenhuma",
        "cidade_adicional_3": "Nenhuma",
        "cidade_adicional_4": "Nenhuma",
        "cidade_adicional_5": "Nenhuma",
        "perfil": np.random.choice(perfis),
        "duracao_dias": np.random.choice(duracoes),
        "orcamento_total": np.random.choice(orcamentos),
        "modo_transporte": np.random.choice(modos),
        "tem_criancas": np.random.choice(tem_criancas),
        "mobilidade_reduzida": np.random.choice(mobilidade_reduzida),
    })

df = pd.DataFrame(rows[:N])
df["prompt_id"] = range(1, N + 1)
cols = ["prompt_id", "n_pessoas", "cidade_inicial",
        "cidade_adicional_1", "cidade_adicional_2", "cidade_adicional_3",
        "cidade_adicional_4", "cidade_adicional_5",
        "perfil", "duracao_dias", "orcamento_total",
        "modo_transporte", "tem_criancas", "mobilidade_reduzida"]
df = df[cols]

df.to_excel("prompts_amostra_50.xlsx", index=False)
print(f"Total entradas: {len(df)}")
print(f"\nDistribuição por perfil:\n{df['perfil'].value_counts()}")
print(f"\nDistribuição por modo:\n{df['modo_transporte'].value_counts()}")
print(f"\nDistribuição por duração:\n{df['duracao_dias'].value_counts().sort_index()}")