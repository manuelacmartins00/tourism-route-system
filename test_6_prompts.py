"""Teste das 6 prompts principais — corre localmente sem servidor."""
import sys, time
from dotenv import load_dotenv
load_dotenv()

from main_system import TourismRouteSystem

system = TourismRouteSystem()

PROMPTS = [
    {
        "id": 1,
        "query": (
            "Olá, somos um grupo de 4 pessoas que quer ir visitar coimbra durante um fim de semana grande, "
            "queremos sair à noite e vida noturna e durante os dias visitar os principais pontos turisticos e culturais. "
            "alojamento incluído. refeições: trato eu. o orçamento é 100 euros por pessoa por dia. vamos de carro."
        ),
        "include_accommodation": True,
        "include_meals": False,
    },
    {
        "id": 2,
        "query": (
            "Somos 3 amigos e queremos passar um fim de semana em Lisboa a ver museus, monumentos históricos "
            "e jantar bem. Temos sexta à noite livre. alojamento: trato eu. refeições incluídas. "
            "o orçamento é 50 euros por pessoa por dia. vamos de transportes públicos."
        ),
        "include_accommodation": False,
        "include_meals": True,
    },
    {
        "id": 3,
        "query": (
            "Eu e a minha namorada queremos ir ao Algarve. Gostamos de praia, natureza e sítios tranquilos "
            "para relaxar. refeições incluídas. a duração é 5 dias. "
            "o orçamento é 100 euros por dia para o grupo. vamos de carro."
        ),
        "include_accommodation": True,
        "include_meals": True,
    },
    {
        "id": 4,
        "query": (
            "Somos uma família com 2 crianças pequenas e queremos fazer uma viagem de 4 dias em Portugal. "
            "O orçamento é de 800 euros no total. Gostamos de parques, zoos e atividades ao ar livre. "
            "alojamento incluído. refeições incluídas. vamos de carro."
        ),
        "include_accommodation": True,
        "include_meals": True,
    },
    {
        "id": 5,
        "query": (
            "Quero fazer uma road trip de uma semana pelo norte de Portugal, começando no Porto e passando "
            "por Braga e Guimarães. Gosto de arquitetura, história e boa comida. Vou de carro. "
            "alojamento: trato eu. refeições incluídas. o orçamento é 60 euros por pessoa por dia."
        ),
        "include_accommodation": False,
        "include_meals": True,
    },
    {
        "id": 6,
        "query": (
            "Somos um grupo de 6 pessoas que quer celebrar um aniversário em Coimbra — vida noturna, "
            "jantares e alguns pontos culturais durante o dia. Vamos de transportes públicos. "
            "refeições: trato eu. a duração é 1 dia. o orçamento é 80 euros por pessoa por dia."
        ),
        "include_accommodation": False,
        "include_meals": False,
    },
]

print("\n" + "="*70)
print("TESTE DAS 6 PROMPTS")
print("="*70)

results = []
for p in PROMPTS:
    print(f"\n>>> Prompt {p['id']}")
    t0 = time.time()
    try:
        result = system.plan_route(
            p["query"],
            use_shap=False,
            verbose=False,
            include_accommodation=p["include_accommodation"],
            include_meals=p["include_meals"],
            generate_map=False,
        )
        elapsed = time.time() - t0
        status = result.get("status", "ok")
        if status in ("needs_clarification", "needs_scope_clarification"):
            print(f"   CLARIFICACAO NECESSARIA: {result.get('missing_fields')} / {result.get('scope_questions')}")
            results.append({"id": p["id"], "status": "clarification", "elapsed": elapsed})
        elif "error" in result:
            print(f"   ERRO: {result['error']}")
            results.append({"id": p["id"], "status": "error", "elapsed": elapsed})
        else:
            algo   = result.get("algorithm_used", "?")
            n_pois = len(result.get("route", []))
            opt    = result.get("optimization", {})
            fitness = opt.get("fitness", 0)
            cost_pp = result.get("cost_per_person", 0)
            max_cost = result.get("preferences", {}).get("max_cost", 0)
            n_days  = len(result.get("day_plan", {}).get("days", []))
            cats = list({p["category"] for p in result.get("route", [])
                         if p["category"] not in ("hotelaria","alojamento_local","turismo_habitacao",
                                                   "turismo_espaco_rural","apartamento_turistico",
                                                   "pousadas_da_juventude","aldeamento_turistico",
                                                   "parques_de_campismo")})
            print(f"   OK | algo={algo} | n={n_pois} POIs | fitness={fitness:.1f} | "
                  f"€{cost_pp:.0f}/€{max_cost:.0f}pp | {n_days} dias | {elapsed:.0f}s")
            print(f"   categorias: {cats}")
            results.append({"id": p["id"], "status": "ok", "algo": algo,
                             "n_pois": n_pois, "fitness": fitness,
                             "cost_pp": cost_pp, "max_cost": max_cost,
                             "n_days": n_days, "elapsed": elapsed, "cats": cats})
    except Exception as e:
        elapsed = time.time() - t0
        print(f"   EXCECAO: {e}")
        results.append({"id": p["id"], "status": "exception", "error": str(e), "elapsed": elapsed})

print("\n" + "="*70)
print("RESUMO")
print("="*70)
for r in results:
    if r["status"] == "ok":
        over = "OVER" if r["cost_pp"] > r["max_cost"] * 1.05 else "OK"
        print(f"  P{r['id']}: {r['algo']:6} | {r['n_pois']:2} POIs | fitness={r['fitness']:.1f} | "
              f"€{r['cost_pp']:.0f}/€{r['max_cost']:.0f}pp {over} | {r['n_days']}d | {r['elapsed']:.0f}s")
    else:
        print(f"  P{r['id']}: {r['status'].upper()} — {r.get('error','')}")
