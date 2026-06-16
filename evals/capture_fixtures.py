"""
evals/capture_fixtures.py
=========================
Corre o pipeline LLM -> RAG para cada prompt do ficheiro de benchmark e
guarda o estado pré-optimizador como um fixture JSON em
data/bench_fixtures/<scenario_id>.json.

Os fixtures capturam exactamente os inputs que chegam ao optimizador em
produção (POIs filtrados, matriz de distâncias, user_prefs), pelo que o
benchmark de algoritmos subsequente é 100% independente de LLM e RAG.

Uso:
    export GROQ_API_KEY="gsk_..."
    python evals/capture_fixtures.py \
        --prompts evals/prompts_70_stratified.txt \
        --out     data/bench_fixtures \
        [--delay 3]   # segundos entre calls ao Groq (default 3)
"""
import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path

# Permite importar o projecto sem instalar
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_system import TourismRouteSystem


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="evals/prompts_70_stratified.txt")
    p.add_argument("--out",     default="data/bench_fixtures")
    p.add_argument("--delay",   type=float, default=3.0,
                   help="Segundos entre calls ao Groq (evitar rate-limit)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Não voltar a capturar fixtures já existentes")
    return p.parse_args()


def load_prompts(path: str):
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 2)
            if len(parts) == 3:
                prompts.append({"id": parts[0].strip(),
                                 "profile": parts[1].strip(),
                                 "query": parts[2].strip()})
    return prompts


def main():
    args = parse_args()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY não definida")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts)
    print(f"[capture_fixtures] {len(prompts)} prompts em '{args.prompts}'")
    print(f"[capture_fixtures] Output: {out_dir}\n")

    system = TourismRouteSystem(api_key=api_key)

    ok = skip = err = 0
    for i, entry in enumerate(prompts, 1):
        sid    = entry["id"]
        query  = entry["query"]
        out_fp = out_dir / f"{sid}.json"

        if args.skip_existing and out_fp.exists():
            print(f"[{i:02d}/{len(prompts)}] {sid} — já existe, skip")
            skip += 1
            continue

        print(f"[{i:02d}/{len(prompts)}] {sid}  ({entry['profile']})  {query[:60]}...")
        try:
            result = system.plan_route(
                query,
                use_shap=False,
                verbose=False,
                generate_map=False,
                generate_explanation=False,
                include_accommodation=False,
                include_meals=False,
                fixture_capture_path=str(out_fp),
            )

            if out_fp.exists():
                # Enriquecer o fixture com metadados do prompt
                with open(out_fp, encoding="utf-8") as f:
                    fixture = json.load(f)
                fixture["scenario_id"] = sid
                fixture["profile"]     = entry["profile"]
                fixture["query"]       = query
                with open(out_fp, "w", encoding="utf-8") as f:
                    json.dump(fixture, f, ensure_ascii=False, indent=2)
                print(f"         -> {fixture.get('n_pois', '?')} POIs, algo={fixture.get('selected_algo', '?')}")
                ok += 1
            else:
                # Sem fixture: provavelmente needs_clarification ou erro de RAG
                status = result.get("status", "") or result.get("error", "sem fixture")
                print(f"         -> SKIP ({status})")
                skip += 1

        except Exception as e:
            print(f"         -> ERRO: {e}")
            traceback.print_exc()
            err += 1

        if i < len(prompts):
            time.sleep(args.delay)

    print(f"\n[capture_fixtures] Concluído: {ok} OK  |  {skip} skip  |  {err} erros")
    print(f"[capture_fixtures] Fixtures em: {out_dir}")


if __name__ == "__main__":
    main()
