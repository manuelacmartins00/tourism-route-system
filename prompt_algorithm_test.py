"""
benchmark_prompts.py
Lê prompts de um ficheiro TXT (formato: prompt_id|perfil|prompt)
e corre cada uma no TourismRouteSystem, guardando os resultados.

Uso:
  python benchmark_prompts.py --input prompts_geradas_2000.txt
  python benchmark_prompts.py --input prompts_geradas_200.txt --mapas
  python benchmark_prompts.py --input prompts_geradas_2000.txt --resume
"""

import os, sys, json, time, argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))
from main_system import TourismRouteSystem


def load_prompts(txt_path: str):
    with open(txt_path, encoding="utf-8") as f:
        linhas = [l.strip() for l in f if l.strip()]
    prompts = []
    for linha in linhas:
        partes = linha.split("|", 2)
        if len(partes) == 3:
            prompts.append({
                "prompt_id": partes[0].strip(),
                "perfil":    partes[1].strip(),
                "prompt":    partes[2].strip(),
            })
    return prompts


def save_result(result: dict, output_dir: Path, prompt_id: str, gerar_mapa: bool):
    result_dir = output_dir / f"P{prompt_id.zfill(4)}"
    result_dir.mkdir(parents=True, exist_ok=True)

    result_to_save = json.loads(json.dumps(result, default=str))
    result_to_save.get("optimization", {}).pop("fitness_history", None)
    result_to_save.pop("shap_explanation", None)
    result_to_save.pop("day_plan", None)

    if not gerar_mapa:
        result_to_save.pop("map_file", None)

    with open(result_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result_to_save, f, indent=2, ensure_ascii=False)

    if gerar_mapa and result.get("map_file"):
        map_src = Path(result["map_file"])
        if map_src.exists():
            map_dst = result_dir / "map.html"
            map_dst.write_bytes(map_src.read_bytes())


def save_summary_line(summary_path: Path, entry: dict):
    write_header = not summary_path.exists()
    with open(summary_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("prompt_id|perfil|algoritmo|fitness|n_pois|visit_min|total_min|custo|status|elapsed_s\n")
        f.write(
            f"{entry['prompt_id']}|{entry['perfil']}|{entry['algoritmo']}|"
            f"{entry['fitness']}|{entry['n_pois']}|{entry['visit_min']}|"
            f"{entry['total_min']}|{entry['custo']}|{entry['status']}|{entry['elapsed_s']}\n"
        )


def load_done(output_dir: Path) -> set:
    """Carrega IDs já processados para suporte a --resume."""
    done = set()
    for p in output_dir.glob("P*/result.json"):
        done.add(p.parent.name[1:].lstrip("0") or "0")
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Ficheiro TXT de prompts")
    parser.add_argument("--mapas",  action="store_true", help="Gerar e guardar mapa HTML por prompt")
    parser.add_argument("--resume", action="store_true", help="Retomar de onde parou")
    args = parser.parse_args()

    prompts = load_prompts(args.input)
    gerar_mapa = args.mapas

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY não encontrada no .env")
        sys.exit(1)

    benchmark_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir   = Path(f"outputs/benchmark_{benchmark_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.txt"

    print(f"\n{'='*60}")
    print(f"BENCHMARK DE PROMPTS")
    print(f"  Input:   {args.input}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Mapas:   {'✓' if gerar_mapa else '✗'}")
    print(f"  Resume:  {'✓' if args.resume else '✗'}")
    print(f"  Output:  {output_dir}")
    print(f"{'='*60}\n")

    # Resume — carregar IDs já feitos
    done = set()
    if args.resume:
        done = load_done(output_dir)
        print(f"  Cache retomado: {len(done)} prompts já processadas\n")

    system  = TourismRouteSystem(api_key=api_key)
    total   = len(prompts)
    errors  = 0
    skipped = 0

    for i, p in enumerate(prompts):
        pid    = p["prompt_id"]
        perfil = p["perfil"]
        prompt = p["prompt"]

        if pid in done:
            skipped += 1
            continue

        print(f"\n[{i+1}/{total}] P{pid.zfill(4)} | {perfil[:20]}")
        print(f"  {prompt[:80]}...")

        t_start = time.time()
        try:
            result  = system.plan_route(prompt, use_shap=False, verbose=False, force_algorithm=None)
            elapsed = round(time.time() - t_start, 1)

            save_result(result, output_dir, pid, gerar_mapa)

            if result.get("status") == "needs_clarification":
                status   = "clarification"
                algoritmo = "-"
                fitness = n_pois = visit_min = total_min = custo = 0
            elif "error" in result:
                status    = f"error_{result['error']}"
                algoritmo = "-"
                fitness = n_pois = visit_min = total_min = custo = 0
                errors += 1
            else:
                status    = "ok"
                algoritmo = result.get("algorithm_used", "-")
                opt       = result.get("optimization", {})
                fitness   = round(opt.get("fitness", 0), 2)
                n_pois    = opt.get("n_selected", 0)
                visit_min = round(opt.get("visit_time_min", 0))
                total_min = round(opt.get("total_time_min", 0))
                custo     = round(sum(p.get("cost", 0) for p in result.get("route", [])), 2)

            save_summary_line(summary_path, {
                "prompt_id": pid, "perfil": perfil, "algoritmo": algoritmo,
                "fitness": fitness, "n_pois": n_pois, "visit_min": visit_min,
                "total_min": total_min, "custo": custo,
                "status": status, "elapsed_s": elapsed,
            })

            print(f"  ✓ {algoritmo} | fitness={fitness} | POIs={n_pois} | {elapsed}s")

        except Exception as e:
            elapsed = round(time.time() - t_start, 1)
            errors += 1
            print(f"  ✗ Erro: {e}")
            save_summary_line(summary_path, {
                "prompt_id": pid, "perfil": perfil, "algoritmo": "-",
                "fitness": 0, "n_pois": 0, "visit_min": 0,
                "total_min": 0, "custo": 0,
                "status": f"exception", "elapsed_s": elapsed,
            })

        # Aviso a cada 100 prompts
        processadas = i + 1 - skipped
        if processadas % 100 == 0:
            print(f"\n  ✔ {processadas} prompts processadas | Erros: {errors}\n")

    print(f"\n{'='*60}")
    print(f"✅ CONCLUÍDO")
    print(f"   Total:    {total}")
    print(f"   Feitas:   {total - skipped - errors}")
    print(f"   Erros:    {errors}")
    print(f"   Saltadas: {skipped}")
    print(f"   Summary:  {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()