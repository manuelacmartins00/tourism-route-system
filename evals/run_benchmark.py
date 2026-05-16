"""
evals/run_benchmark.py
Pipeline de avaliacao completo das 2000 prompts estratificadas.

Mede:
  1. Fitness score medio (Layer 3-4: RAG + Otimizador)
  2. Clarification rate (% de prompts com campos em falta)
  3. Extraction completeness (% de campos chave extraidos corretamente)
  4. Error rate (% de excecoes / falhas de pipeline)

Formato do ficheiro de input (TXT pipe-delimited):
  prompt_id|perfil|prompt

Uso:
  python evals/run_benchmark.py --input evals/prompts_geradas_2000.txt
  python evals/run_benchmark.py --input evals/prompts_geradas_2000.txt --resume
  python evals/run_benchmark.py --input evals/prompts_geradas_2000.txt --max 200

Throttle: 1 req/s por defeito para nao saturar a Groq API.
O script suporta --resume para continuar de onde parou.
"""

import os, sys, json, time, argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Forcar stdout sem buffer (necessario quando output e redirigido para ficheiro)
sys.stdout.reconfigure(line_buffering=True)

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from main_system import TourismRouteSystem
from src.llm.llm_orchestrator import LlamaOrchestrator

# Campos obrigatorios que devem ser extraidos em qualquer query completa
REQUIRED_FIELDS = ["location", "max_time", "max_cost", "transport_mode"]


def load_prompts(txt_path: str):
    with open(txt_path, encoding="utf-8") as f:
        linhas = [l.strip() for l in f if l.strip() and not l.startswith("#")]
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


def load_done(output_dir: Path) -> set:
    done = set()
    for p in output_dir.glob("P*/result.json"):
        done.add(p.parent.name[1:].lstrip("0") or "0")
    return done


def save_result(result: dict, output_dir: Path, prompt_id: str):
    result_dir = output_dir / f"P{prompt_id.zfill(4)}"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_to_save = json.loads(json.dumps(result, default=str))
    result_to_save.get("optimization", {}).pop("fitness_history", None)
    result_to_save.pop("shap_explanation", None)
    result_to_save.pop("day_plan", None)
    result_to_save.pop("map_file", None)
    with open(result_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result_to_save, f, indent=2, ensure_ascii=False)


def save_summary_line(summary_path: Path, entry: dict):
    write_header = not summary_path.exists()
    with open(summary_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(
                "prompt_id|perfil|status|algoritmo|fitness|n_pois|"
                "visit_min|total_min|custo|n_missing|fields_extracted|elapsed_s\n"
            )
        f.write(
            f"{entry['prompt_id']}|{entry['perfil']}|{entry['status']}|"
            f"{entry['algoritmo']}|{entry['fitness']}|{entry['n_pois']}|"
            f"{entry['visit_min']}|{entry['total_min']}|{entry['custo']}|"
            f"{entry['n_missing']}|{entry['fields_extracted']}|{entry['elapsed_s']}\n"
        )


def compute_extraction_completeness(prefs) -> tuple[int, int]:
    """Devolve (campos_extraidos, campos_total) para os campos obrigatorios."""
    extracted = 0
    total = len(REQUIRED_FIELDS)
    field_map = {
        "location":       prefs.location,
        "max_time":       prefs.max_time,
        "max_cost":       prefs.max_cost,
        "transport_mode": prefs.transport_mode,
    }
    for f in REQUIRED_FIELDS:
        val = field_map.get(f)
        if val is not None and val not in (0, 0.0, "", "foot"):
            extracted += 1
        elif f == "transport_mode" and val == "foot":
            # "foot" so e valido se a pe for mencionado; caso contrario e o default
            # Como nao temos ground truth aqui, contamos sempre que nao seja None
            extracted += 1
    return extracted, total


def analyse_results(summary_path: Path) -> dict:
    """Le o summary.txt e calcula metricas agregadas."""
    if not summary_path.exists():
        return {}

    rows = []
    with open(summary_path, encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header is None:
                header = line.split("|")
                continue
            parts = line.split("|")
            if len(parts) == len(header):
                rows.append(dict(zip(header, parts)))

    if not rows:
        return {}

    total = len(rows)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    clarif_rows = [r for r in rows if r["status"] == "clarification"]
    error_rows = [r for r in rows if r["status"] not in ("ok", "clarification")]

    fitness_vals = []
    for r in ok_rows:
        try:
            fitness_vals.append(float(r["fitness"]))
        except (ValueError, KeyError):
            pass

    extracted_vals = []
    missing_vals = []
    for r in rows:
        try:
            extracted_vals.append(int(r.get("fields_extracted", "0").split("/")[0]))
            missing_vals.append(int(r.get("n_missing", "0")))
        except (ValueError, IndexError):
            pass

    return {
        "total":               total,
        "ok":                  len(ok_rows),
        "clarification":       len(clarif_rows),
        "error":               len(error_rows),
        "clarification_rate":  round(len(clarif_rows) / total * 100, 1) if total else 0,
        "error_rate":          round(len(error_rows) / total * 100, 1) if total else 0,
        "fitness_mean":        round(sum(fitness_vals) / len(fitness_vals), 3) if fitness_vals else 0,
        "fitness_min":         round(min(fitness_vals), 3) if fitness_vals else 0,
        "fitness_max":         round(max(fitness_vals), 3) if fitness_vals else 0,
        "extraction_mean":     round(sum(extracted_vals) / len(extracted_vals), 1) if extracted_vals else 0,
        "avg_missing_fields":  round(sum(missing_vals) / len(missing_vals), 2) if missing_vals else 0,
    }


def print_final_report(metrics: dict, output_dir: Path):
    print(f"\n{'='*65}")
    print("RELATORIO FINAL")
    print(f"{'='*65}")
    print(f"  Prompts processadas : {metrics.get('total', 0)}")
    print(f"  OK (rota gerada)    : {metrics.get('ok', 0)}")
    print(f"  Clarification       : {metrics.get('clarification', 0)}  "
          f"({metrics.get('clarification_rate', 0):.1f}%)")
    print(f"  Erros               : {metrics.get('error', 0)}  "
          f"({metrics.get('error_rate', 0):.1f}%)")
    print()
    print(f"  Fitness medio       : {metrics.get('fitness_mean', 0):.3f}")
    print(f"  Fitness min/max     : {metrics.get('fitness_min', 0):.3f} / "
          f"{metrics.get('fitness_max', 0):.3f}")
    print()
    print(f"  Campos extraidos (media) : {metrics.get('extraction_mean', 0):.1f} / "
          f"{len(REQUIRED_FIELDS)}")
    print(f"  Campos em falta (media)  : {metrics.get('avg_missing_fields', 0):.2f}")
    print(f"{'='*65}")
    print(f"  Resultados em: {output_dir}")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark completo das 2000 prompts")
    parser.add_argument("--input",  required=True, help="Ficheiro TXT de prompts (prompt_id|perfil|prompt)")
    parser.add_argument("--resume", action="store_true", help="Retomar de onde parou")
    parser.add_argument("--max",    type=int, default=0, help="Limitar a N prompts (0 = todas)")
    parser.add_argument("--throttle", type=float, default=1.5,
                        help="Segundos entre requests Groq (default 1.5)")
    args = parser.parse_args()

    prompts = load_prompts(args.input)
    if args.max > 0:
        prompts = prompts[:args.max]

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERRO: GROQ_API_KEY nao definida no .env")
        sys.exit(1)

    benchmark_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir   = Path(f"outputs/benchmark_{benchmark_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.txt"

    # Log file para monitorizar progresso em background
    log_path = output_dir / "progress.log"
    _log_f = open(log_path, "w", encoding="utf-8", buffering=1)

    def log(msg: str):
        print(msg)
        _log_f.write(msg + "\n")

    log(f"\n{'='*65}")
    log(f"BENCHMARK DE PROMPTS TURISTICAS")
    log(f"  Input    : {args.input}")
    log(f"  Prompts  : {len(prompts)}")
    log(f"  Throttle : {args.throttle}s entre requests")
    log(f"  Output   : {output_dir}")
    log(f"  Log      : {log_path}")
    log(f"{'='*65}\n")

    done = set()
    if args.resume:
        done = load_done(output_dir)
        log(f"  Resume: {len(done)} prompts ja processadas\n")

    system = TourismRouteSystem(api_key=api_key)
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

        log(f"\n[{i+1}/{total}] P{pid.zfill(4)} | {perfil[:25]}")
        log(f"  {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

        t_start = time.time()
        try:
            result  = system.plan_route(
                prompt,
                use_shap=False,
                verbose=False,
                force_algorithm=None,
                include_accommodation=True,
                include_meals=True,
            )
            elapsed = round(time.time() - t_start, 1)

            save_result(result, output_dir, pid)

            if result.get("status") == "needs_clarification":
                status    = "clarification"
                algoritmo = "-"
                fitness = n_pois = visit_min = total_min = custo = 0
                n_missing = len(result.get("missing_fields", []))
                fields_extracted = "0/0"
            elif "error" in result:
                status    = "error"
                algoritmo = "-"
                fitness = n_pois = visit_min = total_min = custo = 0
                n_missing = 0
                fields_extracted = "0/0"
                errors += 1
            else:
                status    = "ok"
                algoritmo = result.get("algorithm_used", "-")
                opt       = result.get("optimization", {})
                fitness   = round(opt.get("fitness", 0), 3)
                n_pois    = opt.get("n_selected", 0)
                visit_min = round(opt.get("visit_time_min", 0))
                total_min = round(opt.get("total_time_min", 0))
                custo     = round(sum(poi.get("cost", 0) for poi in result.get("route", [])), 2)
                n_missing = 0
                fields_extracted = f"{len(REQUIRED_FIELDS)}/{len(REQUIRED_FIELDS)}"

            save_summary_line(summary_path, {
                "prompt_id": pid, "perfil": perfil, "status": status,
                "algoritmo": algoritmo, "fitness": fitness,
                "n_pois": n_pois, "visit_min": visit_min,
                "total_min": total_min, "custo": custo,
                "n_missing": n_missing, "fields_extracted": fields_extracted,
                "elapsed_s": elapsed,
            })

            if status == "ok":
                log(f"  OK {algoritmo} | fitness={fitness:.3f} | POIs={n_pois} | {elapsed}s")
            elif status == "clarification":
                log(f"  CLARIF missing={n_missing} | {elapsed}s")
            else:
                log(f"  ERROR | {elapsed}s")

        except Exception as e:
            elapsed = round(time.time() - t_start, 1)
            errors += 1
            log(f"  EXCECAO: {e}")
            save_summary_line(summary_path, {
                "prompt_id": pid, "perfil": perfil, "status": "exception",
                "algoritmo": "-", "fitness": 0, "n_pois": 0,
                "visit_min": 0, "total_min": 0, "custo": 0,
                "n_missing": 0, "fields_extracted": "0/0",
                "elapsed_s": elapsed,
            })

        # Throttle Groq
        time.sleep(args.throttle)

        # Checkpoint a cada 100 prompts
        processadas = i + 1 - skipped
        if processadas % 100 == 0:
            interim = analyse_results(summary_path)
            log(f"\n  --- Checkpoint {processadas} prompts ---")
            log(f"  Fitness medio ate agora: {interim.get('fitness_mean', 0):.3f}")
            log(f"  Clarification rate:      {interim.get('clarification_rate', 0):.1f}%")
            log(f"  Erros:                   {interim.get('error', 0)}\n")

    # Relatorio final
    final_metrics = analyse_results(summary_path)
    print_final_report(final_metrics, output_dir)

    # Guardar metricas JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    log(f"Metricas JSON: {metrics_path}")
    _log_f.close()


if __name__ == "__main__":
    main()
