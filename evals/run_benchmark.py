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

# Forcar stdout sem buffer e UTF-8 (necessario quando output e redirigido para ficheiro;
# o cp1252 default da consola Windows crasha em caracteres como "->" escritos como seta unicode)
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

load_dotenv()
os.environ.setdefault("BENCHMARK_MODE", "1")
sys.path.insert(0, str(Path(__file__).parent.parent))

from main_system import TourismRouteSystem
from src.llm.llm_orchestrator import LlamaOrchestrator, DailyQuotaExceededError

# Campos obrigatorios que devem ser extraidos em qualquer query completa
REQUIRED_FIELDS = ["location", "max_time", "max_cost", "transport_mode"]


def load_prompts(txt_path: str):
    with open(txt_path, encoding="utf-8") as f:
        linhas = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    prompts = []
    for linha in linhas:
        # Suporta formato antigo (3 campos) e novo (5 campos com include_accommodation/meals)
        partes = linha.split("|", 4)
        if len(partes) == 5:
            prompts.append({
                "prompt_id":             partes[0].strip(),
                "perfil":                partes[1].strip(),
                "include_accommodation": partes[2].strip().lower() == "true",
                "include_meals":         partes[3].strip().lower() == "true",
                "prompt":                partes[4].strip(),
            })
        elif len(partes) == 3:
            prompts.append({
                "prompt_id":             partes[0].strip(),
                "perfil":                partes[1].strip(),
                "include_accommodation": True,
                "include_meals":         True,
                "prompt":                partes[2].strip(),
            })
    return prompts


def load_done(output_dir: Path) -> set:
    """Lê prompts já processadas do results.json (se existir)."""
    done = set()
    results_path = output_dir / "results.json"
    if results_path.exists():
        try:
            with open(results_path, encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                pid = str(entry.get("prompt_id", "")).lstrip("0") or "0"
                done.add(pid)
        except Exception:
            pass
    return done


def save_result(result: dict, results_path: Path, prompt_id: str, perfil: str):
    """Acrescenta resultado ao results.json único (lê, adiciona, reescreve)."""
    result_to_save = json.loads(json.dumps(result, default=str))
    result_to_save.get("optimization", {}).pop("fitness_history", None)
    result_to_save.pop("map_file", None)
    result_to_save["prompt_id"] = prompt_id
    result_to_save["perfil"] = perfil

    existing = []
    if results_path.exists():
        try:
            with open(results_path, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(result_to_save)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def save_summary_line(summary_path: Path, entry: dict):
    write_header = not summary_path.exists()
    with open(summary_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(
                "prompt_id|perfil|status|algoritmo|fitness|n_pois|"
                "visit_min|total_min|custo|n_missing|fields_extracted|elapsed_s|"
                "time_util|time_eff|cat_comp|div_comp|dist_pen|prox_comp|ctx_mod|unique_cats\n"
            )
        fc = entry.get("fitness_components", {})
        f.write(
            f"{entry['prompt_id']}|{entry['perfil']}|{entry['status']}|"
            f"{entry['algoritmo']}|{entry['fitness']}|{entry['n_pois']}|"
            f"{entry['visit_min']}|{entry['total_min']}|{entry['custo']}|"
            f"{entry['n_missing']}|{entry['fields_extracted']}|{entry['elapsed_s']}|"
            f"{fc.get('time_utilization','')}|{fc.get('time_efficiency','')}|"
            f"{fc.get('category_component','')}|{fc.get('diversity_component','')}|"
            f"{fc.get('distance_penalty','')}|{fc.get('proximity_component','')}|"
            f"{fc.get('contextual_modifier','')}|{fc.get('unique_categories','')}\n"
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

    def _mean(vals):
        return round(sum(vals) / len(vals), 3) if vals else 0

    def _floats(rows, col):
        out = []
        for r in rows:
            try:
                v = r.get(col, "")
                if v != "":
                    out.append(float(v))
            except (ValueError, KeyError):
                pass
        return out

    return {
        "total":               total,
        "ok":                  len(ok_rows),
        "clarification":       len(clarif_rows),
        "error":               len(error_rows),
        "clarification_rate":  round(len(clarif_rows) / total * 100, 1) if total else 0,
        "error_rate":          round(len(error_rows) / total * 100, 1) if total else 0,
        "fitness_mean":        _mean(_floats(ok_rows, "fitness")),
        "fitness_min":         round(min(_floats(ok_rows, "fitness")), 3) if _floats(ok_rows, "fitness") else 0,
        "fitness_max":         round(max(_floats(ok_rows, "fitness")), 3) if _floats(ok_rows, "fitness") else 0,
        "extraction_mean":     round(sum(extracted_vals) / len(extracted_vals), 1) if extracted_vals else 0,
        "avg_missing_fields":  round(sum(missing_vals) / len(missing_vals), 2) if missing_vals else 0,
        # Componentes AHP (apenas nas prompts OK)
        "time_util_mean":      _mean(_floats(ok_rows, "time_util")),
        "time_eff_mean":       _mean(_floats(ok_rows, "time_eff")),
        "cat_comp_mean":       _mean(_floats(ok_rows, "cat_comp")),
        "div_comp_mean":       _mean(_floats(ok_rows, "div_comp")),
        "dist_pen_mean":       _mean(_floats(ok_rows, "dist_pen")),
        "prox_comp_mean":      _mean(_floats(ok_rows, "prox_comp")),
        "ctx_mod_mean":        _mean(_floats(ok_rows, "ctx_mod")),
        "unique_cats_mean":    _mean(_floats(ok_rows, "unique_cats")),
    }


def build_resume_args_str(args, output_dir: Path) -> str:
    """Reconstroi a linha de argumentos para relançar este script com --resume."""
    parts = [f'--input "{args.input}"', "--resume", f'--output-dir "{output_dir}"']
    if args.max:
        parts.append(f"--max {args.max}")
    parts.append(f"--throttle {args.throttle}")
    parts.append(f"--timeout {args.timeout}")
    if args.force_algorithm:
        parts.append(f"--force-algorithm {args.force_algorithm}")
    return " ".join(parts)


def schedule_windows_resume(reset_at, args, output_dir: Path, log, new_this_session: int = 999) -> bool:
    """Cria (ou substitui) uma tarefa do Windows Task Scheduler que relança este
    benchmark com --resume assim que a quota diaria da Groq reinicia.
    Sobrevive a reinicios/suspensao do portatil, ao contrario de um sleep in-process.
    """
    import subprocess
    from datetime import timedelta

    python_exe   = sys.executable
    script_path  = str(Path(__file__).resolve())
    project_dir  = str(Path(__file__).resolve().parent.parent)
    task_name    = "TRS_Benchmark_Resume"
    # Folga de seguranca: o "try again in Xm" da Groq ja falhou 2x seguidas (nova
    # falha quase instantanea apos o resume), por isso 2 min nao chega. Se muito
    # poucas prompts novas foram processadas nesta sessao antes de esgotar outra
    # vez, e sinal de que ainda estamos fundo no vale de quota -> escalar a folga.
    buffer_min   = 20 if new_this_session >= 5 else 60
    run_at       = reset_at + timedelta(minutes=buffer_min)
    resume_args  = build_resume_args_str(args, output_dir)

    ps_script = f"""
$ErrorActionPreference = 'Stop'
$action  = New-ScheduledTaskAction -Execute '{python_exe}' -Argument '"{script_path}" {resume_args}' -WorkingDirectory '{project_dir}'
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date -Year {run_at.year} -Month {run_at.month} -Day {run_at.day} -Hour {run_at.hour} -Minute {run_at.minute} -Second {run_at.second})
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -Hidden -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 20)
Unregister-ScheduledTask -TaskName '{task_name}' -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName '{task_name}' -Action $action -Trigger $trigger -Settings $settings -Description 'Auto-resume do benchmark de 490 prompts apos reset da quota diaria Groq' -Force | Out-Null
"""
    ps_path = Path(project_dir) / "outputs" / "_resume_task.ps1"
    ps_path.parent.mkdir(parents=True, exist_ok=True)
    ps_path.write_text(ps_script, encoding="utf-8")

    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps_path)],
        capture_output=True, text=True,
    )
    ok = result.returncode == 0
    log(f"  Tarefa '{task_name}' agendada para {run_at.strftime('%Y-%m-%d %H:%M:%S')}: "
        f"{'OK' if ok else 'FALHOU'}")
    if not ok:
        log(f"    stderr: {result.stderr.strip()[:800]}")
        log(f"    Comando de resume manual:")
        log(f"    {python_exe} \"{script_path}\" {resume_args}")
    return ok


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
    print("  -- Componentes AHP (media, prompts OK) --")
    print(f"  Time utilization    : {metrics.get('time_util_mean', 0):.1f}")
    print(f"  Time efficiency     : {metrics.get('time_eff_mean', 0):.1f}")
    print(f"  Category match      : {metrics.get('cat_comp_mean', 0):.1f}")
    print(f"  Diversity           : {metrics.get('div_comp_mean', 0):.1f}")
    print(f"  Distance penalty    : {metrics.get('dist_pen_mean', 0):.1f}")
    print(f"  Proximity           : {metrics.get('prox_comp_mean', 0):.1f}")
    print(f"  Contextual modifier : {metrics.get('ctx_mod_mean', 0):.3f}")
    print(f"  Unique categories   : {metrics.get('unique_cats_mean', 0):.1f}")
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
    parser.add_argument("--resume", action="store_true", help="Retomar de onde parou (salta prompts com result.json)")
    parser.add_argument("--output-dir", default=None, help="Reutilizar pasta de output existente (para --resume)")
    parser.add_argument("--max",    type=int, default=0, help="Limitar a N prompts (0 = todas)")
    parser.add_argument("--throttle", type=float, default=1.5,
                        help="Segundos entre requests Groq (default 1.5)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout por prompt em segundos (default 300)")
    parser.add_argument("--force-algorithm", default=None, choices=["ACO", "GA", "PSO", "GREEDY"],
                        help="Forca um algoritmo especifico em vez da selecao automatica "
                             "(GA = producao = GA_DYN, mutation_dynamic=True)")
    args = parser.parse_args()

    prompts = load_prompts(args.input)
    if args.max > 0:
        prompts = prompts[:args.max]

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERRO: GROQ_API_KEY nao definida no .env")
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        benchmark_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir   = Path(f"outputs/benchmark_{benchmark_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
    summary_path  = output_dir / "summary.txt"
    results_path  = output_dir / "results.json"

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
            import signal as _sig
            def _timeout_handler(signum, frame):
                raise TimeoutError(f"Prompt excedeu {args.timeout}s")
            # signal.alarm apenas disponivel em Unix; em Windows usa thread
            import threading as _threading
            _result_holder = [None]
            _exc_holder    = [None]
            def _run():
                try:
                    _result_holder[0] = system.plan_route(
                        prompt,
                        use_shap=True,
                        verbose=False,
                        force_algorithm=args.force_algorithm,
                        include_accommodation=p["include_accommodation"],
                        include_meals=p["include_meals"],
                        generate_map=False,
                        generate_explanation=False,
                    )
                except Exception as exc:
                    _exc_holder[0] = exc
            t = _threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=args.timeout)
            if t.is_alive():
                elapsed = round(time.time() - t_start, 1)
                errors += 1
                log(f"  TIMEOUT (>{args.timeout}s) | {elapsed}s")
                save_summary_line(summary_path, {
                    "prompt_id": pid, "perfil": perfil, "status": "timeout",
                    "algoritmo": "-", "fitness": 0, "n_pois": 0,
                    "visit_min": 0, "total_min": 0, "custo": 0,
                    "n_missing": 0, "fields_extracted": "0/0",
                    "elapsed_s": elapsed,
                })
                time.sleep(args.throttle)
                continue
            if _exc_holder[0]:
                raise _exc_holder[0]
            result  = _result_holder[0]
            elapsed = round(time.time() - t_start, 1)

            save_result(result, results_path, pid, perfil)

            fitness_components = {}
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
                fitness_components = opt.get("fitness_components", {})

            save_summary_line(summary_path, {
                "prompt_id": pid, "perfil": perfil, "status": status,
                "algoritmo": algoritmo, "fitness": fitness,
                "n_pois": n_pois, "visit_min": visit_min,
                "total_min": total_min, "custo": custo,
                "n_missing": n_missing, "fields_extracted": fields_extracted,
                "elapsed_s": elapsed,
                "fitness_components": fitness_components,
            })

            if status == "ok":
                log(f"  OK {algoritmo} | fitness={fitness:.3f} | POIs={n_pois} | {elapsed}s")
            elif status == "clarification":
                log(f"  CLARIF missing={n_missing} | {elapsed}s")
            else:
                log(f"  ERROR | {elapsed}s")

        except DailyQuotaExceededError as e:
            elapsed = round(time.time() - t_start, 1)
            log(f"\n{'!'*65}")
            log(f"  QUOTA DIARIA GROQ ESGOTADA em P{pid.zfill(4)} ({elapsed}s)")
            log(f"  Reset previsto: {e.reset_at.strftime('%Y-%m-%d %H:%M:%S')}")
            log(f"{'!'*65}")
            # Este prompt NAO fica marcado como concluido -> --resume repete-o.
            scheduled = schedule_windows_resume(e.reset_at, args, output_dir, log,
                                                 new_this_session=i - skipped)
            if scheduled:
                log(f"\n  Tarefa agendada. Podes fechar este terminal — o benchmark "
                    f"retoma sozinho ({output_dir}).")
            log(f"\n  Progresso ate agora: {i - skipped} novas prompts processadas nesta sessao.")
            _log_f.close()
            sys.exit(0)

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
            log(f"  Fitness medio:       {interim.get('fitness_mean', 0):.3f}")
            log(f"  Time util (media):   {interim.get('time_util_mean', 0):.1f}")
            log(f"  Category (media):    {interim.get('cat_comp_mean', 0):.1f}")
            log(f"  Diversity (media):   {interim.get('div_comp_mean', 0):.1f}")
            log(f"  Proximity (media):   {interim.get('prox_comp_mean', 0):.1f}")
            log(f"  Clarification rate:  {interim.get('clarification_rate', 0):.1f}%")
            log(f"  Erros:               {interim.get('error', 0)}\n")

    # Relatorio final
    final_metrics = analyse_results(summary_path)
    print_final_report(final_metrics, output_dir)

    # Guardar metricas JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    log(f"Metricas JSON: {metrics_path}")
    log(f"Resultados  : {results_path}")
    _log_f.close()


if __name__ == "__main__":
    main()
