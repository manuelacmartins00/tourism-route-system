# scripts/log_to_hf.py
from huggingface_hub import HfApi
import json, csv, os, uuid
from datetime import datetime
from pathlib import Path

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO  = "ManuelMartinsTeseISCTE/TourismRunsLog"

RUNS_CSV     = Path("data/runs_log.csv")
RUNS_DIR     = Path("data/runs")
MAPS_DIR     = Path("outputs/maps")

def _upload(local_path: str, repo_path: str):
    if not HF_TOKEN:
        return
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=HF_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
        )
    except Exception as e:
        print(f"AVISO: log_to_hf upload erro: {e}")

def log_run(query: str, result: dict,
            clarification_fields: list = None,
            clarification_answers: dict = None,
            elapsed_seconds: float = None,
            map_id: str = None,
            user_ip: str = None):
    """
    Guarda uma run completa localmente e faz upload para HF Dataset.

    Guarda:
    - runs_log.csv       - linha por run (resumo)
    - runs/{run_id}.json - JSON completo da run
    - maps/{run_id}.html - mapa HTML (copia)
    """

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    run_id    = str(uuid.uuid4())[:12]
    timestamp = datetime.utcnow().isoformat()
    prefs     = result.get("preferences", {})
    opt       = result.get("optimization", {})
    route     = result.get("route", [])

    # -- 1. CSV de resumo ----------------------------------------------
    write_header = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_id", "timestamp",
                "query",
                "clarification_fields", "clarification_answers",
                "location", "transport_mode",
                "max_time_min", "max_cost",
                "categories", "interests",
                "algorithm", "fitness",
                "n_candidates", "n_selected",
                "total_cost", "visit_time_min", "travel_time_min", "total_time_min",
                "elapsed_seconds",
                "map_id",
                "n_pois_na_rota",
                "pois_nomes",
                "user_ip",
            ])
        writer.writerow([
            run_id, timestamp,
            query,
            json.dumps(clarification_fields or [], ensure_ascii=False),
            json.dumps(clarification_answers or {}, ensure_ascii=False),
            prefs.get("location", ""),
            prefs.get("transport_mode", "foot"),
            prefs.get("max_time", ""),
            prefs.get("max_cost", ""),
            json.dumps(prefs.get("categories", []), ensure_ascii=False),
            json.dumps(prefs.get("interests", []), ensure_ascii=False),
            result.get("algorithm_used", ""),
            opt.get("fitness", ""),
            opt.get("n_candidates", ""),
            opt.get("n_selected", ""),
            sum(p.get("cost", 0) for p in route),
            opt.get("visit_time_min", ""),
            opt.get("travel_time_min", ""),
            opt.get("total_time_min", ""),
            round(elapsed_seconds, 1) if elapsed_seconds else "",
            map_id or "",
            len(route),
            json.dumps([p.get("name", "") for p in route], ensure_ascii=False),
            user_ip or "",
        ])

    # -- 2. JSON completo da run ---------------------------------------
    run_json = {
        "run_id": run_id,
        "timestamp": timestamp,
        "query": query,
        "clarification_fields": clarification_fields or [],
        "clarification_answers": clarification_answers or {},
        "elapsed_seconds": elapsed_seconds,
        "map_id": map_id,
        "user_ip": user_ip or None,
        "result": result,
    }
    run_json_path = RUNS_DIR / f"{run_id}.json"
    with open(run_json_path, "w", encoding="utf-8") as f:
        json.dump(run_json, f, ensure_ascii=False, indent=2)

    # -- 3. Upload -----------------------------------------------------
    _upload(str(RUNS_CSV),      "runs_log.csv")
    _upload(str(run_json_path), f"runs/{run_id}.json")

    # Mapa HTML
    if map_id:
        map_path = MAPS_DIR / f"{map_id}.html"
        if map_path.exists():
            _upload(str(map_path), f"maps/{run_id}_{map_id}.html")

    return run_id


def log_feedback(run_id: str, feedback_data: dict, sus_score: float):
    """
    Associa um questionario SUS a uma run especifica e faz upload.
    """
    FEEDBACK_DIR = Path("data/feedback")
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    fb_path = FEEDBACK_DIR / f"feedback_{run_id}.json"
    payload = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "sus_score": sus_score,
        **feedback_data
    }
    with open(fb_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    _upload(str(fb_path), f"feedback/feedback_{run_id}.json")

    # Upload do CSV agregado de feedback
    feedback_csv = Path("data/feedback/responses.csv")
    if feedback_csv.exists():
        _upload(str(feedback_csv), "feedback/responses.csv")