"""
Layer 1 eval: mede field-level accuracy da extracao de preferencias do LLM.
Uso: python evals/run_extraction_eval.py
Requer: GROQ_API_KEY no ambiente ou .env
"""
import sys, json, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.llm_orchestrator import LlamaOrchestrator

CASES_FILE = Path(__file__).parent / "extraction_cases.json"
RESULTS_FILE = Path(__file__).parent / "extraction_results.json"

COMPARABLE_FIELDS = [
    "location", "max_time", "transport_mode", "num_people",
    "has_children", "mobility_issues", "start_time", "last_day_end_time",
]

def field_match(expected_val, got_val, field):
    if expected_val is None:
        return True  # campo nao testado neste caso
    if got_val is None:
        return False
    if field == "max_time":
        # tolerancia de 20%
        return abs(int(got_val) - int(expected_val)) <= 0.2 * int(expected_val)
    if field == "max_cost":
        return abs(float(got_val) - float(expected_val)) <= 0.15 * float(expected_val)
    return str(got_val).lower() == str(expected_val).lower()

def missing_fields_match(expected, got):
    exp_set = set(expected or [])
    got_set = set(got or [])
    return exp_set == got_set

def run_eval():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERRO: GROQ_API_KEY nao definida")
        sys.exit(1)

    llm = LlamaOrchestrator(api_key=api_key)

    with open(CASES_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    total_fields = 0
    correct_fields = 0

    print(f"\nA correr {len(cases)} casos de teste...\n")
    print(f"{'ID':<8} {'Query':<55} {'Campos OK':<12} {'Missing OK'}")
    print(f"{'-'*90}")

    for case in cases:
        cid = case["id"]
        query = case["query"]
        expected = case["expected"]

        try:
            prefs = llm.extract_preferences(query)
        except Exception as e:
            print(f"{cid:<8} ERRO: {e}")
            continue

        got = {
            "location":          prefs.location,
            "max_time":          prefs.max_time,
            "max_cost":          prefs.max_cost,
            "transport_mode":    prefs.transport_mode,
            "num_people":        prefs.num_people,
            "has_children":      prefs.has_children,
            "mobility_issues":   prefs.mobility_issues,
            "start_time":        prefs.start_time,
            "last_day_end_time": prefs.last_day_end_time,
        }

        field_results = {}
        case_correct = 0
        case_total = 0
        for field in COMPARABLE_FIELDS:
            if field not in expected:
                continue
            match = field_match(expected[field], got.get(field), field)
            field_results[field] = {"expected": expected[field], "got": got.get(field), "match": match}
            case_total += 1
            total_fields += 1
            if match:
                case_correct += 1
                correct_fields += 1

        missing_ok = missing_fields_match(expected.get("missing_fields"), prefs.missing_fields) if "missing_fields" in expected else None

        short_query = query[:52] + "..." if len(query) > 52 else query
        missing_str = "OK" if missing_ok else ("FAIL" if missing_ok is False else "N/A")
        print(f"{cid:<8} {short_query:<55} {case_correct}/{case_total:<9} {missing_str}")

        results.append({
            "id": cid,
            "query": query,
            "fields": field_results,
            "missing_fields_match": missing_ok,
        })

    accuracy = correct_fields / total_fields * 100 if total_fields > 0 else 0
    print(f"\n{'='*60}")
    print(f"Accuracy total: {correct_fields}/{total_fields} campos ({accuracy:.1f}%)")
    print(f"{'='*60}\n")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy_pct": round(accuracy, 1),
            "correct_fields": correct_fields,
            "total_fields": total_fields,
            "cases": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"Resultados guardados em {RESULTS_FILE}")

if __name__ == "__main__":
    run_eval()
