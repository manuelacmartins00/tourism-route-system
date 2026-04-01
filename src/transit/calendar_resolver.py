# src/transit/calendar_resolver.py
import csv
from datetime import datetime, date
from typing import Set, Dict
from pathlib import Path


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ─────────────────────────────────────────────
# Metro Lisboa — 3 service_ids fixos (1/2/3)
# calendar.txt standard + calendar_dates.txt
# ─────────────────────────────────────────────
def resolve_metro_lisboa(gtfs_dir: Path, query_date: date) -> Set[str]:
    active = set()
    day_name = query_date.strftime("%A").lower()  # "monday", "tuesday", ...
    date_str = query_date.strftime("%Y%m%d")

    # 1. calendar.txt — regras semanais base
    for row in _load_csv(gtfs_dir / "calendar.txt"):
        start = datetime.strptime(row["start_date"], "%Y%m%d").date()
        end = datetime.strptime(row["end_date"], "%Y%m%d").date()
        if start <= query_date <= end and row[day_name] == "1":
            active.add(row["service_id"])

    # 2. calendar_dates.txt — excepções (1=adicionar, 2=remover)
    for row in _load_csv(gtfs_dir / "calendar_dates.txt"):
        if row["date"] == date_str:
            if row["exception_type"] == "1":
                active.add(row["service_id"])
            elif row["exception_type"] == "2":
                active.discard(row["service_id"])

    return active


# ─────────────────────────────────────────────
# Metro Porto — 19 service_ids com sufixos U/S/DF por linha
# calendar.txt standard + calendar_dates.txt
# ─────────────────────────────────────────────
def resolve_metro_porto(gtfs_dir: Path, query_date: date) -> Set[str]:
    # Mesma lógica que Metro Lisboa — calendar.txt standard
    return resolve_metro_lisboa(gtfs_dir, query_date)


# ─────────────────────────────────────────────
# STCP — só calendar_dates.txt com datas explícitas
# service_ids: "DIAS UTEIS", "SÁBADOS", "DOMINGOS|FERIADOS"
# ─────────────────────────────────────────────
def resolve_stcp(gtfs_dir: Path, query_date: date) -> Set[str]:
    active = set()
    date_str = query_date.strftime("%Y%m%d")
    for row in _load_csv(gtfs_dir / "calendar_dates.txt"):
        if row["date"] == date_str and row["exception_type"] == "1":
            active.add(row["service_id"])
    return active


# ─────────────────────────────────────────────
# CarrisMetropolitana — sistema proprietário [PLAN_ID]PATTERN
# Usa dates.txt (não-standard) para classificar a data
# ─────────────────────────────────────────────
def resolve_carris(gtfs_dir: Path, query_date: date) -> Set[str]:
    date_str = query_date.strftime("%Y%m%d")

    # 1. Determinar day_type e period a partir do dates.txt
    day_type, period = None, None
    for row in _load_csv(gtfs_dir / "dates.txt"):
        if row["date"] == date_str:
            day_type = row["day_type"]   # 1=DU, 2=SAB, 3=DOM
            period = row["period"]       # 1=ESC, 2=FER, 3=VER
            break

    if not day_type:
        # Fallback: inferir pelo dia da semana, assumir período escolar
        wd = query_date.weekday()  # 0=Mon ... 6=Sun
        day_type = "1" if wd < 5 else ("2" if wd == 5 else "3")
        period = "1"

    period_map = {"1": "ESC", "2": "FER", "3": "VER"}
    day_map = {"1": "DU", "2": "SAB", "3": "DOM"}
    target_pattern = f"{period_map[period]}_{day_map[day_type]}"

    # 2. Encontrar service_ids activos com esse pattern
    active = set()
    
    # Carregar plans activos para esta data
    active_plans = set()
    for row in _load_csv(gtfs_dir / "plans.txt"):
        start = datetime.strptime(row["plan_start_date"], "%Y%m%d").date()
        end = datetime.strptime(row["plan_end_date"], "%Y%m%d").date()
        if start <= query_date <= end:
            active_plans.add(row["plan_id"])

    # service_id formato: [PLAN_ID]PATTERN
    for row in _load_csv(gtfs_dir / "calendar_dates.txt"):
        sid = row["service_id"]
        if row["date"] == date_str and row["exception_type"] == "1":
            # Verificar se pertence a um plan activo e tem o pattern certo
            if sid.startswith("[") and "]" in sid:
                plan_id = sid[1:sid.index("]")]
                pattern = sid[sid.index("]") + 1:]
                if plan_id in active_plans and pattern == target_pattern:
                    active.add(sid)

    return active


# ─────────────────────────────────────────────
# CP — 2500 service_ids individuais com datas próprias
# calendar.txt com start_date/end_date por service_id + dia da semana
# ─────────────────────────────────────────────
def resolve_cp(gtfs_dir: Path, query_date: date) -> Set[str]:
    # CP usa calendar.txt standard mas com um service_id por trip/frequência
    # A lógica é igual ao Metro Lisboa — calendar.txt + calendar_dates.txt
    return resolve_metro_lisboa(gtfs_dir, query_date)


# ─────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────
RESOLVERS = {
    "metro_lisboa":        resolve_metro_lisboa,
    "metro_porto":         resolve_metro_porto,
    "stcp":                resolve_stcp,
    "carris_metropolitana": resolve_carris,
    "cp":                  resolve_cp,
}

def get_active_services(operator: str, gtfs_dir: Path, query_date: date) -> Set[str]:
    resolver = RESOLVERS.get(operator)
    if not resolver:
        raise ValueError(f"Operador desconhecido: {operator}")
    return resolver(gtfs_dir, query_date)