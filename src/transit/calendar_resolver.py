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


# ---------------------------------------------
# Metro Lisboa - 3 service_ids fixos (1/2/3)
# calendar.txt standard + calendar_dates.txt
# ---------------------------------------------
def resolve_metro_lisboa(gtfs_dir: Path, query_date: date) -> Set[str]:
    active = set()
    day_name = query_date.strftime("%A").lower()  # "monday", "tuesday", ...
    date_str = query_date.strftime("%Y%m%d")

    # 1. calendar.txt - regras semanais base
    for row in _load_csv(gtfs_dir / "calendar.txt"):
        start = datetime.strptime(row["start_date"], "%Y%m%d").date()
        end = datetime.strptime(row["end_date"], "%Y%m%d").date()
        if start <= query_date <= end and row[day_name] == "1":
            active.add(row["service_id"])

    # 2. calendar_dates.txt - excepcoes (1=adicionar, 2=remover)
    for row in _load_csv(gtfs_dir / "calendar_dates.txt"):
        if row["date"] == date_str:
            if row["exception_type"] == "1":
                active.add(row["service_id"])
            elif row["exception_type"] == "2":
                active.discard(row["service_id"])

    return active


# ---------------------------------------------
# Metro Porto - 19 service_ids com sufixos U/S/DF por linha
# calendar.txt standard + calendar_dates.txt
# ---------------------------------------------
def resolve_metro_porto(gtfs_dir: Path, query_date: date) -> Set[str]:
    # Mesma logica que Metro Lisboa - calendar.txt standard
    return resolve_metro_lisboa(gtfs_dir, query_date)


# ---------------------------------------------
# STCP - so calendar_dates.txt com datas explicitas
# service_ids: "DIAS UTEIS", "SABADOS", "DOMINGOS|FERIADOS"
# ---------------------------------------------
def resolve_stcp(gtfs_dir: Path, query_date: date) -> Set[str]:
    active = set()
    date_str = query_date.strftime("%Y%m%d")
    for row in _load_csv(gtfs_dir / "calendar_dates.txt"):
        if row["date"] == date_str and row["exception_type"] == "1":
            active.add(row["service_id"])
    return active


# ---------------------------------------------
# CarrisMetropolitana - sistema proprietario [PLAN_ID]PATTERN
# Usa dates.txt (nao-standard) para classificar a data
# ---------------------------------------------
def resolve_carris(gtfs_dir: Path, query_date: date) -> Set[str]:
    date_str = query_date.strftime("%Y%m%d")

    # calendar_dates.txt already encodes which services run on each date -
    # both named-period services (e.g. [KFULM]ESC_DU) and numeric ones
    # (e.g. [89CJD]11). Just trust those explicit entries and filter by
    # whether the service's plan is currently active.
    active_plans = set()
    for row in _load_csv(gtfs_dir / "plans.txt"):
        start = datetime.strptime(row["plan_start_date"], "%Y%m%d").date()
        end = datetime.strptime(row["plan_end_date"], "%Y%m%d").date()
        if start <= query_date <= end:
            active_plans.add(row["plan_id"])

    active = set()
    for row in _load_csv(gtfs_dir / "calendar_dates.txt"):
        if row["date"] == date_str and row["exception_type"] == "1":
            sid = row["service_id"]
            if sid.startswith("[") and "]" in sid:
                plan_id = sid[1:sid.index("]")]
                if plan_id in active_plans:
                    active.add(sid)

    return active


# ---------------------------------------------
# CP - 2500 service_ids individuais com datas proprias
# calendar.txt com start_date/end_date por service_id + dia da semana
# ---------------------------------------------
def resolve_cp(gtfs_dir: Path, query_date: date) -> Set[str]:
    # CP usa calendar.txt standard mas com um service_id por trip/frequencia
    # A logica e igual ao Metro Lisboa - calendar.txt + calendar_dates.txt
    return resolve_metro_lisboa(gtfs_dir, query_date)


# ---------------------------------------------
# Dispatcher
# ---------------------------------------------
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