"""
Download and extract GTFS feeds for all Portuguese operators into data/gtfs/.

Called automatically on startup when GTFS directories are missing.
Each operator's zip is extracted to data/gtfs/<operator_name>/.
"""

import zipfile
import shutil
import time
import requests
from pathlib import Path

GTFS_ROOT = Path("data/gtfs")

# Official public GTFS feed URLs
OPERATORS = {
    "metro_lisboa": {
        "url": "https://www.metrolisboa.pt/wp-content/uploads/google_transit.zip",
        "fallback": None,
    },
    "metro_porto": {
        "url": "https://opendata.porto.digital/dataset/2b3ff67f-3fc2-4571-92a6-cef89ae8bc04/resource/2f5d0f55-71fc-4db5-9063-9b0e72b7b8c4/download/porto_metro_gtfs.zip",
        "fallback": None,
    },
    "stcp": {
        "url": "https://www.stcp.pt/wp-content/uploads/google_transit_stcp.zip",
        "fallback": None,
    },
    "carris_metropolitana": {
        # Carris Metropolitana publishes GTFS via their open API
        "url": "https://api.carrismetropolitana.pt/gtfs",
        "fallback": "https://github.com/carrismetropolitana/gtfs/raw/main/CarrisMetropolitana.zip",
    },
    "cp": {
        "url": "https://www.cp.pt/StaticFiles/CP/imagens/PDF/CP_GTFS.zip",
        "fallback": None,
    },
}

REQUIRED_FILES = {"stops.txt", "trips.txt", "stop_times.txt"}


def _is_already_extracted(dest: Path) -> bool:
    if not dest.exists():
        return False
    present = {f.name for f in dest.iterdir() if f.is_file()}
    return REQUIRED_FILES.issubset(present)


def _download_zip(url: str, dest_zip: Path, timeout: int = 60) -> bool:
    print(f"    GET {url}")
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        with open(dest_zip, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    [FAIL] Falhou: {e}")
        return False


def download_all(force: bool = False) -> dict:
    """
    Download and extract all GTFS feeds.

    Args:
        force: Re-download even if the directory already exists.

    Returns:
        dict mapping operator name to True (success) or False (failed).
    """
    GTFS_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}

    for operator, cfg in OPERATORS.items():
        dest = GTFS_ROOT / operator
        dest_zip = GTFS_ROOT / f"{operator}.zip"

        if not force and _is_already_extracted(dest):
            print(f"  [OK] {operator}: ja existe, a saltar.")
            results[operator] = True
            continue

        print(f"\n  [D] {operator}")

        urls_to_try = [cfg["url"]]
        if cfg.get("fallback"):
            urls_to_try.append(cfg["fallback"])

        downloaded = False
        for url in urls_to_try:
            if _download_zip(url, dest_zip):
                downloaded = True
                break
            time.sleep(1)

        if not downloaded:
            print(f"    [FAIL] {operator}: todos os URLs falharam.")
            results[operator] = False
            continue

        # Extract
        try:
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            with zipfile.ZipFile(dest_zip, "r") as z:
                z.extractall(dest)
            dest_zip.unlink(missing_ok=True)

            present = {f.name for f in dest.iterdir() if f.is_file()}
            missing = REQUIRED_FILES - present
            if missing:
                print(f"    AVISO: {operator}: zip extraido mas faltam {missing}")
                results[operator] = False
            else:
                print(f"    [OK] {operator}: OK ({len(list(dest.iterdir()))} ficheiros)")
                results[operator] = True

        except Exception as e:
            print(f"    [FAIL] {operator}: erro na extraccao: {e}")
            results[operator] = False

    return results


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    print("=== Download GTFS Portugal ===\n")
    results = download_all(force=force)
    ok = sum(v for v in results.values())
    print(f"\nResultado: {ok}/{len(results)} operadores OK")
    for op, status in results.items():
        mark = "[OK]" if status else "[FAIL]"
        print(f"  {mark} {op}")
    sys.exit(0 if ok > 0 else 1)
