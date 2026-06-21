"""
evals/sample_fixtures.py
========================
Cria uma amostra de 100 fixtures representativa em TODOS os campos relevantes,
usando K-Means clustering sobre as features normalizadas.

Para cada cluster seleciona o fixture mais próximo do centróide — garante
cobertura máxima de profile, duração, grupo, corredor, crianças, mobilidade,
nightlife, alojamento e n_pois simultaneamente.

Uso:
    python evals/sample_fixtures.py
    python evals/sample_fixtures.py --n 100 --seed 42 --out data/bench_fixtures_grid100
"""

import json, argparse, shutil, math
import numpy as np
from pathlib import Path
from collections import Counter

FIXTURE_DIR = Path("data/bench_fixtures_direct")
DEFAULT_OUT  = Path("data/bench_fixtures_grid100")


def load_fixtures(fixture_dir: Path):
    files = sorted(fixture_dir.glob("*.json"))
    records = []
    for f in files:
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        up = d.get("user_prefs", {})
        records.append({
            "file": f,
            "profile": d.get("profile", "?"),
            "max_time": up.get("max_time", 480),
            "n_pois": d.get("n_pois", 20),
            "num_people": up.get("num_people", 1),
            "is_corridor": int(up.get("is_corridor", False)),
            "has_children": int(up.get("has_children", False)),
            "mobility_issues": int(up.get("mobility_issues", False)),
            "has_nightlife": int(up.get("has_nightlife", False)),
            "include_accommodation": int(up.get("include_accommodation", True)),
            "max_cost": up.get("max_cost") or 0,
            "max_radius_km": up.get("max_radius_km", 30),
        })
    return records


def encode_features(records):
    """
    Codifica cada fixture como vetor numérico normalizado [0,1].
    Profile -> one-hot; variáveis contínuas -> min-max; binárias -> 0/1.
    """
    profiles = sorted({r["profile"] for r in records})
    prof_idx  = {p: i for i, p in enumerate(profiles)}

    rows = []
    for r in records:
        # One-hot profile (7 dims)
        ph = [0.0] * len(profiles)
        ph[prof_idx[r["profile"]]] = 1.0

        row = ph + [
            r["max_time"],
            r["n_pois"],
            r["num_people"],
            float(r["is_corridor"]),
            float(r["has_children"]),
            float(r["mobility_issues"]),
            float(r["has_nightlife"]),
            float(r["include_accommodation"]),
            r["max_cost"],
            r["max_radius_km"],
        ]
        rows.append(row)

    X = np.array(rows, dtype=float)

    # Min-max normalização por coluna (evita que max_time domine)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # evitar divisão por zero
    X = (X - mins) / ranges

    return X, profiles


def kmeans(X, k, seed=42, max_iter=300):
    """K-Means simples sem dependências externas."""
    rng = np.random.default_rng(seed)
    # Inicialização: K-Means++ simplificado
    centers = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        dists = np.array([min(np.sum((x - c)**2) for c in centers) for x in X])
        probs = dists / dists.sum()
        centers.append(X[rng.choice(len(X), p=probs)])
    centers = np.array(centers)

    labels = np.zeros(len(X), dtype=int)
    for iteration in range(max_iter):
        # Atribuição
        dists = np.array([[np.sum((x - c)**2) for c in centers] for x in X])
        new_labels = dists.argmin(axis=1)
        if np.all(new_labels == labels):
            print(f"  K-Means convergiu na iteração {iteration+1}")
            break
        labels = new_labels
        # Actualização dos centróides
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                centers[j] = members.mean(axis=0)

    return labels, centers


def main(n: int = 100, seed: int = 42, out_dir: Path = DEFAULT_OUT):
    print(f"A carregar fixtures de {FIXTURE_DIR}...")
    records = load_fixtures(FIXTURE_DIR)
    print(f"  {len(records)} fixtures carregadas")

    print("A codificar features...")
    X, profiles = encode_features(records)
    print(f"  Vetor de {X.shape[1]} dimensões por fixture")
    print(f"  Perfis detectados: {profiles}")

    print(f"\nA correr K-Means com k={n}...")
    labels, centers = kmeans(X, k=n, seed=seed)

    # Para cada cluster: selecionar o fixture mais próximo do centróide
    selected = []
    for j in range(n):
        members_idx = np.where(labels == j)[0]
        if len(members_idx) == 0:
            continue
        members_X = X[members_idx]
        dists = np.sum((members_X - centers[j])**2, axis=1)
        best = members_idx[dists.argmin()]
        selected.append(records[best])

    print(f"\nFixtures selecionadas: {len(selected)}")

    # Verificar distribuição resultante
    print("\nDistribuição da amostra vs. original:")
    def show_dist(field, selected, records):
        orig = Counter(r[field] for r in records)
        samp = Counter(r[field] for r in selected)
        all_vals = sorted(orig.keys())
        for v in all_vals:
            o = orig[v]; s = samp.get(v, 0)
            print(f"    {field}={v}: original={o} ({o/len(records)*100:.0f}%)  "
                  f"amostra={s} ({s/len(selected)*100:.0f}%)")

    show_dist("profile", selected, records)

    def bucket_time(r):
        t = r["max_time"]
        return "short(<=1d)" if t<=480 else ("medium(2-3d)" if t<=1440 else "long(>3d)")
    def bucket_pois(r):
        n = r["n_pois"]
        return "small(<20)" if n<20 else ("medium(20-50)" if n<=50 else "large(>50)")
    def bucket_group(r):
        n = r["num_people"]
        return "solo(1)" if n==1 else ("small(2-3)" if n<=3 else "large(4+)")

    for label, fn in [("duration", bucket_time), ("n_pois", bucket_pois), ("group", bucket_group)]:
        orig = Counter(fn(r) for r in records)
        samp = Counter(fn(r) for r in selected)
        print(f"\n  {label}:")
        for v in sorted(orig.keys()):
            o = orig[v]; s = samp.get(v, 0)
            print(f"    {v}: original={o} ({o/len(records)*100:.0f}%)  "
                  f"amostra={s} ({s/len(selected)*100:.0f}%)")

    for field in ["is_corridor", "has_children", "mobility_issues",
                  "has_nightlife", "include_accommodation"]:
        orig = Counter(r[field] for r in records)
        samp = Counter(r[field] for r in selected)
        print(f"\n  {field}: original {dict(orig)}  ->  amostra {dict(samp)}")

    # Copiar ficheiros
    out_dir.mkdir(parents=True, exist_ok=True)
    # Limpar pasta se já existia
    for f in out_dir.glob("*.json"):
        f.unlink()

    for r in selected:
        shutil.copy(r["file"], out_dir / r["file"].name)

    ids = sorted(r["file"].stem for r in selected)
    with open(out_dir / "selected_ids.txt", "w") as fp:
        fp.write("\n".join(ids))

    print(f"\nFixtures copiadas para: {out_dir}/")
    print(f"IDs guardados em: {out_dir}/selected_ids.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int,  default=100)
    parser.add_argument("--seed", type=int,  default=42)
    parser.add_argument("--out",  type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(n=args.n, seed=args.seed, out_dir=args.out)
