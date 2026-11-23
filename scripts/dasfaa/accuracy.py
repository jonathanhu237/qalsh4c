"""Mean relative error of each sampling method at equal sample sizes, from saved weights and exact distances.

QALSH4C uses the crude distances of the full-randomness runs (one estimate per run and T, so the error includes the
randomness of the projections). Quadtree weights are deterministic and HNSW weights come from one graph; for them and
for uniform sampling, the error is over --repeats sampling draws. Exact distances d(a, B) are looked up, which gives
the same estimates as running the exact queries.

Usage: python3 accuracy.py --root RESULTS --out accuracy.json [--config lineup.json]
lineup.json (optional) is {"datasets": [...], "T_eps01": {"<ds>_l<p>": T, ...}}; without it the SIGMOD-era lineup
below is used. RESULTS must hold exact/<ds>_l<p>.bin, failure/weights/<ds>_l<p>/run*.bin, phase2/quadtree/<ds>_l<p>.bin and,
for l2, phase2/hnsw/<ds>/hnsw_ef<ef>.bin.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

DATASETS = ("federalist", "mnist", "trevi", "gist")
GRID = (10, 20, 50, 100, 200, 500, 1000)
T_EPS01 = {("federalist", 1): 1181, ("federalist", 2): 1048, ("mnist", 1): 1142, ("mnist", 2): 1061,
           ("trevi", 1): 1167, ("trevi", 2): 1080, ("gist", 1): 1157, ("gist", 2): 1089}


def estimate_errors(weights: np.ndarray, exact: np.ndarray, t: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    ch = exact.sum()
    total = weights.sum()
    support = np.flatnonzero(weights > 0)
    prob = weights[support] / total
    picks = rng.choice(support, size=(draws, t), replace=True, p=prob)
    eta = (total / weights[picks] * exact[picks]).mean(axis=1)
    return np.abs(eta - ch) / ch


def uniform_errors(exact: np.ndarray, t: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    ch = exact.sum()
    picks = rng.integers(0, exact.size, size=(draws, t))
    eta = exact.size * exact[picks].mean(axis=1)
    return np.abs(eta - ch) / ch


def summarize(errors: np.ndarray) -> dict:
    return {"mean": float(errors.mean()), "p90": float(np.quantile(errors, 0.9)), "n": int(errors.size)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--config", default=None, help="JSON with the datasets and T(eps = 0.1) per dataset and norm")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    datasets = DATASETS
    t_eps01 = dict(T_EPS01)
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        datasets = tuple(config["datasets"])
        t_eps01 = {(key.rsplit("_l", 1)[0], int(key.rsplit("_l", 1)[1])): int(t)
                   for key, t in config["T_eps01"].items()}
    result: dict = {}
    for ds in datasets:
        for p in (1, 2):
            key = f"{ds}_l{p}"
            exact_path = os.path.join(args.root, "exact", f"{key}.bin")
            if not os.path.exists(exact_path):
                continue
            exact = np.fromfile(exact_path, dtype=np.float64)
            grid = sorted(set(GRID) | {t_eps01[(ds, p)]})
            entry: dict = {"T_eps01": t_eps01[(ds, p)], "methods": {}}
            runs = sorted(glob.glob(os.path.join(args.root, "failure", "weights", key, "run*.bin")))
            if runs:
                weights = [np.fromfile(path, dtype=np.float32).astype(np.float64) for path in runs]
                entry["methods"]["qalsh4c"] = {
                    str(t): summarize(np.concatenate([estimate_errors(w, exact, t, 1, rng) for w in weights]))
                    for t in grid}
                entry["D_over_CH"] = float(np.mean([w.sum() for w in weights]) / exact.sum())
            entry["methods"]["uniform"] = {str(t): summarize(uniform_errors(exact, t, args.repeats, rng)) for t in grid}
            quad = os.path.join(args.root, "phase2", "quadtree", f"{key}.bin")
            if os.path.exists(quad):
                w = np.fromfile(quad, dtype=np.float32).astype(np.float64)
                entry["methods"]["quadtree"] = {str(t): summarize(estimate_errors(w, exact, t, args.repeats, rng))
                                                for t in grid}
                entry["quadtree_D_over_CH"] = float(w.sum() / exact.sum())
            if p == 2:
                for path in sorted(glob.glob(os.path.join(args.root, "phase2", "hnsw", ds, "hnsw_ef*.bin"))):
                    ef = os.path.basename(path).removeprefix("hnsw_ef").removesuffix(".bin")
                    w = np.fromfile(path, dtype=np.float32).astype(np.float64)
                    entry["methods"][f"hnsw_ef{ef}"] = {
                        str(t): summarize(estimate_errors(w, exact, t, args.repeats, rng)) for t in grid}
                    entry[f"hnsw_ef{ef}_D_over_CH"] = float(w.sum() / exact.sum())
            result[key] = entry
            print(key, {m: round(v[str(entry['T_eps01'])]["mean"] * 100, 3) for m, v in entry["methods"].items()},
                  flush=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
