"""Time-accuracy comparison of QALSH4C, uniform sampling and quadtree weights at equal running time.

Costs follow the measured stages: QALSH4C costs I + C + T e (index, crude queries, T exact queries), quadtree weights
cost Q + T e, uniform sampling costs T e, where e is the measured cost of one exact query in the sampling stage
(sampling time / T of the median QALSH4C run). For each time budget, every method gets the largest T it can afford,
and its relative error is measured with exact distances looked up (QALSH4C: the crude distances of the 50
full-randomness runs, a few draws each; uniform and quadtree: many draws).

Usage: python3 time_accuracy.py FINAL_DIR cost_summary.json out.json [T_l1 T_l2]
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

FRACTIONS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)  # budgets as fractions of the exact-scan time


def errors_is(weights: np.ndarray, exact: np.ndarray, t: int, draws: int, rng) -> np.ndarray:
    ch = exact.sum()
    total = weights.sum()
    prob = weights / total
    support = np.flatnonzero(prob > 0)
    picks = rng.choice(support, size=(draws, t), replace=True, p=prob[support])
    eta = np.mean(total / weights[picks] * exact[picks], axis=1)
    return np.abs(eta - ch) / ch


def errors_uniform(exact: np.ndarray, t: int, draws: int, rng) -> np.ndarray:
    ch = exact.sum()
    picks = rng.integers(0, exact.size, size=(draws, t))
    eta = exact[picks].mean(axis=1) * exact.size
    return np.abs(eta - ch) / ch


def summary(err: np.ndarray) -> dict:
    return {"mean": float(err.mean()), "p90": float(np.quantile(err, 0.9)), "max": float(err.max()), "n": int(err.size)}


def main() -> None:
    final, cost_path, out_path = sys.argv[1:4]
    t_main = {1: int(sys.argv[4]) if len(sys.argv) > 4 else 1137, 2: int(sys.argv[5]) if len(sys.argv) > 5 else 1121}
    rng = np.random.default_rng(20261121)
    costs = {(r["dataset"], r["norm"]): r for r in json.load(open(cost_path)) if r["threads"] == 24}
    result = {}
    for (ds, p), c in sorted(costs.items()):
        exact_path = os.path.join(final, "exact", f"{ds}_l{p}.bin")
        runs = sorted(glob.glob(os.path.join(final, "randomness", "weights", f"{ds}_l{p}", "run*.bin")))
        if not os.path.exists(exact_path) or not runs:
            continue
        exact = np.fromfile(exact_path, dtype=np.float64)
        weights = [np.fromfile(r, dtype=np.float32).astype(np.float64) for r in runs]
        t_q = t_main[p]
        e = c["sampling_s"] / t_q
        fixed_q = c["index_s"] + c["crude_s"]
        quad_path = os.path.join(final, "quadtree", f"{ds}_l{p}.bin")
        quad = np.fromfile(quad_path, dtype=np.float32).astype(np.float64) if os.path.exists(quad_path) else None
        fixed_quad = (c["quadtree_s"] - c["sampling_s"]) if (quad is not None and c.get("quadtree_s")) else None
        budgets = sorted({f * c["exact_s"] for f in FRACTIONS} | {c["qalsh4c_s"]})
        entry = {"exact_s": c["exact_s"], "e_s": e, "qalsh4c_fixed_s": fixed_q, "quadtree_fixed_s": fixed_quad,
                 "cv2_uniform": float(exact.size * np.sum(exact ** 2) / exact.sum() ** 2 - 1), "budgets": []}
        vq = []
        for w in weights:
            sup = w > 0
            vq.append(float(w.sum() * np.sum(exact[sup] ** 2 / w[sup]) / exact.sum() ** 2 - 1))
        entry["cond_rel_var_qalsh4c_mean"] = float(np.mean(vq))
        for b in budgets:
            row = {"budget_s": b, "fraction_of_exact": b / c["exact_s"]}
            tu = int(b / e)
            if tu >= 1:
                row["uniform"] = {"T": tu, **summary(errors_uniform(exact, min(tu, 10**6), 400, rng))}
            tq = int((b - fixed_q) / e)
            if tq >= 1:
                err = np.concatenate([errors_is(w, exact, tq, 8, rng) for w in weights])
                row["qalsh4c"] = {"T": tq, **summary(err)}
            if quad is not None and fixed_quad is not None:
                tqt = int((b - fixed_quad) / e)
                if tqt >= 1:
                    row["quadtree"] = {"T": tqt, **summary(errors_is(quad, exact, tqt, 400, rng))}
            entry["budgets"].append(row)
        result[f"{ds}_l{p}"] = entry
        at = [r for r in entry["budgets"] if abs(r["budget_s"] - c["qalsh4c_s"]) < 1e-9][0]
        print(f"{ds}_l{p}: CV2={entry['cv2_uniform']:.4f} V_q4c={entry['cond_rel_var_qalsh4c_mean']:.4f} at QALSH4C's "
              f"time {c['qalsh4c_s']:.2f}s: uniform T={at.get('uniform', {}).get('T')} mean "
              f"{100 * at.get('uniform', {}).get('mean', float('nan')):.3f}%, QALSH4C mean "
              f"{100 * at['qalsh4c']['mean']:.3f}% (max {100 * at['qalsh4c']['max']:.3f}%), quadtree "
              f"{100 * at.get('quadtree', {}).get('mean', float('nan')):.3f}%", flush=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
