"""LaTeX tables for the DASFAA experiments from the final summaries.

Usage: python3 make_tables.py FINAL_DIR > tables.tex
Reads cost_summary.json (summarize_cost.py), contrast.json (contrast.py), randomness/*.jsonl (failure_rate.py),
time_accuracy.json (time_accuracy.py) and csweep/*.jsonl, and prints the dataset, cost, full-randomness and c tables.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

NAMES = {"p53": "P53", "trevi": "Trevi", "notredame": "Notre Dame", "dbpedia": "DBpedia", "enron": "Enron",
         "gist": "GIST", "mnist": "MNIST", "federalist": "Federalist"}
ORDER = ["p53", "trevi", "notredame", "dbpedia", "enron", "gist", "mnist", "federalist"]
KAPPA = {1: 2.136466, 2: 2.120719}


def fmt_s(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def upper_bound(failures: int, n: int, level: float = 0.95) -> float:
    if failures >= n:
        return 1.0
    lo, hi = failures / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        cdf = sum(math.comb(n, k) * mid ** k * (1 - mid) ** (n - k) for k in range(failures + 1))
        lo, hi = (mid, hi) if cdf > 1 - level else (lo, mid)
    return hi


def main() -> None:
    final = sys.argv[1]
    cost = {(r["dataset"], r["norm"]): r for r in json.load(open(f"{final}/cost_summary.json")) if r["threads"] == 24}
    ta = json.load(open(f"{final}/time_accuracy.json")) if os.path.exists(f"{final}/time_accuracy.json") else {}
    contrast = {}
    if os.path.exists(f"{final}/contrast.json"):
        for r in json.load(open(f"{final}/contrast.json")):
            contrast[(r["dataset"], r["norm"])] = r
    print("% ---- cost table: dataset & norm & exact & index & crude & sampling & total & speedup & D/CH & err "
          "q4c & err uniform & err quadtree")
    for ds in ORDER:
        for p in (1, 2):
            c = cost.get((ds, p))
            if c is None:
                continue
            exact = c.get("best_exact_s") or (min(c["exact_s"], c["faiss_s"])
                                              if not math.isnan(c.get("faiss_s", float("nan"))) else c["exact_s"])
            row = [NAMES[ds] if p == 1 else "", f"$\\ell_{p}$", fmt_s(exact), fmt_s(c["index_s"]), fmt_s(c["crude_s"]),
                   fmt_s(c["sampling_s"]), fmt_s(c["qalsh4c_s"]), f"{exact / c['qalsh4c_s']:.1f}",
                   f"{c['D_over_CH']:.2f}"]
            entry = ta.get(f"{ds}_l{p}")
            if entry:
                at = min(entry["budgets"], key=lambda b: abs(b["budget_s"] - c["qalsh4c_s"]))
                for method in ("qalsh4c", "uniform", "quadtree"):
                    v = at.get(method)
                    row.append(f"{100 * v['mean']:.2f}" if v else "--")
            print(" & ".join(row) + " \\\\")
    print("% ---- full randomness + equal time: dataset & norm & D/CH mean & max err eps=0.3 & mean err eps=0.1 & "
          "max err eps=0.1 & equal-time mean err q4c & uniform & quadtree (errors in %); failures listed in comments")
    for ds in ORDER:
        for p in (1, 2):
            path = f"{final}/randomness/{ds}_l{p}.jsonl"
            if not os.path.exists(path):
                continue
            rows = [json.loads(line) for line in open(path)]
            ratio = np.array([r["D_over_CH"] for r in rows])
            e3 = np.array([r["errors"]["0.3"]["rel_error"] for r in rows])
            e1 = np.array([r["errors"]["0.1"]["rel_error"] for r in rows])
            cells = [NAMES[ds] if p == 1 else "", f"$\\ell_{p}$", f"{ratio.mean():.2f}", f"{100 * e3.max():.2f}",
                     f"{100 * e1.mean():.2f}", f"{100 * e1.max():.2f}"]
            entry = ta.get(f"{ds}_l{p}")
            c = cost.get((ds, p))
            if entry and c:
                at = min(entry["budgets"], key=lambda b: abs(b["budget_s"] - c["qalsh4c_s"]))
                for method in ("qalsh4c", "uniform", "quadtree"):
                    v = at.get(method)
                    cells.append(f"{100 * v['mean']:.2f}" if v else "--")
            fails = (int((e3 > 0.3).sum()), int((e1 > 0.1).sum()))
            print(" & ".join(cells) + f" \\\\  % runs {len(rows)}, failures eps0.3/0.1 {fails}, "
                  f"cond var {np.mean([r['cond_rel_var'] for r in rows]):.4f}")
    print("% ---- datasets: name & d & |A| & |B| & RC_he l2")
    for ds in ORDER:
        r = contrast.get((ds, 2))
        if r:
            print(f"{NAMES[ds]} & {r['d']} & -- & {r['n_b']} & {r['relative_contrast_he']:.2f} \\\\")
    print("% ---- c sweep: dataset & norm & c & T & D/CH & fails & max err & crude wall")
    for path in sorted(glob.glob(f"{final}/csweep/*.jsonl")):
        name = os.path.basename(path).removesuffix(".jsonl")
        rows = [json.loads(line) for line in open(path)]
        err = np.array([r["errors"]["0.1"]["rel_error"] for r in rows])
        print(f"{name} & T={rows[0]['errors']['0.1']['T']} & D/CH={np.mean([r['D_over_CH'] for r in rows]):.3f} & "
              f"fails={int((err > 0.1).sum())}/{len(rows)} & max={100 * err.max():.2f}% & "
              f"queries_s={np.median([r.get('queries_ms', float('nan')) for r in rows]) / 1000:.2f} \\\\")


if __name__ == "__main__":
    main()
