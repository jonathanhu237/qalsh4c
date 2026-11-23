"""Equal-time accuracy with directly timed competitor budgets.

For uniform sampling and quadtree weights, validate.txt holds the total time with T = 1 (fixed cost: loading, and
for quadtree the weights) and with the modeled T. The per-draw cost is (t_T - t_1) / (T - 1); the corrected count is
the largest T with t_1 + (T - 1) * cost <= the QALSH4C median time. Uniform uses the faster of 12 and 24 threads.
The C++ binary forms every estimate, looking up the exact distances: uniform and quadtree over 400 draws, QALSH4C
over 8 draws for each of its 50 crude-distance sets from the full-randomness runs.
Usage: python3 recalib.py FINAL_DIR BINARY NAME=DATASET_DIR ...
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from binary_estimates import estimates  # noqa: E402

final, binary = sys.argv[1], sys.argv[2]
dirs = dict(arg.split("=", 1) for arg in sys.argv[3:])
cost = {(r["dataset"], r["norm"]): r for r in json.load(open(f"{final}/cost_summary.json")) if r["threads"] == 24}
meas = {}
for line in open(f"{final}/validate.txt"):
    f = line.split()
    if len(f) == 6 and f[2] in ("uniform", "quadtree"):
        meas.setdefault((f[0], int(f[1]), f[2], int(f[3])), {})[int(f[4])] = float(f[5]) / 1000
T = {}
for line in open(f"{final}/kappa.txt"):
    f = dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)
    if line.startswith("main ") and f.get("c") == "2" and f.get("epsilon") == "0.1":
        T[int(f["norm"])] = int(f["T"])


def errors(ds, p, t, repeats, method):
    exact_path = f"{final}/exact/{ds}_l{p}.bin"
    ch = np.fromfile(exact_path, dtype=np.float64).sum()
    values = np.array(estimates(binary, dirs[ds], p, t, repeats, exact_path, method))
    return np.abs(values - ch) / ch


out = {}
for (ds, p), c in sorted(cost.items()):
    runs = sorted(glob.glob(f"{final}/randomness/weights/{ds}_l{p}/run*.bin"))
    if not runs or ds not in dirs:
        continue
    budget = c["qalsh4c_s"]
    row = {"budget_s": budget}
    # uniform: best thread count
    best = None
    for thr in (12, 24):
        m = meas.get((ds, p, "uniform", thr))
        if not m or len(m) < 2:
            continue
        tmod = max(k for k in m if k > 1)
        per = (m[tmod] - m[1]) / (tmod - 1)
        tcount = int((budget - m[1]) / per) + 1
        if best is None or tcount > best[0]:
            best = (tcount, thr, m[1], per, m[tmod], tmod)
    if best:
        tu = best[0]
        e = errors(ds, p, tu, 400, ["uniform"])
        row["uniform"] = {"T": tu, "threads": best[1], "mean": float(e.mean()), "max": float(e.max()),
                          "timed_modeled_T": best[5], "timed_s": best[4]}
    m = meas.get((ds, p, "quadtree", 24))
    qpath = f"{final}/quadtree/{ds}_l{p}.bin"
    if m and len(m) >= 2 and os.path.exists(qpath):
        tmod = max(k for k in m if k > 1)
        per = (m[tmod] - m[1]) / (tmod - 1)
        tq = int((budget - m[1]) / per) + 1
        if tq >= 1:
            e = errors(ds, p, tq, 400, ["precomputed", "-f", qpath])
            row["quadtree"] = {"T": tq, "mean": float(e.mean()), "max": float(e.max())}
    e = np.concatenate([errors(ds, p, T[p], 8, ["precomputed", "-f", r]) for r in runs])
    row["qalsh4c"] = {"T": T[p], "mean": float(e.mean()), "max": float(e.max())}
    out[f"{ds}_l{p}"] = row
    u = row.get("uniform", {}); q = row.get("quadtree", {})
    print(f"{ds:10s} l{p} budget {budget:6.2f}s  QALSH4C {100*row['qalsh4c']['mean']:.2f}%  uniform T={u.get('T')}"
          f"@{u.get('threads')} {100*u.get('mean', float('nan')):.2f}%  quadtree T={q.get('T')} "
          f"{100*q.get('mean', float('nan')):.2f}%", flush=True)
json.dump(out, open(f"{final}/time_accuracy_timed.json", "w"), indent=1)
