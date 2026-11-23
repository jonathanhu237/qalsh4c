"""Equal-time accuracy with directly timed competitor budgets.

For uniform sampling and quadtree weights, validate.txt holds the total time with T = 1 (fixed cost: loading, and
for quadtree the weights) and with the modeled T. The per-draw cost is (t_T - t_1) / (T - 1); the corrected count is
the largest T with t_1 + (T - 1) * cost <= the QALSH4C median time. Uniform uses the faster of 12 and 24 threads.
Errors use exact lookups: uniform and quadtree over 400 draws, QALSH4C over 8 draws for each of its 50 weight sets.
Usage: python3 recalib.py FINAL_DIR
"""
import glob, json, os, sys
import numpy as np

final = sys.argv[1]
rng = np.random.default_rng(20261122)
cost = {(r["dataset"], r["norm"]): r for r in json.load(open(f"{final}/cost_summary.json")) if r["threads"] == 24}
meas = {}
for line in open(f"{final}/validate.txt"):
    f = line.split()
    if len(f) == 6 and f[2] in ("uniform", "quadtree"):
        meas.setdefault((f[0], int(f[1]), f[2], int(f[3])), {})[int(f[4])] = float(f[5]) / 1000
T = {1: 1137, 2: 1121}

def err_is(w, exact, t, draws):
    ch = exact.sum(); tot = w.sum(); prob = w / tot; sup = np.flatnonzero(prob > 0)
    picks = rng.choice(sup, size=(draws, t), replace=True, p=prob[sup])
    return np.abs(np.mean(tot / w[picks] * exact[picks], axis=1) - ch) / ch

def err_uniform(exact, t, draws):
    ch = exact.sum(); picks = rng.integers(0, exact.size, size=(draws, t))
    return np.abs(exact[picks].mean(axis=1) * exact.size - ch) / ch

out = {}
for (ds, p), c in sorted(cost.items()):
    runs = sorted(glob.glob(f"{final}/randomness/weights/{ds}_l{p}/run*.bin"))
    if not runs:
        continue
    exact = np.fromfile(f"{final}/exact/{ds}_l{p}.bin", dtype=np.float64)
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
        e = err_uniform(exact, tu, 400)
        row["uniform"] = {"T": tu, "threads": best[1], "mean": float(e.mean()), "max": float(e.max()),
                          "timed_modeled_T": best[5], "timed_s": best[4]}
    m = meas.get((ds, p, "quadtree", 24))
    qpath = f"{final}/quadtree/{ds}_l{p}.bin"
    if m and len(m) >= 2 and os.path.exists(qpath):
        tmod = max(k for k in m if k > 1)
        per = (m[tmod] - m[1]) / (tmod - 1)
        tq = int((budget - m[1]) / per) + 1
        if tq >= 1:
            w = np.fromfile(qpath, dtype=np.float32).astype(np.float64)
            e = err_is(w, exact, tq, 400)
            row["quadtree"] = {"T": tq, "mean": float(e.mean()), "max": float(e.max())}
    weights = [np.fromfile(r, dtype=np.float32).astype(np.float64) for r in runs]
    e = np.concatenate([err_is(w, exact, T[p], 8) for w in weights])
    row["qalsh4c"] = {"T": T[p], "mean": float(e.mean()), "max": float(e.max())}
    out[f"{ds}_l{p}"] = row
    u = row.get("uniform", {}); q = row.get("quadtree", {})
    print(f"{ds:10s} l{p} budget {budget:6.2f}s  QALSH4C {100*row['qalsh4c']['mean']:.2f}%  uniform T={u.get('T')}"
          f"@{u.get('threads')} {100*u.get('mean', float('nan')):.2f}%  quadtree T={q.get('T')} "
          f"{100*q.get('mean', float('nan')):.2f}%")
json.dump(out, open(f"{final}/time_accuracy_timed.json", "w"), indent=1)
