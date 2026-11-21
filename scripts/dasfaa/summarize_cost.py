"""Summarize the final cost runs (final/cost.txt and final/faiss.jsonl) into one row per dataset and norm.

For each dataset, norm and thread count: median exact time (ours), FAISS compute time, median QALSH4C total and the
stage split of the run with the median total (index, crude queries, sampling = total - weights), the speedup over the
faster exact implementation, D/CH, uniform and quadtree times.

Usage: python3 summarize_cost.py FINAL_DIR [--json out.json]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def main() -> None:
    final = sys.argv[1]
    rows = defaultdict(lambda: defaultdict(list))
    with open(f"{final}/cost.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            ds, p, threads, rep, method = parts[:5]
            values = parts[5:]
            key = (ds, int(p), int(threads))
            rows[key][method].append(values)
    faiss = defaultdict(list)
    try:
        with open(f"{final}/faiss.jsonl") as f:
            for line in f:
                r = json.loads(line)
                faiss[(r["dataset"], r["norm"], r["threads"])].append(r)
    except FileNotFoundError:
        pass
    names = {"dbpedia3072-b200k": "dbpedia", "notredame-a20k": "notredame", "p53-half": "p53", "trevi-a20k": "trevi",
             "gist": "gist", "enron-a20k": "enron", "mnist-a20k": "mnist", "federalist": "federalist"}
    out = []
    print(f"{'dataset':10s} {'p':>2s} {'thr':>3s} {'exact_s':>8s} {'faiss_s':>8s} {'q4c_s':>7s} {'index':>6s} "
          f"{'crude':>6s} {'sample':>6s} {'speedup':>7s} {'D/CH':>6s} {'unif_s':>7s} {'quad_s':>7s} hits/q eval/q")
    for (ds, p, threads), methods in sorted(rows.items()):
        exact = [float(v[0]) / 1000 for v in methods.get("exact", []) if v]
        q = [v for v in methods.get("qalsh4c", []) if v]
        if not exact or not q:
            continue
        q_sorted = sorted(q, key=lambda v: float(v[0]))
        mid = q_sorted[len(q_sorted) // 2]
        total = float(mid[0]) / 1000
        index, crude, weights = float(mid[1]) / 1000, float(mid[2]) / 1000, float(mid[3]) / 1000
        sampling = total - weights
        dch = float(mid[4])
        stats = " ".join(mid[5:]).strip("[]")
        hits = next((s.split("=")[1] for s in stats.split() if s.startswith("hits_per_query=")), "-")
        evaluated = next((s.split("=")[1] for s in stats.split() if s.startswith("evaluated_per_query=")), "-")
        # FAISS total = reading the vectors (numpy) + add + search, comparable to our times, which include loading.
        fz = [r["load_s"] + r["compute_s"] for (name, norm, thr), rs in faiss.items() for r in rs
              if names.get(name, name) == ds and norm == p and thr == threads]
        faiss_s = min(fz) if fz else float("nan")
        exact_med = statistics.median(exact)
        best_exact = min(exact_med, faiss_s) if fz else exact_med
        unif = [float(v[0]) / 1000 for v in methods.get("uniform", []) if v]
        quad = [float(v[0]) / 1000 for v in methods.get("quadtree", []) if v]
        row = {"dataset": ds, "norm": p, "threads": threads, "exact_s": exact_med, "faiss_s": faiss_s,
               "qalsh4c_s": total, "index_s": index, "crude_s": crude, "sampling_s": sampling,
               "speedup": best_exact / total, "D_over_CH": dch, "uniform_s": unif[0] if unif else None,
               "quadtree_s": quad[0] if quad else None, "hits_per_query": hits, "evaluated_per_query": evaluated,
               "qalsh4c_all_s": [float(v[0]) / 1000 for v in q], "exact_all_s": exact}
        out.append(row)
        print(f"{ds:10s} {p:2d} {threads:3d} {exact_med:8.2f} {faiss_s:8.2f} {total:7.2f} {index:6.2f} {crude:6.2f} "
              f"{sampling:6.2f} {row['speedup']:7.2f} {dch:6.3f} {(unif[0] if unif else float('nan')):7.2f} "
              f"{(quad[0] if quad else float('nan')):7.2f} {hits} {evaluated}")
    # Best exact baseline per dataset and norm over implementations and thread counts (QALSH4C stays at 24 threads,
    # where the thread check found it fastest).
    print("\nbest exact baseline (min over ours and FAISS at 12 and 24 threads) and the speedup of QALSH4C at 24:")
    by_cell = {}
    for (ds, p, threads), methods in rows.items():
        exact = [float(v[0]) / 1000 for v in methods.get("exact", []) if v]
        if exact:
            by_cell.setdefault((ds, p), []).append((statistics.median(exact), f"ours@{threads}"))
    for (name, norm, thr), rs in faiss.items():
        key = (names.get(name, name), norm)
        by_cell.setdefault(key, []).append((min(r["load_s"] + r["compute_s"] for r in rs), f"faiss@{thr}"))
    for row in out:
        if row["threads"] != 24:
            continue
        options = by_cell.get((row["dataset"], row["norm"]), [])
        if options:
            best, which = min(options)
            row["best_exact_s"], row["best_exact_impl"] = best, which
            row["best_speedup"] = best / row["qalsh4c_s"]
            print(f"{row['dataset']:10s} {row['norm']} best exact {best:8.3f} s ({which:9s}) QALSH4C {row['qalsh4c_s']:7.3f} s "
                  f"speedup {row['best_speedup']:5.2f}   all: " + ", ".join(f"{w}={t:.2f}" for t, w in sorted(options)))
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
