"""Summarize the final cost runs (final/cost.txt and final/faiss.jsonl) into one row per dataset and norm.

For each dataset, norm and thread count: the mean exact time (ours), the FAISS time, and the mean over the cold
QALSH4C runs (new projections each run) of the total and of each stage (index, crude queries, sampling = total -
weights), with the fastest and slowest total; the speedup over the faster exact implementation, the mean D/CH, and the
uniform and quadtree times. Running time is a cost, so runs are summarized by their arithmetic mean (Hoefler and
Belli, SC15, Rule 3); the spread of QALSH4C comes from its random projections and is part of its cost.

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
        def mean_of(column: int) -> float:
            return statistics.mean(float(v[column]) for v in q)

        totals = [float(v[0]) / 1000 for v in q]
        total = statistics.mean(totals)
        index, crude, weights = mean_of(1) / 1000, mean_of(2) / 1000, mean_of(3) / 1000
        sampling = total - weights
        dch = mean_of(4)

        def mean_stat(name: str) -> float:
            values = [float(item.split("=")[1]) for v in q for item in " ".join(v[5:]).strip("[]").split()
                      if item.startswith(name + "=")]
            return statistics.mean(values) if values else float("nan")

        hits = mean_stat("hits_per_query")
        evaluated = mean_stat("evaluated_per_query")
        fraction = mean_stat("fraction_of_entries")
        # FAISS total = reading the vectors (numpy) + add + search, comparable to our times, which include loading.
        fz = [r["load_s"] + r["compute_s"] for (name, norm, thr), rs in faiss.items() for r in rs
              if names.get(name, name) == ds and norm == p and thr == threads]
        faiss_s = statistics.mean(fz) if fz else float("nan")
        exact_mean = statistics.mean(exact)
        best_exact = min(exact_mean, faiss_s) if fz else exact_mean
        unif = [float(v[0]) / 1000 for v in methods.get("uniform", []) if v]
        quad = [float(v[0]) / 1000 for v in methods.get("quadtree", []) if v]
        row = {"dataset": ds, "norm": p, "threads": threads, "exact_s": exact_mean, "faiss_s": faiss_s,
               "qalsh4c_s": total, "qalsh4c_min_s": min(totals), "qalsh4c_max_s": max(totals), "qalsh4c_runs": len(q),
               "index_s": index, "crude_s": crude, "sampling_s": sampling,
               "speedup": best_exact / total, "D_over_CH": dch, "uniform_s": unif[0] if unif else None,
               "quadtree_s": quad[0] if quad else None, "hits_per_query": hits, "evaluated_per_query": evaluated,
               "fraction_of_entries": fraction, "qalsh4c_all_s": totals, "exact_all_s": exact}
        out.append(row)
        print(f"{ds:10s} {p:2d} {threads:3d} {exact_mean:8.2f} {faiss_s:8.2f} {total:7.2f} {index:6.2f} {crude:6.2f} "
              f"{sampling:6.2f} {row['speedup']:7.2f} {dch:6.3f} {(unif[0] if unif else float('nan')):7.2f} "
              f"{(quad[0] if quad else float('nan')):7.2f} {hits:.0f} {evaluated:.1f}  "
              f"({len(q)} runs, {min(totals):.2f} to {max(totals):.2f} s)")
    # Best exact baseline per dataset and norm over implementations and thread counts (QALSH4C stays at 24 threads,
    # where the thread check found it fastest).
    print("\nbest exact baseline (fastest of ours and FAISS at 12 and 24 threads, each a mean over its runs) and the "
          "speedup of QALSH4C at 24:")
    by_cell = {}
    for (ds, p, threads), methods in rows.items():
        exact = [float(v[0]) / 1000 for v in methods.get("exact", []) if v]
        if exact:
            by_cell.setdefault((ds, p), []).append((statistics.mean(exact), f"ours@{threads}"))
    for (name, norm, thr), rs in faiss.items():
        key = (names.get(name, name), norm)
        by_cell.setdefault(key, []).append((statistics.mean(r["load_s"] + r["compute_s"] for r in rs), f"faiss@{thr}"))
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
