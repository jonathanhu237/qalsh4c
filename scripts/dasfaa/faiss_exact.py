"""Exact Chamfer distance with FAISS brute force (IndexFlatL2 / IndexFlat with METRIC_L1), as a reference for our own
exact scan. The search is batched over all source points with k = 1; for l2, FAISS returns squared distances, so we
take square roots before summing.

Usage: python faiss_exact.py --dataset DIR --norm 1|2 --threads 24 [--repeats 3] [--exact EXACT.bin] [--l1-queries N]
Prints one JSON line per repeat: load time (reading A.bin and B.bin), compute time (add + search) and the Chamfer sum.

FAISS has no blocked kernel for l1: IndexFlat with METRIC_L1 streams all of B once per query, so a full run takes hours
on our largest sets. For l1 the script therefore times the first N source points (default 1,000) and reports the
time scaled to |A| as compute_s (compute_s_measured is the raw time); the Chamfer check then covers those N points.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import faiss
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--norm", type=int, choices=(1, 2), required=True)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--exact", default=None, help="float64 d(a,B) from exact_nn, to check the result")
    parser.add_argument("--l1-queries", type=int, default=1000, help="Source points timed for l1 (0 = all)")
    args = parser.parse_args()
    faiss.omp_set_num_threads(args.threads)
    with open(os.path.join(args.dataset, "metadata.json")) as f:
        meta = json.load(f)
    d, na, nb = meta["num_dimensions"], meta["num_points"]["a"], meta["num_points"]["b"]
    for rep in range(args.repeats):
        start = time.perf_counter()
        a = np.fromfile(os.path.join(args.dataset, "A.bin"), dtype=np.float32).reshape(na, d)
        nq = na if (args.norm == 2 or args.l1_queries <= 0) else min(na, args.l1_queries)
        a = np.ascontiguousarray(a[:nq])
        b = np.fromfile(os.path.join(args.dataset, "B.bin"), dtype=np.float32).reshape(nb, d)
        loaded = time.perf_counter()
        index = faiss.IndexFlatL2(d) if args.norm == 2 else faiss.IndexFlat(d, faiss.METRIC_L1)
        index.add(b)
        dist, _ = index.search(a, 1)
        done = time.perf_counter()
        values = dist[:, 0].astype(np.float64)
        if args.norm == 2:
            values = np.sqrt(np.maximum(values, 0.0))
        record = {"dataset": os.path.basename(os.path.normpath(args.dataset)), "norm": args.norm,
                  "threads": args.threads, "repeat": rep, "load_s": round(loaded - start, 4),
                  "queries_timed": nq, "compute_s_measured": round(done - loaded, 4),
                  "compute_s": round((done - loaded) * na / nq, 4), "chamfer_timed": float(values.sum())}
        if args.exact:
            exact = np.fromfile(args.exact, dtype=np.float64)[:nq]
            record["rel_diff_vs_exact"] = float(abs(values.sum() - exact.sum()) / exact.sum())
        print(json.dumps(record), flush=True)
        del index, a, b


if __name__ == "__main__":
    main()
