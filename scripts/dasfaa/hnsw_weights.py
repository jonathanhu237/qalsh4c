"""HNSW-derived crude distances (an empirical CrudeNN without a computable kappa), l2 only.

Builds an hnswlib graph on B, queries every a in A for one neighbor at several efSearch values, and recomputes the
Euclidean distance to the returned point in float64 (hnswlib's l2 space reports squared distances). The distances are
upper bounds D_a >= d(a, B) because they are distances to actual target points, so they can weight the same
importance-sampling estimator; no expected-overestimation bound is derived for them.

Usage: python hnsw_weights.py --dataset DIR --out-dir OUT [--m 32] [--ef-construction 200] [--ef 10 50 200]
Writes OUT/hnsw_ef<ef>.bin (float32 weights) and OUT/hnsw_timing.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import hnswlib
import numpy as np


def load(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    with open(os.path.join(dataset, "metadata.json")) as f:
        meta = json.load(f)
    d = meta["num_dimensions"]
    a = np.fromfile(os.path.join(dataset, "A.bin"), dtype=np.float32).reshape(meta["num_points"]["a"], d)
    b = np.fromfile(os.path.join(dataset, "B.bin"), dtype=np.float32).reshape(meta["num_points"]["b"], d)
    return a, b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef", type=int, nargs="+", default=[10, 50, 200])
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    a, b = load(args.dataset)
    index = hnswlib.Index(space="l2", dim=a.shape[1])
    index.set_num_threads(args.threads)
    start = time.perf_counter()
    index.init_index(max_elements=b.shape[0], ef_construction=args.ef_construction, M=args.m, random_seed=args.seed)
    index.add_items(b, np.arange(b.shape[0]))
    build_s = time.perf_counter() - start
    timing = {"build_s": build_s, "M": args.m, "ef_construction": args.ef_construction, "queries": {}}
    for ef in args.ef:
        index.set_ef(ef)
        start = time.perf_counter()
        labels, _ = index.knn_query(a, k=1)
        query_s = time.perf_counter() - start
        nearest = b[labels[:, 0]].astype(np.float64)
        weights = np.sqrt(((a.astype(np.float64) - nearest) ** 2).sum(axis=1)).astype(np.float32)
        weights.tofile(os.path.join(args.out_dir, f"hnsw_ef{ef}.bin"))
        timing["queries"][str(ef)] = {"query_s": query_s}
        print(f"ef={ef}: build {build_s:.2f} s, query {query_s:.2f} s, sum of weights {float(weights.sum()):.6g}",
              flush=True)
    with open(os.path.join(args.out_dir, "hnsw_timing.json"), "w") as f:
        json.dump(timing, f, indent=2)


if __name__ == "__main__":
    main()
