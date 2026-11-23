"""HNSW crude distances as an empirical CrudeNN: cold time and accuracy at the QALSH4C sample size.

Usage: python3 hnsw_summary.py FINAL_DIR BINARY NAME=DATASET_DIR ...
Reads FINAL_DIR/hnsw/<ds>/hnsw_timing.json and hnsw_ef<ef>.bin, FINAL_DIR/exact/<ds>_l2.bin, the l2 rows of
FINAL_DIR/cost_summary.json (for the sampling-stage time) and T(eps = 0.1) for l2 from FINAL_DIR/kappa.txt. The total
is graph construction + one HNSW query per source vector + the same sampling stage as QALSH4C. The C++ binary forms the
400 estimates per efSearch from the HNSW crude distances, looking up the exact distances.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_estimates import estimates  # noqa: E402


def main() -> None:
    final, binary = sys.argv[1], sys.argv[2]
    dirs = dict(arg.split("=", 1) for arg in sys.argv[3:])
    cost = {(r["dataset"], r["norm"]): r for r in json.load(open(f"{final}/cost_summary.json")) if r["threads"] == 24}
    T = None
    for line in open(f"{final}/kappa.txt"):
        f = dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)
        if line.startswith("main ") and f.get("norm") == "2" and f.get("c") == "2" and f.get("epsilon") == "0.1":
            T = int(f["T"])
    for d in sorted(glob.glob(f"{final}/hnsw/*")):
        ds = os.path.basename(d)
        timing = json.load(open(f"{d}/hnsw_timing.json"))
        exact_path = f"{final}/exact/{ds}_l2.bin"
        exact = np.fromfile(exact_path, dtype=np.float64)
        ch = exact.sum()
        c = cost[(ds, 2)]
        print(f"{ds}: build {timing['build_s']:.2f} s; exact {c['exact_s']:.2f} s; QALSH4C {c['qalsh4c_s']:.2f} s "
              f"(D/CH {c['D_over_CH']:.3f})")
        for ef, q in sorted(timing["queries"].items(), key=lambda kv: int(kv[0])):
            weights_path = f"{d}/hnsw_ef{ef}.bin"
            total = np.fromfile(weights_path, dtype=np.float32).astype(np.float64).sum()
            eta = np.array(estimates(binary, dirs[ds], 2, T, 400, exact_path, ["precomputed", "-f", weights_path]))
            err = np.abs(eta - ch) / ch
            cold = timing["build_s"] + q["query_s"] + c["sampling_s"]
            print(f"   ef={ef}: query {q['query_s']:.2f} s, cold total {cold:.2f} s, D/CH {total / ch:.4f}, "
                  f"mean error at T={T}: {100 * err.mean():.3f}% (max {100 * err.max():.3f}%)")


if __name__ == "__main__":
    main()
