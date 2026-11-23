"""HNSW crude distances as an empirical CrudeNN: cold time and accuracy at the QALSH4C sample size.

Usage: python3 hnsw_summary.py FINAL_DIR
Reads FINAL_DIR/hnsw/<ds>/hnsw_timing.json and hnsw_ef<ef>.bin, FINAL_DIR/exact/<ds>_l2.bin and the l2 rows of
FINAL_DIR/cost_summary.json (for the sampling-stage time at T = 1,121). The total is graph construction + one HNSW
query per source vector + the same sampling stage as QALSH4C.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

T = 1121


def main() -> None:
    final = sys.argv[1]
    cost = {(r["dataset"], r["norm"]): r for r in json.load(open(f"{final}/cost_summary.json")) if r["threads"] == 24}
    rng = np.random.default_rng(20261121)
    for d in sorted(glob.glob(f"{final}/hnsw/*")):
        ds = os.path.basename(d)
        timing = json.load(open(f"{d}/hnsw_timing.json"))
        exact = np.fromfile(f"{final}/exact/{ds}_l2.bin", dtype=np.float64)
        ch = exact.sum()
        c = cost[(ds, 2)]
        print(f"{ds}: build {timing['build_s']:.2f} s; exact {c['exact_s']:.2f} s; QALSH4C {c['qalsh4c_s']:.2f} s "
              f"(D/CH {c['D_over_CH']:.3f})")
        for ef, q in sorted(timing["queries"].items(), key=lambda kv: int(kv[0])):
            w = np.fromfile(f"{d}/hnsw_ef{ef}.bin", dtype=np.float32).astype(np.float64)
            total = w.sum()
            prob = w / total
            sup = np.flatnonzero(prob > 0)
            picks = rng.choice(sup, size=(400, T), replace=True, p=prob[sup])
            eta = np.mean(total / w[picks] * exact[picks], axis=1)
            err = np.abs(eta - ch) / ch
            cold = timing["build_s"] + q["query_s"] + c["sampling_s"]
            print(f"   ef={ef}: query {q['query_s']:.2f} s, cold total {cold:.2f} s, D/CH {total / ch:.4f}, "
                  f"mean error at T={T}: {100 * err.mean():.3f}% (max {100 * err.max():.3f}%)")


if __name__ == "__main__":
    main()
