"""Summarize failure_rate.py runs: realized overestimation D/CH against kappa, and error tails against eps and delta.

Usage: python3 summarize_failure.py runs/*.jsonl
Each file name must look like <dataset>_l<norm>.jsonl. Prints, per dataset, norm and eps: T, failures, the one-sided
95% exact (Clopper-Pearson) upper confidence bound on the failure probability, and the mean, 90th percentile and
maximum relative error; and per dataset and norm the mean realized D/CH and conditional relative variance.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

KAPPA_BETA1 = {1: 2.136466, 2: 2.120719}  # certified at c = 2, beta = 1 (the same for every dataset)
KAPPA = {  # earlier certified upper bounds at c = 2, beta = min(100/n_B, 1), round-end stopping
    ("federalist", 1): 2.180562, ("federalist", 2): 2.047861,
    ("mnist", 1): 2.141171, ("mnist", 2): 2.060613,
    ("trevi", 1): 2.166167, ("trevi", 2): 2.079964,
    ("gist", 1): 2.156947, ("gist", 2): 2.088774,
}


def binomial_cdf(x: int, n: int, p: float) -> float:
    return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(x + 1))


def upper_bound(failures: int, n: int, level: float = 0.95) -> float:
    """Smallest p with Pr[Bin(n, p) <= failures] <= 1 - level (exact one-sided upper bound)."""
    if failures >= n:
        return 1.0
    lo, hi = failures / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if binomial_cdf(failures, n, mid) > 1 - level:
            lo = mid
        else:
            hi = mid
    return hi


def main() -> None:
    for path in sys.argv[1:]:
        name = os.path.basename(path).removesuffix(".jsonl")
        dataset, norm = name.rsplit("_l", 1)
        rows = [json.loads(line) for line in open(path)]
        n = len(rows)
        ratio = np.array([row["D_over_CH"] for row in rows])
        rel_var = np.array([row.get("cond_rel_var", float("nan")) for row in rows])
        print(f"{dataset} l{norm}: runs {n}, D/CH mean {ratio.mean():.4f} max {ratio.max():.4f} "
              f"(kappa {KAPPA_BETA1[int(norm)]}), cond. rel. variance mean {np.nanmean(rel_var):.4f}, "
              f"crude wall median {np.median([row['crude_wall_s'] for row in rows]):.2f} s")
        for eps in rows[0]["errors"]:
            errors = np.array([row["errors"][eps]["rel_error"] for row in rows])
            fails = int((errors > float(eps)).sum())
            print(f"    eps {eps:>4}: T {rows[0]['errors'][eps]['T']:>6}, failures {fails}/{n} "
                  f"(95% UB {100 * upper_bound(fails, n):.2f}%), mean {100 * errors.mean():.3f}%, "
                  f"p90 {100 * np.quantile(errors, 0.9):.3f}%, max {100 * errors.max():.3f}%")


if __name__ == "__main__":
    main()
