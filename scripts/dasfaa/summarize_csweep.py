"""Summarize the measured c sweep: per dataset, norm and c, the sample size T, the median time of one crude-distance
run, the mean realized D/CH, and the mean, 90th percentile and maximum relative error at eps = 0.1.

Usage: python3 summarize_csweep.py csweep/*.jsonl
"""

from __future__ import annotations

import json
import os
import statistics
import sys

import numpy as np


def main() -> None:
    for path in sorted(sys.argv[1:]):
        rows = [json.loads(line) for line in open(path)]
        name = os.path.basename(path).removesuffix(".jsonl")
        errors = np.array([row["errors"]["0.1"]["rel_error"] for row in rows])
        wall = statistics.median(row["crude_wall_s"] for row in rows)
        ratio = float(np.mean([row["D_over_CH"] for row in rows]))
        t = rows[0]["errors"]["0.1"]["T"]
        print(f"{name:18s} N={len(rows)} T={t:5d} crude_wall_median={wall:6.2f}s D/CH={ratio:.4f} "
              f"err_mean={100 * errors.mean():.3f}% p90={100 * np.quantile(errors, 0.9):.3f}% "
              f"max={100 * errors.max():.3f}% failures={int((errors > 0.1).sum())}")


if __name__ == "__main__":
    main()
