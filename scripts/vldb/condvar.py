"""Relative variance of one importance sample given the crude distances, from the randomness runs' saved weights.

For crude distances D_a (one file per run) and exact distances d_a, one sample D d_x / D_x with x ~ D_x / D has
relative variance V1 = D sum_a d_a^2 / (D_a C^2) - 1 <= D / C - 1 (since d_a <= D_a), where D = sum D_a and
C = sum d_a. The certified T bounds V1 in expectation by kappa - 1. Uniform sampling has V1 = |A| sum d_a^2 / C^2 - 1.

Usage: python3 condvar.py RANDOMNESS_WEIGHTS_DIR EXACT_DIR   (subdirectories <dataset>_l<p> of run*.bin, and
       <dataset>_l<p>.bin of float64 exact distances)
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

KAPPA_MINUS_ONE = {1: 0.416132, 2: 0.468210}


def main() -> None:
    weights_root, exact_root = sys.argv[1], sys.argv[2]
    print("dataset norm runs D/C-1 V1_mean V1_max (kappa-1)/V1_mean uniform_V1")
    for d in sorted(glob.glob(os.path.join(weights_root, "*_l[12]"))):
        ds, p = os.path.basename(d).rsplit("_l", 1)
        p = int(p)
        exact_path = os.path.join(exact_root, f"{ds}_l{p}.bin")
        runs = sorted(glob.glob(os.path.join(d, "run*.bin")))
        if not os.path.exists(exact_path) or not runs:
            continue
        e = np.fromfile(exact_path, dtype=np.float64)
        c = e.sum()
        v1, excess = [], []
        for path in runs:
            w = np.fromfile(path, dtype=np.float32).astype(np.float64)
            total = w.sum()
            positive = w > 0
            v1.append((total * np.sum(e[positive] ** 2 / w[positive]) - c * c) / (c * c))
            excess.append(total / c - 1)
        v1 = np.array(v1)
        uniform = len(e) * np.sum(e ** 2) / (c * c) - 1
        print(f"{ds} {p} {len(runs)} {np.mean(excess):.4f} {v1.mean():.5f} {v1.max():.5f} "
              f"{KAPPA_MINUS_ONE[p] / v1.mean():.1f} {uniform:.4f}")


if __name__ == "__main__":
    main()
