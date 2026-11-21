"""Distance-ratio statistics behind QALSH's collision-counting cost.

For each dataset and norm: sample source points a, compute d(a, b) to a sample of B, and divide by the exact
nearest-neighbor distance r* = d(a, B). Report how far the bulk of B lies from a relative to r*, and the expected
fraction f of (table, point) entries that fall inside a query's windows: f = mean over b of P(collide | d(a, b), R),
where P is the per-table collision probability of QALSH's projections (Cauchy for l1, Gaussian for l2) at radius R.

Usage: python3 contrast.py <datasets_dir> <exact_dir> <out.json>
<datasets_dir>/<name> must hold A.bin, B.bin and metadata.json (symlinks are fine); <exact_dir>/<name>_l<p>.bin are
the exact distances. Also reports He, Kumar and Chang's relative contrast E[mean distance] / E[NN distance].
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

DATASETS = ("p53", "trevi", "notredame", "dbpedia", "enron", "gist", "mnist", "federalist")
NUM_QUERIES = 100
MAX_BASE = 30000
C = 2.0
BINS = np.round(np.arange(1.0, 6.0001, 0.1), 1)  # ratio bins; ratios above 6 count as overflow

_erf = np.frompyfunc(math.erf, 1, 1)


def collide(p: int, w: float, u: np.ndarray) -> np.ndarray:
    """Per-table collision probability of a point at distance u * R from the query (window half-width w * R / 2)."""
    x = (w / 2.0) / np.maximum(np.asarray(u, dtype=np.float64), 1e-12)
    if p == 1:
        return 2.0 / math.pi * np.arctan(x)
    return np.asarray(_erf(x / math.sqrt(2.0)), dtype=np.float64)


def qalsh_params(p: int, n: int) -> tuple[float, int, int]:
    """Bucket width w, table count m and collision threshold l, as in DeriveQalshParameters (delta = 1/e)."""
    beta = 1.0
    t1, t2 = math.sqrt(math.log(2.0 / beta)), 1.0
    eta = t1 / t2
    w = 2.0 * math.sqrt(C) if p == 1 else math.sqrt(8.0 * C * C * math.log(C) / (C * C - 1.0))
    p1, p2 = float(collide(p, w, 1.0)), float(collide(p, w, C))
    m = math.ceil((t1 + t2) ** 2 / (2.0 * (p1 - p2) ** 2))
    l = math.ceil((eta * p1 + p2) / (1.0 + eta) * m)
    return w, m, l


def u_alpha(p: int, w: float, alpha: float) -> float:
    """Largest distance-to-radius ratio u at which the per-table collision probability is still alpha."""
    lo, hi = 1e-6, 1e6
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if float(collide(p, w, mid)) >= alpha:
            lo = mid
        else:
            hi = mid
    return lo


def main() -> None:
    data_dir, exact_dir, out_path = sys.argv[1:4]
    rng = np.random.default_rng(20261121)
    results = []
    for ds in DATASETS:
        meta = json.load(open(f"{data_dir}/{ds}/metadata.json"))
        d, na, nb = meta["num_dimensions"], meta["num_points"]["a"], meta["num_points"]["b"]
        a_all = np.memmap(f"{data_dir}/{ds}/A.bin", dtype=np.float32, mode="r", shape=(na, d))
        b_all = np.memmap(f"{data_dir}/{ds}/B.bin", dtype=np.float32, mode="r", shape=(nb, d))
        base_idx = np.sort(rng.choice(nb, MAX_BASE, replace=False)) if nb > MAX_BASE else np.arange(nb)
        b = np.asarray(b_all[base_idx], dtype=np.float32)
        for p in (1, 2):
            r_star = np.fromfile(f"{exact_dir}/{ds}_l{p}.bin", dtype=np.float64)
            assert r_star.size == na, (ds, p, r_star.size, na)
            valid = np.flatnonzero(r_star > 0)
            q_idx = np.sort(rng.choice(valid, min(NUM_QUERIES, valid.size), replace=False))
            a = np.asarray(a_all[q_idx], dtype=np.float32)
            if p == 2:
                a64, b64 = a.astype(np.float64), b.astype(np.float64)
                sq = (a64 * a64).sum(1)[:, None] + (b64 * b64).sum(1)[None, :] - 2.0 * (a64 @ b64.T)
                dist = np.sqrt(np.maximum(sq, 0.0))
                del a64, b64, sq
            else:
                dist = np.empty((q_idx.size, base_idx.size))
                for i in range(q_idx.size):
                    dist[i] = np.abs(b - a[i]).sum(1, dtype=np.float64)
            rq = r_star[q_idx]
            ratio = dist / rq[:, None]
            w, m, l = qalsh_params(p, nb)
            ua = u_alpha(p, w, l / m)
            # Radius = r*: the window is just wide enough that the nearest neighbor collides with probability p1.
            f_rstar = float(collide(p, w, ratio).mean())
            # Radius grid R0 * 2^t: the first power of two at which the NN's collision probability reaches l / m.
            r_final = 2.0 ** np.ceil(np.log2(rq / ua))
            f_grid = float(collide(p, w, dist / r_final[:, None]).mean())
            hist, _ = np.histogram(ratio, bins=BINS)
            row = {
                "dataset": ds, "norm": p, "d": d, "n_b": nb, "base_sample": int(base_idx.size),
                "queries": int(q_idx.size), "m": m, "l": l, "alpha": l / m, "u_alpha": ua,
                "p_at_ratio": {str(s): float(collide(p, w, s)) for s in (1, 1.5, 2, 3, 4, 6, 10)},
                "min_ratio": float(ratio.min()),
                "ratio_quantiles": {str(q): float(np.quantile(ratio, q)) for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9)},
                "median_of_query_medians": float(np.median(np.median(ratio, axis=1))),
                "relative_contrast": float(np.median(dist.mean(axis=1) / rq)),
                "relative_contrast_he": float(dist.mean() / rq.mean()),
                "frac_within_2": float((ratio <= 2.0).mean()),
                "f_at_rstar": f_rstar, "f_grid": f_grid,
                "rfinal_over_rstar_median": float(np.median(r_final / rq)),
                "hist_bins": BINS.tolist(), "hist_frac": (hist / ratio.size).tolist(),
                "overflow_frac": float((ratio > BINS[-1]).mean()),
            }
            results.append(row)
            qs = row["ratio_quantiles"]
            print(f"{ds:10s} l{p} d={d:5d} m={m:3d} l={l:3d} min={row['min_ratio']:.3f} "
                  f"q10={qs['0.1']:.2f} q50={qs['0.5']:.2f} q90={qs['0.9']:.2f} RC={row['relative_contrast']:.2f} "
                  f"RC_he={row['relative_contrast_he']:.2f} "
                  f"within2={row['frac_within_2']:.3f} f(R=r*)={f_rstar:.3f} f(grid)={f_grid:.3f}", flush=True)
    json.dump(results, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
