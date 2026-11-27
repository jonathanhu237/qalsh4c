"""Error and time of QALSH4C as the error tolerance eps varies, from the crude distances saved by the randomness runs.

For each eps, T comes from the binary's certified kappa (delta = 0.1). The error is measured over every saved run
(one draw of projections each) with --per-run independent samples per run, looking up the exact distances of the
sampled source vectors, so the estimates cover both sources of randomness. The time of the sampling stage at T is one
real run of the binary with the first run's crude distances (it includes reading the vectors); the end-to-end time
adds the mean index and crude time of the cost runs.

Usage: python3 epsweep.py --name DS --dataset DIR --exact EX.bin --norm 1|2 --binary BIN --weights-dir DIR --out OUT.jsonl
       [--eps 0.3 0.2 0.1 0.05 0.02 0.01] [--per-run 4]
Appends one JSON line per eps to OUT.jsonl.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dasfaa"))
from binary_estimates import estimates  # noqa: E402

R0 = "1.401298464324817e-45"


def certified_t(binary: str, norm: int, eps: list[float]) -> dict[float, int]:
    out = subprocess.run([binary, "--initial-radius", R0, "kappa", "-p", str(norm), "-c", "2", "-e",
                          *[str(e) for e in eps]], capture_output=True, text=True, check=True)
    text = out.stdout + out.stderr
    found = {float(e): int(t) for e, t in re.findall(r"^T epsilon=(\S+) delta=\S+ T=(\d+)", text, re.M)}
    if len(found) != len(eps):
        raise SystemExit(f"kappa output without all T values:\n{text}")
    return found


def stage_seconds(binary: str, dataset: str, norm: int, t: int, weights: str, bound: list[str]) -> float:
    out = subprocess.run([binary, "-T", "24", "estimate", "-p", str(norm), "-d", dataset, "--direction", "ab",
                          "sampling", "-n", str(t), "precomputed", "-f", weights, *bound], capture_output=True,
                         text=True, check=True)
    return float(re.search(r"Time Consumed: ([0-9.]+) ms", out.stdout + out.stderr).group(1)) / 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exact", required=True)
    parser.add_argument("--norm", type=int, choices=(1, 2), required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    parser.add_argument("--per-run", type=int, default=4)
    parser.add_argument("--upper-bounds", action="store_true",
                        help="pass --upper-bounds to the precomputed QALSH weights, so the timed sampling stage prunes "
                             "as the live run does (binaries from 2026-11-25 on)")
    args = parser.parse_args()
    bound = ["--upper-bounds"] if args.upper_bounds else []
    runs = sorted(glob.glob(os.path.join(args.weights_dir, "run*.bin")))
    if not runs:
        raise SystemExit(f"no saved runs in {args.weights_dir}")
    ch = float(np.fromfile(args.exact, dtype=np.float64).sum())
    ts = certified_t(args.binary, args.norm, args.eps)
    with open(args.out, "a") as out:
        for eps in args.eps:
            t = ts[eps]
            est = np.concatenate([estimates(args.binary, args.dataset, args.norm, t, args.per_run, args.exact,
                                            ["precomputed", "-f", w], threads=24) for w in runs])
            err = np.abs(est - ch) / ch
            record = {"dataset": args.name, "norm": args.norm, "eps": eps, "T": t, "runs": len(runs),
                      "estimates": int(err.size), "failures": int((err > eps).sum()), "mean_err": float(err.mean()),
                      "p95_err": float(np.quantile(err, 0.95)), "max_err": float(err.max()),
                      "stage_s": stage_seconds(args.binary, args.dataset, args.norm, t, runs[0], bound)}
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
