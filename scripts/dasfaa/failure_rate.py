"""Full-randomness test of the QALSH4C guarantee (Theorems 1 and 2).

Every run draws fresh QALSH projections (the binary is called without --use-fixed-seed), recomputes the crude
distances D_a for all source points, and then draws an importance sample of the size T(eps) from Sect. 4.3 for
each tolerance eps. The exact distances d(a, B) are computed once by exact_nn and looked up for the sampled points,
which gives the same estimate as computing them during sampling.

Per run it records D / CH (the realized overestimation, whose mean Theorem 2 bounds by kappa) and, for every eps,
the relative error |eta - CH| / CH (whose tail Theorem 1 bounds: Pr[error > eps] <= delta).

Usage:
  python3 failure_rate.py --binary BIN --dataset DIR --exact EXACT.bin --norm 1 --runs 200 --kappa 4.283342 \
      --eps 1 0.5 0.3 0.1 --initial-radius 2.8025969286496341e-45 --out runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time

import numpy as np

DELTA = 0.1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--dataset", required=True, help="Directory with A.bin, B.bin and metadata.json")
    parser.add_argument("--exact", required=True, help="float64 array of d(a, B) from exact_nn")
    parser.add_argument("--norm", type=int, choices=(1, 2), required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--kappa", type=float, required=True, help="Certified upper bound on kappa (Table 1)")
    parser.add_argument("--eps", type=float, nargs="+", required=True)
    parser.add_argument("--initial-radius", default="2.8025969286496341e-45")
    parser.add_argument("--approximation-ratio", default="2")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261120, help="Seed of the sampling RNG (projections are random)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights-dir", default=None, help="Save every run's crude distances here as float32 .bin")
    parser.add_argument("--extra-args", default="", help="Global binary options placed before 'estimate', for example "
                        "'--qalsh-tables 8 --qalsh-threshold 5'")
    args = parser.parse_args()

    exact = np.fromfile(args.exact, dtype=np.float64)
    ch = float(exact.sum())
    sample_sizes = {eps: max(1, math.ceil((args.kappa - 1.0) / (DELTA * eps * eps) - 1e-9)) for eps in args.eps}
    rng = np.random.default_rng(args.seed)
    cache_name = f"l{args.norm}_qalsh_weights_A.bin"

    work = tempfile.mkdtemp(prefix="qalsh4c-failure-")
    for name in ("A.bin", "B.bin", "metadata.json"):
        os.symlink(os.path.abspath(os.path.join(args.dataset, name)), os.path.join(work, name))
    done = 0
    if os.path.exists(args.out):
        with open(args.out) as existing:
            done = sum(1 for _ in existing)
    try:
        with open(args.out, "a") as out:
            for run in range(done, args.runs):
                cache_path = os.path.join(work, cache_name)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                command = [args.binary, "-T", str(args.threads), "--initial-radius", args.initial_radius,
                           *shlex.split(args.extra_args), "estimate", "-p", str(args.norm), "-d", work, "--direction", "ab",
                           "sampling", "-n", "1", "--use-cache", "qalsh", "-c", args.approximation_ratio]
                start = time.perf_counter()
                completed = subprocess.run(command, capture_output=True, text=True, check=False)
                wall = time.perf_counter() - start
                if completed.returncode != 0 or not os.path.exists(cache_path):
                    raise SystemExit(f"run {run} failed:\n{completed.stdout}\n{completed.stderr}")
                weights = np.fromfile(cache_path, dtype=np.float32).astype(np.float64)
                if weights.size != exact.size:
                    raise SystemExit(f"weights size {weights.size} != {exact.size}")
                if np.any(weights < exact * (1 - 1e-5) - 1e-6):
                    raise SystemExit(f"run {run}: a crude distance is below the exact distance")
                total = float(weights.sum())
                prob = weights / total
                support = np.flatnonzero(prob > 0)
                # Conditional relative variance of one importance sample: Var(eta_1 | F) / CH^2.
                rel_var = total / (ch * ch) * float(np.sum(exact[support] ** 2 / weights[support])) - 1.0
                record = {"run": run, "D_over_CH": total / ch, "cond_rel_var": rel_var, "crude_wall_s": round(wall, 3),
                          "errors": {}}
                log = completed.stdout + completed.stderr
                for key, pattern in (("index_ms", r"index_ms=([0-9.]+)"), ("queries_ms", r"queries_ms=([0-9.]+)"),
                                     ("hits_per_query", r"hits_per_query=([0-9.]+)"),
                                     ("evaluated_per_query", r"evaluated_per_query=([0-9.]+)")):
                    match = re.search(pattern, log)
                    if match:
                        record[key] = float(match.group(1))
                if args.weights_dir:
                    os.makedirs(args.weights_dir, exist_ok=True)
                    shutil.copyfile(cache_path, os.path.join(args.weights_dir, f"run{run:04d}.bin"))
                for eps, t in sample_sizes.items():
                    picks = rng.choice(support, size=t, replace=True, p=prob[support])
                    eta = float(np.mean(total / weights[picks] * exact[picks]))
                    record["errors"][str(eps)] = {"T": t, "rel_error": abs(eta - ch) / ch}
                out.write(json.dumps(record) + "\n")
                out.flush()
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
