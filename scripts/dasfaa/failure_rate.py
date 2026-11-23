"""Full-randomness test of the QALSH4C guarantee (Theorems 1 and 2), driving the C++ binary.

The sample sizes T(eps) come from the binary's certified bound on kappa (`qalsh4c kappa`). Every run calls the binary
without --use-fixed-seed, so it draws fresh QALSH projections, recomputes the crude distances D_a of all source points
and draws an importance sample of size T(eps) for each tolerance. The exact distances d(a, B) are computed once by
the binary's linear scan (`ann --save-distances`), and the binary looks up the sampled ones (--exact-distances), which gives the same estimate as running the
exact queries. This script only runs the binary and summarizes its output.

Per run it records D / CH (the realized overestimation, whose mean Theorem 2 bounds by kappa), the conditional
relative variance of one importance sample and, for every eps, the relative error |eta - CH| / CH, whose tail
Theorem 1 bounds: Pr[error > eps] <= delta.

Usage:
  python3 failure_rate.py --binary BIN --dataset DIR --exact EXACT.bin --norm 1 --runs 50 --eps 0.3 0.1 \
      --initial-radius 1.401298464324817e-45 --out runs.jsonl [--weights-dir DIR] [--extra-args "..."]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time

import numpy as np

DELTA = 0.1


def run(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise SystemExit(f"command failed: {shlex.join(command)}\n{output}")
    return output


def certified_sample_sizes(binary: str, global_args: list[str], norm: int, c: str, eps: list[float]) -> tuple[str, list[int]]:
    """kappa_up and T(eps) from the binary's `kappa` command."""
    output = run([binary, *global_args, "kappa", "-p", str(norm), "-c", c, "-e", *map(str, eps),
                  "--delta", str(DELTA)])
    kappa = re.search(r"kappa_up=([0-9.]+)", output).group(1)
    sizes = [int(t) for t in re.findall(r"^T epsilon=\S+ delta=\S+ T=(\d+)$", output, flags=re.M)]
    if len(sizes) != len(eps):
        raise SystemExit(f"unexpected kappa output:\n{output}")
    return kappa, sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--dataset", required=True, help="Directory with A.bin, B.bin and metadata.json")
    parser.add_argument("--exact", required=True, help="float64 array of d(a, B) from ann --save-distances")
    parser.add_argument("--norm", type=int, choices=(1, 2), required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--eps", type=float, nargs="+", required=True)
    parser.add_argument("--initial-radius", default="1.401298464324817e-45",
                        help="R_0 of the round rules (2^-149); the continuous front ignores it")
    parser.add_argument("--approximation-ratio", default="2")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights-dir", default=None, help="Save every run's crude distances here as float32 .bin")
    parser.add_argument("--extra-args", default="", help="Global binary options placed before the command, for "
                        "example the stopping rule")
    args = parser.parse_args()

    global_args = ["--initial-radius", args.initial_radius, *shlex.split(args.extra_args)]
    kappa, sizes = certified_sample_sizes(args.binary, global_args, args.norm, args.approximation_ratio, args.eps)
    exact = np.fromfile(args.exact, dtype=np.float64)
    with open(os.path.join(args.dataset, "metadata.json")) as f:
        dims = json.load(f)["num_dimensions"]
    # The crude and the exact distances are float32 sums of dims terms taken in different orders, so each can be off
    # by about (dims + 1) * 2^-24 relative to the real distance. The check allows both errors.
    tolerance = 2 * (dims + 2) * 2.0**-24
    ch = float(exact.sum())
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
            for index in range(done, args.runs):
                cache_path = os.path.join(work, cache_name)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                record = {"run": index, "kappa_up": float(kappa), "errors": {}}
                for position, (eps, t) in enumerate(zip(args.eps, sizes)):
                    # The first call draws the projections and caches the crude distances; the later calls reuse them,
                    # so all tolerances share one projection draw.
                    command = [args.binary, "-T", str(args.threads), *global_args, "estimate", "-p", str(args.norm),
                               "-d", work, "--direction", "ab", "sampling", "-n", str(t), "--use-cache",
                               "--exact-distances", args.exact, "qalsh", "-c", args.approximation_ratio]
                    start = time.perf_counter()
                    log = run(command)
                    wall = time.perf_counter() - start
                    estimate = float(re.search(r"Sample estimate: repeat=1 T=\d+ estimate=(\S+)", log).group(1))
                    record["errors"][str(eps)] = {"T": t, "rel_error": abs(estimate - ch) / ch}
                    if position == 0:
                        record["crude_wall_s"] = round(wall, 3)
                        for key, pattern in (("index_ms", r"index_ms=([0-9.]+)"),
                                             ("queries_ms", r"queries_ms=([0-9.]+)"),
                                             ("hits_per_query", r"hits_per_query=([0-9.]+)"),
                                             ("evaluated_per_query", r"evaluated_per_query=([0-9.]+)")):
                            match = re.search(pattern, log)
                            if match:
                                record[key] = float(match.group(1))
                weights = np.fromfile(cache_path, dtype=np.float32).astype(np.float64)
                if weights.size != exact.size:
                    raise SystemExit(f"weights size {weights.size} != {exact.size}")
                if np.any(weights < exact * (1 - tolerance) - 1e-6):
                    raise SystemExit(f"run {index}: a crude distance is below the exact distance")
                total = float(weights.sum())
                support = weights > 0
                record["D_over_CH"] = total / ch
                # Conditional relative variance of one importance sample: Var(eta_1 | F) / CH^2.
                record["cond_rel_var"] = total / (ch * ch) * float(np.sum(exact[support] ** 2 / weights[support])) - 1
                if args.weights_dir:
                    os.makedirs(args.weights_dir, exist_ok=True)
                    shutil.copyfile(cache_path, os.path.join(args.weights_dir, f"run{index:04d}.bin"))
                out.write(json.dumps(record) + "\n")
                out.flush()
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
