"""Time and error of the CrudeNN choices on one dataset, l2: ANN indexes (summed or used for importance sampling),
uniform sampling and QALSH4C. Every estimator samples and evaluates through the same binary, so the sampling stage
(exact queries of the sampled source vectors) is measured with one implementation.

For each crude-distance file w (one per ANN setting):
  sum error   |sum(w) / C - 1| (ANN-Sum),
  IS error    mean, 95th percentile and max of |eta - C| / C over --repeats samples of size T (ANN-IS), using the
              exact distances of the sampled vectors (same estimates as computing them),
  IS time     one real run of the sampling stage at each T (load + sampling + exact queries of the sampled vectors).
Uniform sampling, and quadtree weights when given (no upper-bound pruning: they are not upper bounds), get the same
at each T of --uniform-T. QALSH4C gets the IS error from the crude distances saved by the randomness runs (one draw
of projections per file) at each certified T, and one real sampling-stage run per T. One record "load" holds the
time to read A as the ANN scripts read it, which the end-to-end time of ANN-IS adds (its sampling stage reads B).

Usage: python anncmp.py --name DS --dataset DIR --exact EX.bin --binary BIN --venv PY --out OUT
       --ann "hnsw --m 16 --ef-construction 100 --query 10 40 100" [--ann ...] --T 118 469 1873
       [--qalsh-weights DIR] [--uniform-T 64 128 ...]
Appends JSON lines to OUT/anncmp.jsonl.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list[str]) -> str:
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"failed: {shlex.join(cmd)}\n{done.stdout}\n{done.stderr}")
    return done.stdout + done.stderr


def total_ms(log: str) -> float:
    return float(re.search(r"Time Consumed: ([0-9.]+) ms", log).group(1))


def errors(binary: str, dataset: str, exact: str, source: list[str], t: int, repeats: int, ch: float) -> np.ndarray:
    log = run([binary, "-T", "24", "estimate", "-p", "2", "-d", dataset, "--direction", "ab", "sampling", "-n", str(t),
               "--repeats", str(repeats), "--exact-distances", exact, *source])
    est = np.array([float(x) for x in re.findall(r"Sample estimate: repeat=\d+ T=\d+ estimate=(\S+)", log)])
    if est.size != repeats:
        raise SystemExit(f"expected {repeats} estimates, got {est.size}")
    return np.abs(est - ch) / ch


def stage_seconds(binary: str, dataset: str, source: list[str], t: int) -> float:
    log = run([binary, "-T", "24", "estimate", "-p", "2", "-d", dataset, "--direction", "ab", "sampling", "-n", str(t),
               *source])
    return total_ms(log) / 1000


def summary(err: np.ndarray) -> dict:
    return {"mean_err": float(err.mean()), "p95_err": float(np.quantile(err, 0.95)), "max_err": float(err.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exact", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--venv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ann", action="append", default=[])
    parser.add_argument("--T", type=int, nargs="+", default=[118, 469, 1873])
    parser.add_argument("--uniform-T", type=int, nargs="+", default=[64, 128, 256, 469, 1024, 2048, 4096])
    parser.add_argument("--qalsh-weights", default=None)
    parser.add_argument("--quadtree-weights", default=None, help="float32 quadtree weights of the cost runs")
    parser.add_argument("--qalsh-runs", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--reuse-weights", default=None,
                        help="directory of an earlier run's ANN weights and timing.jsonl: skip the index builds and "
                             "evaluate those weights (their build and query times do not depend on the binary)")
    parser.add_argument("--upper-bounds", action="store_true",
                        help="pass --upper-bounds to precomputed ANN and QALSH weights, so the sampling stage prunes "
                             "with them as the live QALSH4C run does (binaries from 2026-11-25 on)")
    args = parser.parse_args()
    bound = ["--upper-bounds"] if args.upper_bounds else []
    os.makedirs(args.out, exist_ok=True)
    exact = np.fromfile(args.exact, dtype=np.float64)
    ch = float(exact.sum())
    out = open(os.path.join(args.out, "anncmp.jsonl"), "a")

    def emit(record: dict) -> None:
        record = {"dataset": args.name, **record}
        out.write(json.dumps(record) + "\n")
        out.flush()
        print(json.dumps(record), flush=True)

    with open(os.path.join(args.dataset, "metadata.json")) as f:
        meta = json.load(f)
    start = time.perf_counter()
    np.fromfile(os.path.join(args.dataset, "A.bin"), dtype=np.float32).reshape(meta["num_points"]["a"], -1)
    emit({"method": "load", "tag": "load", "load_a_s": time.perf_counter() - start})

    wdir = os.path.join(args.out, "weights", args.name)
    specs = args.ann
    if args.reuse_weights:
        wdir = args.reuse_weights
        specs = [None]
    for spec in specs:
        before = set(glob.glob(os.path.join(wdir, "*.bin"))) if spec is not None else set()
        if spec is not None:
            run([args.venv, os.path.join(HERE, "ann_weights.py"), "--dataset", args.dataset, "--out-dir", wdir,
                 *shlex.split(spec)])
        timing = {}
        with open(os.path.join(wdir, "timing.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                timing[r["tag"]] = r
        for path in sorted(set(glob.glob(os.path.join(wdir, "*.bin"))) - before):
            tag = os.path.basename(path)[:-4]
            w = np.fromfile(path, dtype=np.float32).astype(np.float64)
            base = {"method": timing[tag]["method"], "tag": tag, "load_s": timing[tag]["load_s"],
                    "build_s": timing[tag]["build_s"], "query_s": timing[tag]["query_s"],
                    "dist_s": timing[tag]["dist_s"],
                    "sum_err": abs(w.sum() / ch - 1), "D_over_C": w.sum() / ch}
            for t in args.T:
                err = errors(args.binary, args.dataset, args.exact, ["precomputed", "-f", path, *bound], t,
                             args.repeats, ch)
                emit({**base, "T": t, **summary(err),
                      "stage_s": stage_seconds(args.binary, args.dataset, ["precomputed", "-f", path, *bound], t)})

    for t in sorted(set(args.uniform_T)):
        err = errors(args.binary, args.dataset, args.exact, ["uniform"], t, args.repeats, ch)
        emit({"method": "uniform", "tag": "uniform", "T": t, **summary(err),
              "stage_s": stage_seconds(args.binary, args.dataset, ["uniform"], t)})
        if args.quadtree_weights:
            source = ["precomputed", "-f", args.quadtree_weights]
            err = errors(args.binary, args.dataset, args.exact, source, t, args.repeats, ch)
            emit({"method": "quadtree", "tag": "quadtree", "T": t, **summary(err),
                  "stage_s": stage_seconds(args.binary, args.dataset, source, t)})

    if args.qalsh_weights:
        files = sorted(glob.glob(os.path.join(args.qalsh_weights, "run*.bin")))[:args.qalsh_runs]
        per_run = max(1, args.repeats // len(files))
        for t in args.T:
            err = np.concatenate([errors(args.binary, args.dataset, args.exact, ["precomputed", "-f", f, *bound], t,
                                         per_run, ch) for f in files])
            emit({"method": "qalsh4c", "tag": "qalsh4c", "T": t, **summary(err), "runs": len(files),
                  "stage_s": stage_seconds(args.binary, args.dataset, ["precomputed", "-f", files[0], *bound], t)})
    out.close()


if __name__ == "__main__":
    main()
