"""Run the C++ estimator with looked-up exact distances and collect its estimates.

The binary draws the importance samples and forms every estimate itself; --exact-distances makes it look up d(a, B)
for the sampled source points instead of running the exact queries, which gives the same estimates. Scripts use this
to repeat the estimator many times on fixed crude distances (saved QALSH4C runs, quadtree or HNSW weights) or with
uniform sampling.
"""

from __future__ import annotations

import re
import shlex
import subprocess

ESTIMATE = re.compile(r"Sample estimate: repeat=\d+ T=\d+ estimate=(\S+)")


def estimates(binary: str, dataset: str, norm: int, samples: int, repeats: int, exact: str,
              method: list[str], threads: int = 1) -> list[float]:
    """Estimates of CH(A -> B) from `repeats` independent samples of size `samples`.

    method is the weights subcommand, for example ["uniform"] or ["precomputed", "-f", "weights.bin"].
    """
    command = [binary, "-T", str(threads), "estimate", "-p", str(norm), "-d", dataset, "--direction", "ab",
               "sampling", "-n", str(samples), "--repeats", str(repeats), "--exact-distances", exact, *method]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output = completed.stdout + completed.stderr
    values = [float(v) for v in ESTIMATE.findall(output)]
    if completed.returncode != 0 or len(values) != repeats:
        raise SystemExit(f"estimator failed: {shlex.join(command)}\n{output[-2000:]}")
    return values
