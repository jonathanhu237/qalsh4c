"""LaTeX rows of the c table: per c and norm, the beta = 1 parameters, kappa_hat and T(0.1), and for each dataset the
median index + crude-query time over the runs, the mean D/CH and the maximum relative error at eps = 0.1.

Usage: python3 csweep_table.py FINAL_DIR ds1 ds2
c = 2 comes from FINAL_DIR/randomness, c = 1.5 and 3 from FINAL_DIR/csweep.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

PARAMS = {  # (c, norm): (m, l, kappa_hat, T at eps = delta = 0.1), from certify_kappa.certify(c, ., norm, True, 1.0)
    (1.5, 1): (103, 51, 1.549489, 550), (1.5, 2): (45, 31, 1.550916, 551),
    (2.0, 1): (36, 18, 2.136466, 1137), (2.0, 2): (17, 12, 2.120719, 1121),
    (3.0, 1): (16, 8, 3.300289, 2301), (3.0, 2): (8, 5, 3.051487, 2052),
}


def load(final: str, ds: str, p: int, c: float) -> list[dict]:
    path = (f"{final}/randomness/{ds}_l{p}.jsonl" if c == 2.0
            else f"{final}/csweep/{ds}_l{p}_c{c:g}.jsonl")
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def main() -> None:
    final, datasets = sys.argv[1], sys.argv[2:]
    for p in (1, 2):
        for c in (1.5, 2.0, 3.0):
            m, l, kappa, t = PARAMS[(c, p)]
            cells = [f"$\\\\ell_{p}$" if c == 1.5 else "", f"{c:g}", str(m), str(l), f"{kappa:.3f}", f"{t:,}".replace(",", "{,}")]
            for ds in datasets:
                rows = load(final, ds, p, c)
                if not rows:
                    cells += ["--", "--", "--"]
                    continue
                crude = np.median([(r.get("index_ms", 0) + r.get("queries_ms", 0)) / 1000 for r in rows])
                ratio = np.mean([r["D_over_CH"] for r in rows])
                err = max(r["errors"]["0.1"]["rel_error"] for r in rows)
                cells += [f"{crude:.2f}", f"{ratio:.2f}", f"{100 * err:.2f}"]
            print(" & ".join(cells) + " \\\\\\\\")


if __name__ == "__main__":
    main()
