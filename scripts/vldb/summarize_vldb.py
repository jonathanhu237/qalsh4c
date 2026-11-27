"""Summaries of the VLDB experiment phases (scripts/vldb), printed as text and written as JSON for the paper.

Usage: python3 summarize_vldb.py RESULTS_DIR [ablation|scaling|ann ...]
  ablation  ablation.txt: per dataset, norm and stopping rule, T and the mean of total, crude and sampling time and D/C
            over the runs (a run stopped by the cap counts as the cap and is reported as such)
  scaling   bscaling.txt and dscaling.txt: per point, exact time, mean QALSH4C total and stages, speedup, and the
            share f of the m|B| projection entries a query reads and the candidates evaluated per query
  ann       ann/anncmp.jsonl: per dataset and method setting, the cold time (build, crude queries and sampling stage)
            and the error of ANN-Sum and of importance sampling
Writes RESULTS_DIR/summary_<phase>.json.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from collections import defaultdict


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def ablation(root: str) -> list[dict]:
    cells: dict[tuple, list[list[str]]] = defaultdict(list)
    with open(os.path.join(root, "ablation.txt")) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ds, p, rule, rep, t, *rest = line.split()
            cells[(ds, int(p), rule, int(t))].append(rest)
    out = []
    for (ds, p, rule, t), runs in cells.items():
        capped = [r for r in runs if r[0].startswith(">")]
        done = [r for r in runs if not r[0].startswith(">")]
        totals = [float(r[0]) / 1000 for r in done] + [float(r[0][1:]) / 1000 for r in capped]
        row = {"dataset": ds, "norm": p, "rule": rule, "T": t, "runs": len(runs), "capped": len(capped),
               "total_s": mean(totals), "total_min_s": min(totals), "total_max_s": max(totals),
               "crude_s": mean([float(r[3]) / 1000 for r in done]),
               "sampling_s": mean([(float(r[0]) - float(r[3])) / 1000 for r in done]),
               "D_over_C": mean([float(r[4]) for r in done])}
        out.append(row)
        print(f"{ds:10s} l{p} {rule:9s} T={t:5d} runs={len(runs)} capped={len(capped)} total {row['total_s']:8.2f} s "
              f"({row['total_min_s']:.1f}-{row['total_max_s']:.1f}) crude {row['crude_s']:7.2f} "
              f"sampling {row['sampling_s']:6.2f} D/C {row['D_over_C']:.3f}")
    return out


def stat(text: str, name: str) -> float | None:
    m = re.search(name + r"=([0-9.e+]+)", text)
    return float(m.group(1)) if m else None


def scaling(root: str) -> list[dict]:
    out = []
    for name in ("bscaling.txt", "dscaling.txt"):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            continue
        exact: dict[tuple, float] = {}
        runs: dict[tuple, list[str]] = defaultdict(list)
        with open(path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if name == "bscaling.txt":
                    ds, p, na, nb, method, rep, *rest = parts
                else:
                    ds, p, na, nb, method, rep, *rest = parts
                key = (ds, int(p), int(na), int(nb))
                if method == "exact":
                    exact[key] = float(rest[0]) / 1000
                else:
                    runs[key].append(" ".join(rest))
        for key in sorted(runs, key=lambda k: (k[0], k[1], k[2], k[3])):
            ds, p, na, nb = key
            rs = runs[key]
            totals = [float(r.split()[0]) / 1000 for r in rs]
            total = mean(totals)
            median = statistics.median(totals)
            index = mean([float(r.split()[1]) / 1000 for r in rs])
            queries = mean([float(r.split()[2]) / 1000 for r in rs])
            weights = mean([float(r.split()[3]) / 1000 for r in rs])
            hits = [stat(r, "hits_per_query") for r in rs]
            evaluated = [stat(r, "evaluated_per_query") for r in rs]
            m = 36 if p == 1 else 17
            fractions = [stat(r, "fraction_of_entries") for r in rs]
            f_share = (mean([x for x in fractions if x is not None]) if any(x is not None for x in fractions)
                       else mean([h / (m * nb) for h in hits if h is not None]))
            row = {"series": name[:-4], "dataset": ds, "norm": p, "na": na, "nb": nb, "exact_s": exact.get(key),
                   "qalsh4c_s": total, "index_s": index, "crude_s": queries, "sampling_s": total - weights,
                   "speedup": exact[key] / total if key in exact else None, "f": f_share,
                   "qalsh4c_median_s": median, "qalsh4c_min_s": min(totals), "qalsh4c_max_s": max(totals),
                   "speedup_median": exact[key] / median if key in exact else None,
                   "evaluated_per_query": mean([e for e in evaluated if e is not None])}
            out.append(row)
            print(f"{row['series']:9s} {ds:16s} l{p} |A|={na:7d} |B|={nb:7d} exact {row['exact_s'] or float('nan'):8.2f}"
                  f" q4c {total:8.2f} (index {index:6.2f} crude {queries:7.2f} sampling {row['sampling_s']:6.2f})"
                  f" speedup {row['speedup'] or float('nan'):5.2f} median {median:7.2f} -> {row['speedup_median'] or float('nan'):5.2f}"
                  f" f {f_share:.3f} eval/q {row['evaluated_per_query']:.1f}")
    return out


def ann(root: str) -> list[dict]:
    out = []
    with open(os.path.join(root, "ann", "anncmp.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            crude = r.get("build_s", 0.0) + r.get("query_s", 0.0) + r.get("dist_s", 0.0)
            r["cold_s"] = crude + r["stage_s"]
            out.append(r)
            print(f"{r['dataset']:12s} {r['tag']:34s} T={r['T']:5d} cold {r['cold_s']:8.2f} s (build "
                  f"{r.get('build_s', 0):7.2f} query {r.get('query_s', 0):6.2f} stage {r['stage_s']:5.2f}) "
                  f"IS err mean {100 * r['mean_err']:.2f}% max {100 * r['max_err']:.2f}%"
                  + (f" sum err {100 * r['sum_err']:.2f}%" if "sum_err" in r else ""))
    return out


def main() -> None:
    root = sys.argv[1]
    for phase in sys.argv[2:] or ["ablation", "scaling", "ann"]:
        print(f"== {phase}")
        rows = globals()[phase](root)
        with open(os.path.join(root, f"summary_{phase}.json"), "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
