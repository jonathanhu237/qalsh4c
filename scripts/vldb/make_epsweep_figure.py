"""Write the pgfplots source of the error-tolerance figure from epsweep/epsweep.jsonl and the cost summaries.

Four panels against eps (log): the largest observed relative error for l1 and for l2, with the line error = eps, and
the end-to-end time of QALSH4C relative to the best exact baseline for l1 and for l2: the mean end-to-end time of the
cost runs (eps = 0.1) plus the change in the sampling stage from eps = 0.1 to eps (two runs of the stage through the
same precomputed-weights path, so loading the vectors cancels).

Usage: python3 make_epsweep_figure.py epsweep.jsonl OUT.tex COST_SUMMARY.json [COST_SUMMARY.json ...] [--placeholder]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

ORDER = ["p53", "trevi", "notredame", "flickr2coco", "coco2flickr", "dbpedia1m", "enron"]
NAME = {"p53": "P53", "trevi": "Trevi", "notredame": "Notre Dame", "flickr2coco": r"Flickr\arrowto{}COCO",
        "coco2flickr": r"COCO\arrowto{}Flickr", "dbpedia1m": "DBpedia", "enron": "Enron"}
CNAME = {"p53": "ca", "trevi": "cb", "notredame": "cc", "flickr2coco": "cd", "coco2flickr": "ce",
         "dbpedia1m": "cf", "enron": "cg"}
COLOR = {"p53": "2A78D6", "trevi": "EB6834", "notredame": "1BAF7A", "flickr2coco": "9B59B6",
         "coco2flickr": "E5B012", "dbpedia1m": "D6457A", "enron": "5B6B7A"}
MARK = {"p53": "*", "trevi": "square*", "notredame": "triangle*", "flickr2coco": "diamond*", "coco2flickr": "pentagon*",
        "dbpedia1m": "o", "enron": "square"}


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--placeholder"]
    placeholder = "--placeholder" in sys.argv
    sweep_path, out_path, *cost_paths = args
    cost = {}
    for path in cost_paths:
        for r in json.load(open(path)):
            if r["threads"] == 24:
                cost[(r["dataset"], r["norm"])] = r
    err = defaultdict(list)
    rel = defaultdict(list)
    rows = [json.loads(line) for line in open(sweep_path)]
    base = {(r["dataset"], r["norm"]): r["stage_s"] for r in rows if abs(r["eps"] - 0.1) < 1e-9}
    for r in rows:
        key = (r["dataset"], r["norm"])
        err[key].append((r["eps"], 100 * r["max_err"]))
        if key in cost and key in base:
            c = cost[key]
            rel[key].append((r["eps"], (c["qalsh4c_s"] + r["stage_s"] - base[key]) / c["best_exact_s"]))
    datasets = [d for d in ORDER if (d, 1) in err or (d, 2) in err]
    out = [r"""\documentclass[border=2pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc,plotmarks}
% Legends stored by name expand their text, so the arrow must not expand.
\protected\def\arrowto{\ensuremath{\to}}
"""]
    for d in datasets:
        out.append(r"\definecolor{%s}{HTML}{%s}" % (CNAME[d], COLOR[d]) + "\n")
    out.append(r"""\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
    group style={group size=4 by 1, horizontal sep=1.5cm},
    width=4.5cm, height=3.9cm,
    xmode=log, ymode=log, xmin=0.008, xmax=0.38, xtick={0.01,0.03,0.1,0.3}, xticklabels={0.01,0.03,0.1,0.3},
    ymajorgrids, grid style={black!10},
    axis x line*=bottom, axis y line*=left,
    tick label style={font=\footnotesize}, label style={font=\small}, title style={font=\small, yshift=-3pt},
    every axis plot/.append style={line width=0.9pt, mark size=1.5pt},
    xlabel={$\epsilon$},
]
""")
    panels = [("err", 1, r"Largest error, $\ell_1$", r"Error (\%)"), ("err", 2, r"Largest error, $\ell_2$", ""),
              ("rel", 1, r"Time, $\ell_1$", "Time / linear scan"), ("rel", 2, r"Time, $\ell_2$", "")]
    for k, (kind, p, title, ylabel) in enumerate(panels):
        opts = [f"title={{{title}}}"]
        if ylabel:
            opts.append(f"ylabel={{{ylabel}}}")
        if k == 0:
            opts.append(r"legend to name=epslegend, legend columns=4, legend cell align=left, legend style={draw=none, font=\footnotesize,"
                        r" /tikz/every even column/.append style={column sep=6pt}}")
        out.append(r"\nextgroupplot[" + ", ".join(opts) + "]\n")
        series = err if kind == "err" else rel
        for d in datasets:
            pts = sorted(series.get((d, p), []))
            out.append(r"\addplot[forget plot, %s, mark=%s] coordinates {%s};" % (
                CNAME[d], MARK[d], " ".join(f"({x},{y:.4f})" for x, y in pts)) + "\n")
        if kind == "err":
            out.append(r"\addplot[forget plot, black!60, dashed, domain=0.008:0.38, samples=2] {100*x};" + "\n")
        else:
            out.append(r"\addplot[forget plot, black!60, dashed, domain=0.008:0.38, samples=2] {1};" + "\n")
        if k == 0:
            for d in datasets:
                out.append(r"\addlegendimage{%s, mark=%s}\addlegendentry{%s}" % (CNAME[d], MARK[d], NAME[d]) + "\n")
            out.append(r"\addlegendimage{black!60, dashed}\addlegendentry{$\epsilon$ or linear scan}" + "\n")
    out.append("\\end{groupplot}\n")
    out.append(r"\node[anchor=south] at ($(group c1r1.north west)!0.5!(group c4r1.north east)+(0,12pt)$)"
               r" {\pgfplotslegendfromname{epslegend}};" + "\n")
    if placeholder:
        out.append(r"\node[red, opacity=0.45, font=\Huge\bfseries] at (current bounding box.center) {PLACEHOLDER};"
                   + "\n")
    out.append("\\end{tikzpicture}\n\\end{document}\n")
    with open(out_path, "w") as f:
        f.write("".join(out))


if __name__ == "__main__":
    main()
