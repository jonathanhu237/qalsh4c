"""Write the pgfplots source of the equal-time figure from time_accuracy_timed.json files (runs/final/recalib.py).

For every dataset and norm: the mean relative error of QALSH4C, and of uniform and quadtree sampling with as many
samples as fit in QALSH4C's mean end-to-end time, on a log axis so that ratios read the same on every dataset; a
cross marks a quadtree whose crude distances alone take longer than QALSH4C.

Usage: python3 make_equal_time_figure.py OUT.tex JSON [JSON ...] [--datasets p53 trevi ...] [--placeholder]
"""

from __future__ import annotations

import json
import sys

NAME = {"p53": "P53", "trevi": "Trevi", "notredame": "Notre Dame", "flickr2coco": r"Flickr$\to$COCO",
        "coco2flickr": r"COCO$\to$Flickr", "dbpedia1m": "DBpedia", "dbpedia": "DBpedia", "enron": "Enron",
        "gist": "GIST"}
YMIN, YMAX = 0.05, 10.0


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--placeholder"]
    placeholder = "--placeholder" in sys.argv
    datasets = ["p53", "trevi", "notredame", "flickr2coco", "coco2flickr", "dbpedia1m", "enron"]
    if "--datasets" in args:
        i = args.index("--datasets")
        datasets = args[i + 1:]
        args = args[:i]
    out_path, *paths = args
    rows = {}
    for path in paths:
        rows.update(json.load(open(path)))
    datasets = [d for d in datasets if f"{d}_l1" in rows]
    bars = {"qalsh4c": [], "uniform": [], "quadtree": []}
    crosses, labels, ticks = [], [], []
    for k, ds in enumerate(datasets):
        for p in (1, 2):
            x = 3 * k + (p - 1)
            ticks.append(x)
            r = rows[f"{ds}_l{p}"]
            for method, shift in (("qalsh4c", -0.27), ("uniform", 0.0), ("quadtree", 0.27)):
                if method not in r:
                    if method == "quadtree":
                        crosses.append(x + shift)
                    continue
                bars[method].append((x, 100 * r[method]["mean"]))
        labels.append((3 * k + 0.5, NAME.get(ds, ds)))
    n = len(datasets)
    out = [r"""\documentclass[border=4pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{cq}{HTML}{2A78D6}
\definecolor{cu}{HTML}{EB6834}
\definecolor{ct}{HTML}{1BAF7A}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=%.1fcm, height=3.5cm,
    ybar=0pt, bar width=0.27,
    ymode=log, log origin=infty, ymin=%g, ymax=%g,
    xmin=-0.8, xmax=%g,
    xtick={%s},
    xticklabels={%s},
    xticklabel style={font=\footnotesize},
    major x tick style={draw=none},
    ytick={0.1,1,10}, yticklabels={0.1,1,10},
    ylabel={Mean error (\%%)},
    ymajorgrids, grid style={black!12},
    axis x line*=bottom, axis y line*=left,
    legend style={draw=none, fill=none, font=\small, at={(0.5,1.06)}, anchor=south, legend columns=3, /tikz/every even column/.append style={column sep=6pt}},
    legend image code/.code={\draw[#1, draw=none] (0cm,-0.1cm) rectangle (0.3cm,0.12cm);},
    clip=false,
]
""" % (2.4 * n + 0.8, YMIN, YMAX, 3 * n - 1.2, ",".join(str(t) for t in ticks),
       ",".join([r"$\ell_1$", r"$\ell_2$"] * n))]
    for method, color, label in (("qalsh4c", "cq", "QALSH4C"), ("uniform", "cu", "Uniform"),
                                 ("quadtree", "ct", "Quadtree")):
        coords = " ".join(f"({x},{v:.3f})" for x, v in bars[method])
        out.append(r"\addplot[fill=%s, draw=white, line width=0.3pt] coordinates {%s};" % (color, coords) + "\n")
        out.append(r"\addlegendentry{%s}" % label + "\n")
    for x in crosses:
        out.append(r"\node[ct, font=\scriptsize, inner sep=0pt] at (axis cs:%.2f,%g) {$\times$};" % (x, YMIN * 1.6) + "\n")
    for x, name in labels:
        out.append(r"\node[font=\small, anchor=north] at (axis cs:%.1f,%g) [yshift=-16pt] {%s};" % (x, YMIN, name) + "\n")
    out.append("\\end{axis}\n")
    if placeholder:
        out.append(r"\node[red, opacity=0.45, font=\Huge\bfseries] at (current bounding box.center) {PLACEHOLDER};"
                   + "\n")
    out.append("\\end{tikzpicture}\n\\end{document}\n")
    with open(out_path, "w") as f:
        f.write("".join(out))


if __name__ == "__main__":
    main()
