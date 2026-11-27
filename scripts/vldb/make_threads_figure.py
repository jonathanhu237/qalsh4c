"""Write the pgfplots source of the thread-scaling figure from threads.txt (threads.sh).

One panel per dataset, both axes logarithmic: the time of the exact scan (one run) and of QALSH4C (median of three
cold runs, with a bar from the fastest to the slowest run) for both norms at 1 to 24 threads.

Usage: python3 make_threads_figure.py threads.txt OUT.tex [--datasets trevi flickr2coco] [--placeholder]
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict

NAME = {"trevi": "Trevi", "flickr2coco": r"Flickr$\to$COCO", "notredame": "Notre Dame", "p53": "P53"}
STYLE = {("exact", 1): "ce, mark=square*", ("exact", 2): "ce, densely dashed, mark=square, mark options={solid}",
         ("qalsh4c", 1): "cq, mark=*", ("qalsh4c", 2): "cq, densely dashed, mark=o, mark options={solid}"}
BARS = r"error bars/.cd, y dir=both, y explicit, error bar style={solid, line width=0.6pt}"
LABEL = {("exact", 1): r"Linear scan, $\ell_1$", ("exact", 2): r"Linear scan, $\ell_2$",
         ("qalsh4c", 1): r"QALSH4C, $\ell_1$", ("qalsh4c", 2): r"QALSH4C, $\ell_2$"}


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--placeholder"]
    placeholder = "--placeholder" in sys.argv
    datasets = ["trevi", "flickr2coco"]
    if "--datasets" in args:
        i = args.index("--datasets")
        datasets = args[i + 1:]
        args = args[:i]
    path, out_path = args
    times = defaultdict(list)
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        ds, p, t, method, _rep, total, *_ = line.split()
        times[(ds, int(p), method, int(t))].append(float(total) / 1000)
    out = [r"""\documentclass[border=2pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc}
\definecolor{cq}{HTML}{2A78D6}
\definecolor{ce}{HTML}{EB6834}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
    group style={group size=%d by 1, horizontal sep=1.0cm},
    width=4.6cm, height=4.0cm,
    xmode=log, ymode=log, log basis x=2, xtick={1,2,4,8,12,24}, xticklabels={1,2,4,8,12,24},
    ymajorgrids, grid style={black!10},
    axis x line*=bottom, axis y line*=left,
    tick label style={font=\footnotesize}, label style={font=\small}, title style={font=\small, yshift=-3pt},
    every axis plot/.append style={line width=1pt, mark size=1.6pt},
    xlabel={Threads},
]
""" % len(datasets)]
    for k, ds in enumerate(datasets):
        opts = [f"title={{{NAME.get(ds, ds)}}}"]
        if k == 0:
            opts.append("ylabel={Time (s)}")
            opts.append(r"legend to name=thrlegend, legend columns=2, legend style={draw=none, font=\footnotesize,"
                        r" /tikz/every even column/.append style={column sep=6pt}}")
        out.append(r"\nextgroupplot[" + ", ".join(opts) + "]\n")
        for series in [("exact", 1), ("exact", 2), ("qalsh4c", 1), ("qalsh4c", 2)]:
            method, p = series
            pts = sorted((t, statistics.median(v), min(v), max(v)) for (d, q, m, t), v in times.items()
                         if d == ds and q == p and m == method)
            style = STYLE[series] + (", " + BARS if method == "qalsh4c" else "")
            out.append(r"\addplot[forget plot, %s] coordinates {%s};" % (style, " ".join(
                f"({t},{y:.3f}) += (0,{hi - y:.3f}) -= (0,{y - lo:.3f})" if method == "qalsh4c" else f"({t},{y:.3f})"
                for t, y, lo, hi in pts)) + "\n")
        if k == 0:
            for series in [("exact", 1), ("exact", 2), ("qalsh4c", 1), ("qalsh4c", 2)]:
                out.append(r"\addlegendimage{%s}\addlegendentry{%s}" % (STYLE[series], LABEL[series]) + "\n")
    out.append("\\end{groupplot}\n")
    out.append(r"\node[anchor=south] at ($(group c1r1.north west)!0.5!(group c%dr1.north east)+(0,12pt)$)"
               r" {\pgfplotslegendfromname{thrlegend}};" % len(datasets) + "\n")
    if placeholder:
        out.append(r"\node[red, opacity=0.45, font=\Large\bfseries] at (current bounding box.center) {PLACEHOLDER};"
                   + "\n")
    out.append("\\end{tikzpicture}\n\\end{document}\n")
    with open(out_path, "w") as f:
        f.write("".join(out))


if __name__ == "__main__":
    main()
