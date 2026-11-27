"""Write the pgfplots source of the scalability figure from summary_scaling.json (summarize_vldb.py scaling).

Three panels, both axes logarithmic, end-to-end time of the exact scan (once, 12 threads) and of QALSH4C (median of
its cold runs, with a bar from the fastest to the slowest run) for both norms: (a) Trevi with |A| = 20,000 and growing |B| (bscaling series), (b) DBpedia with
|A| = 50,000 and growing |B| (dscaling series on the full set), (c) DBpedia with |B| = 500,000 and growing |A|.

Usage: python3 make_scaling_figure.py summary_scaling.json OUT.tex [--placeholder]
  --placeholder  stamp the figure, for layout before the final numbers exist
"""

from __future__ import annotations

import json
import sys

HEAD = r"""\documentclass[border=2pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc}
\definecolor{cq}{HTML}{2A78D6}
\definecolor{ce}{HTML}{EB6834}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
    group style={group size=3 by 1, horizontal sep=1.1cm},
    width=6.2cm, height=4.0cm,
    xmode=log, ymode=log, log basis x=10, xtick={1000,10000,100000,1000000,10000000},
    ymajorgrids, grid style={black!10},
    axis x line*=bottom, axis y line*=left,
    tick label style={font=\footnotesize}, label style={font=\small}, title style={font=\small, yshift=-3pt},
    every axis plot/.append style={line width=1pt, mark size=1.7pt},
]
"""

BARS = r"error bars/.cd, y dir=both, y explicit, error bar style={solid, line width=0.6pt}"
STYLE = {("exact", 1): "ce, mark=square*", ("exact", 2): "ce, densely dashed, mark=square, mark options={solid}",
         ("qalsh4c", 1): "cq, mark=*, " + BARS,
         ("qalsh4c", 2): "cq, densely dashed, mark=o, mark options={solid}, " + BARS}
LABEL = {("exact", 1): r"Linear scan, $\ell_1$", ("exact", 2): r"Linear scan, $\ell_2$",
         ("qalsh4c", 1): r"QALSH4C, $\ell_1$", ("qalsh4c", 2): r"QALSH4C, $\ell_2$"}


def coords(points: list[tuple]) -> str:
    """(x, y) points, or (x, median, min, max) points drawn with a bar from the smallest to the largest run."""
    out = []
    for pt in sorted(points):
        if len(pt) == 2:
            out.append(f"({pt[0]},{pt[1]:.3f})")
        else:
            x, y, lo, hi = pt
            out.append(f"({x},{y:.3f}) += (0,{hi - y:.3f}) -= (0,{y - lo:.3f})")
    return " ".join(out)


def main() -> None:
    rows = json.load(open(sys.argv[1]))
    panels = {"trevi": {}, "dbB": {}, "dbA": {}}
    for r in rows:
        if r["exact_s"] is None:
            continue
        if r["series"] == "bscaling" and r["dataset"] == "trevi":
            key, x = "trevi", r["nb"]
        elif r["series"] == "dscaling" and r["na"] == 50000:
            key, x = "dbB", r["nb"]
        else:
            key, x = None, None
        targets = [(key, x)] if key else []
        if r["series"] == "dscaling" and r["nb"] == 500000:
            targets.append(("dbA", r["na"]))
        for key, x in targets:
            panels[key].setdefault(("exact", r["norm"]), []).append((x, r["exact_s"]))
            panels[key].setdefault(("qalsh4c", r["norm"]), []).append(
                (x, r["qalsh4c_median_s"], r["qalsh4c_min_s"], r["qalsh4c_max_s"]))
    titles = {"trevi": r"Trevi, $|A|=20{,}000$", "dbB": r"DBpedia, $|A|=50{,}000$",
              "dbA": r"DBpedia, $|B|=500{,}000$"}
    xlabels = {"trevi": "$|B|$", "dbB": "$|B|$", "dbA": "$|A|$"}
    out = [HEAD]
    for k, key in enumerate(["trevi", "dbB", "dbA"]):
        opts = [f"title={{{titles[key]}}}", f"xlabel={{{xlabels[key]}}}"]
        if k == 0:
            opts.append("ylabel={Time (s)}")
            opts.append(r"legend to name=scalelegend, legend columns=4, legend style={draw=none, font=\footnotesize,"
                        r" /tikz/every even column/.append style={column sep=8pt}}")
        out.append(r"\nextgroupplot[" + ", ".join(opts) + "]\n")
        for series in [("exact", 1), ("exact", 2), ("qalsh4c", 1), ("qalsh4c", 2)]:
            pts = panels[key].get(series, [])
            out.append(r"\addplot[%s] coordinates {%s};" % (STYLE[series], coords(pts)) + "\n")
            if k == 0:
                out.append(r"\addlegendentry{%s}" % LABEL[series] + "\n")
    out.append("\\end{groupplot}\n")
    out.append(r"\node[anchor=south] at ($(group c1r1.north west)!0.5!(group c3r1.north east)+(0,12pt)$)"
               r" {\pgfplotslegendfromname{scalelegend}};" + "\n")
    if "--placeholder" in sys.argv:
        out.append(r"\node[red, opacity=0.45, font=\Huge\bfseries] at (current bounding box.center) {PLACEHOLDER};"
                   + "\n")
    out.append("\\end{tikzpicture}\n\\end{document}\n")
    with open(sys.argv[2], "w") as f:
        f.write("".join(out))


if __name__ == "__main__":
    main()
