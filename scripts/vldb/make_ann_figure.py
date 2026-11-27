"""Write the pgfplots source of the ANN and sampling comparison figure (l2) from ann/anncmp.jsonl and the cost
summaries.

One panel per dataset, four per row, end-to-end time (log) against mean relative error in percent (log):
  ANN-Sum   reading A and B + build + query + exact distances to the returned ids,
  ANN-IS    reading A + build + query + exact distances + the sampling stage at the certified T of QALSH4C at
            eps = 0.1 (the stage reads B),
  Uniform   the sampling stage at each T,
  Quadtree  the mean end-to-end time of the quadtree cost run (at that T), plus the change in its sampling stage,
  QALSH4C   the mean end-to-end time of the cost runs, plus the change in the sampling stage from that T to the other
            certified T (two runs of the stage through the same path, so reading the vectors cancels),
  exact     the best exact baseline of the cost runs, as a vertical line.

Usage: python3 make_ann_figure.py anncmp.jsonl OUT.tex COST_SUMMARY.json [COST_SUMMARY.json ...] [--placeholder]
       [--datasets p53 trevi notredame enron flickr2coco coco2flickr dbpedia1m]
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

NAME = {"notredame": "Notre Dame", "flickr2coco": r"Flickr$\to$COCO", "coco2flickr": r"COCO$\to$Flickr",
        "dbpedia1m": "DBpedia", "dbpedia": "DBpedia (200k)", "trevi": "Trevi", "p53": "P53", "enron": "Enron"}
FAMILY = {"hnsw": ("ch", "*", "o", "HNSW"), "ivfflat": ("cf", "square*", "square", "IVF-Flat"),
          "ivfpq": ("cp", "triangle*", "triangle", "IVF-PQ")}


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--placeholder"]
    placeholder = "--placeholder" in sys.argv
    datasets = ["p53", "trevi", "notredame", "enron", "flickr2coco", "coco2flickr", "dbpedia1m"]
    if "--datasets" in args:
        i = args.index("--datasets")
        datasets = args[i + 1:]
        args = args[:i]
    ann_path, out_path, *cost_paths = args
    cost = {}
    for path in cost_paths:
        for r in json.load(open(path)):
            if r["norm"] == 2 and r["threads"] == 24:
                cost[r["dataset"]] = r
    rows = defaultdict(list)
    for line in open(ann_path):
        r = json.loads(line)
        rows[r["dataset"]].append(r)
    datasets = [d for d in datasets if d in rows and d in cost]
    xs, ys = [], []
    panels = []
    for ds in datasets:
        c = cost[ds]
        pts = {"sum": defaultdict(set), "is": defaultdict(set)}
        uniform, quad, qalsh = [], [], []
        ts = sorted(r["T"] for r in rows[ds] if r["method"] == "qalsh4c")
        is_t = ts[len(ts) // 2]
        load_a = next((r["load_a_s"] for r in rows[ds] if r["method"] == "load"), 0.0)
        # The stage runs read precomputed weights, so each one reads B again and skips the D_x bound. Only their
        # difference from the run at the cost runs' T is added to the measured end-to-end time.
        base = {m: next((r["stage_s"] for r in rows[ds] if r["method"] == m and r["T"] == is_t), None)
                for m in ("qalsh4c", "quadtree")}
        for r in rows[ds]:
            if r["method"] == "load":
                continue
            err = 100 * r["mean_err"]
            if r["method"] in FAMILY:
                crude = r["build_s"] + r["query_s"] + r["dist_s"]
                if r["T"] == is_t:
                    pts["is"][r["method"]].add((load_a + crude + r["stage_s"], err))
                pts["sum"][r["method"]].add((r["load_s"] + crude, 100 * r["sum_err"]))
            elif r["method"] == "uniform":
                uniform.append((r["stage_s"], err))
            elif r["method"] == "quadtree" and base["quadtree"] is not None and c.get("quadtree_s"):
                quad.append((c["quadtree_s"] + r["stage_s"] - base["quadtree"], err))
            elif r["method"] == "qalsh4c":
                qalsh.append((c["qalsh4c_s"] + r["stage_s"] - base["qalsh4c"], err))
        exact = c["best_exact_s"]
        panels.append((ds, pts, sorted(uniform), sorted(quad), sorted(qalsh), exact))
        for kind in pts.values():
            for s in kind.values():
                for x, y in s:
                    xs.append(x)
                    ys.append(y)
        xs += [x for x, _ in uniform + quad + qalsh] + [exact]
        ys += [y for _, y in uniform + quad + qalsh]
    ys = [y for y in ys if y > 0]
    ymin = 10 ** math.floor(math.log10(min(ys)))
    ymax = 10 ** math.ceil(math.log10(max(ys)))

    out = [r"""\documentclass[border=2pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc}
\definecolor{cq}{HTML}{2A78D6}
\definecolor{cu}{HTML}{888888}
\definecolor{ch}{HTML}{EB6834}
\definecolor{cf}{HTML}{1BAF7A}
\definecolor{cp}{HTML}{9B59B6}
\definecolor{ct}{HTML}{C9A227}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
    group style={group size=4 by %d, horizontal sep=0.45cm, vertical sep=1.6cm, y descriptions at=edge left},
    width=5.1cm, height=4.0cm,
    xmode=log, ymode=log, ymin=%g, ymax=%g,
    ymajorgrids, xmajorgrids, grid style={black!10},
    tick label style={font=\footnotesize}, label style={font=\small}, title style={font=\small, yshift=-4pt},
    every axis plot/.append style={mark size=1.7pt},
]
""" % ((len(panels) + 3) // 4, ymin, ymax)]
    for k, (ds, pts, uniform, quad, qalsh, exact) in enumerate(panels):
        px = [x for kind in pts.values() for v in kind.values() for x, _ in v] + [x for x, _ in uniform + quad + qalsh]
        px.append(exact)
        opts = [f"title={{{NAME.get(ds, ds)}}}", "xlabel={Time (s)}",
                f"xmin={10 ** (math.floor(math.log10(min(px)) * 2) / 2):g}",
                f"xmax={10 ** (math.ceil(math.log10(max(px)) * 2) / 2):g}"]
        if k % 4 == 0:
            opts.append("ylabel={Mean error (\\%)}")
        if k == 0:
            opts.append(r"legend to name=annlegend, legend columns=6, legend style={draw=none, font=\footnotesize,"
                        r" /tikz/every even column/.append style={column sep=4pt}}")
        out.append(r"\nextgroupplot[" + ", ".join(opts) + "]\n")
        for fam, (color, filled, hollow, label) in FAMILY.items():
            s = sorted(pts["sum"].get(fam, []))
            out.append(r"\addplot[forget plot, only marks, %s, mark=%s, mark options={fill=white}] coordinates {%s};" % (
                color, hollow, " ".join(f"({x:.3f},{y:.4f})" for x, y in s if y > 0)) + "\n")
            s = sorted(pts["is"].get(fam, []))
            out.append(r"\addplot[forget plot, only marks, %s, mark=%s] coordinates {%s};" % (
                color, filled, " ".join(f"({x:.3f},{y:.4f})" for x, y in s if y > 0)) + "\n")
        out.append(r"\addplot[forget plot, cu, thick, mark=*, mark size=1.2pt] coordinates {%s};" % (
            " ".join(f"({x:.3f},{y:.4f})" for x, y in uniform)) + "\n")
        out.append(r"\addplot[forget plot, ct, thick, mark=pentagon*, mark size=1.4pt] coordinates {%s};" % (
            " ".join(f"({x:.3f},{y:.4f})" for x, y in quad)) + "\n")
        out.append(r"\addplot[forget plot, cq, very thick, mark=diamond*, mark size=2.4pt] coordinates {%s};" % (
            " ".join(f"({x:.3f},{y:.4f})" for x, y in qalsh)) + "\n")
        if k == 0:
            # Legend entries are drawn explicitly, so a family without points in a panel cannot shift them.
            for fam, (color, filled, hollow, label) in FAMILY.items():
                out.append(r"\addlegendimage{only marks, %s, mark=%s, mark options={fill=white}}\addlegendentry{%s-Sum}"
                           % (color, hollow, label) + "\n")
                out.append(r"\addlegendimage{only marks, %s, mark=%s}\addlegendentry{%s-IS}" % (color, filled, label)
                           + "\n")
            out.append(r"\addlegendimage{cu, thick, mark=*, mark size=1.2pt}\addlegendentry{Uniform}" + "\n")
            out.append(r"\addlegendimage{ct, thick, mark=pentagon*, mark size=1.4pt}\addlegendentry{Quadtree}" + "\n")
            out.append(r"\addlegendimage{cq, very thick, mark=diamond*, mark size=2.4pt}\addlegendentry{QALSH4C}" + "\n")
            out.append(r"\addlegendimage{black!70, dashed, thick}\addlegendentry{Linear scan}" + "\n")
        out.append(r"\draw[black!70, dashed, thick] (axis cs:%g,%g) -- (axis cs:%g,%g);" % (exact, ymin, exact, ymax)
                   + "\n")
    out.append("\\end{groupplot}\n")
    out.append(r"\node[anchor=south] at ($(group c1r1.north west)!0.5!(group c4r1.north east)+(0,14pt)$)"
               r" {\pgfplotslegendfromname{annlegend}};" + "\n")
    if placeholder:
        out.append(r"\node[red, opacity=0.45, font=\Huge\bfseries] at (current bounding box.center) {PLACEHOLDER};"
                   + "\n")
    out.append("\\end{tikzpicture}\n\\end{document}\n")
    with open(out_path, "w") as f:
        f.write("".join(out))


if __name__ == "__main__":
    main()
