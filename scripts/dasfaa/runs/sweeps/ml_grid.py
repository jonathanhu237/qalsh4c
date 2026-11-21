"""Quick double-precision map of kappa(m, l) at c = 2 and a data-independent counting proxy.

proxy = m * E[max(z_(l:m), w/4)]: the window half-width at the stop, in units of d(a,B), is the larger of the l-th
smallest |projection coefficient| among m tables (the NN must collide l times) and w/4 (round-end T1 needs
d(a,B) <= cR); far points fall inside a window with probability roughly proportional to its width.
"""
import math, sys
import numpy as np
from scipy.stats import binom

c = 2.0
rng = np.random.default_rng(1)
N = 200000
for norm in (1, 2):
    if norm == 1:
        w = float(np.float32(2.0 * math.sqrt(c)))
        pc = lambda s: 2.0 / math.pi * math.atan(w / (2.0 * s))
        z = np.abs(rng.standard_cauchy((N, 64)))
    else:
        w = float(np.float32(math.sqrt(8.0 * c * c * math.log(c) / (c * c - 1.0))))
        pc = lambda s: math.erf(w / (2.0 * math.sqrt(2.0) * s))
        z = np.abs(rng.standard_normal((N, 64)))
    rows = []
    for m in range(2, 41 if norm == 2 else 61):
        zs = np.sort(z[:, :m], axis=1)
        for l in range(1, m + 1):
            S = 0.0
            for j in range(0, 60):
                rho = 1.0 - pc(c ** (-j))
                d = binom.sf(m - l, m, rho)  # Pr[misses >= m-l+1]
                S += c ** j * d
                if c ** j * d < 1e-14 and j > 3:
                    break
            kappa = c * (1 + (c - 1) * S)
            proxy = m * float(np.mean(np.maximum(zs[:, l - 1], w / 4)))
            rows.append((m, l, kappa, proxy))
    print(f"== l{norm} w={w:.4f}")
    # Pareto frontier: lowest proxy for kappa <= cap
    for cap in (2.05, 2.08, 2.10, 2.12, 2.15, 2.20, 2.30, 2.50):
        best = min((r for r in rows if r[2] <= cap), key=lambda r: r[3], default=None)
        print(f"kappa<={cap}: m={best[0]} l={best[1]} kappa={best[2]:.4f} proxy={best[3]:.2f}")
    for (m, l) in ((17, 12), (36, 18), (8, 5), (10, 7), (12, 8), (6, 4), (22, 12), (16, 8), (12, 6)):
        r = [x for x in rows if x[0] == m and x[1] == l]
        if r:
            print(f"  m={m} l={l} kappa={r[0][2]:.4f} proxy={r[0][3]:.2f}")
