"""Certified upper bounds on the QALSH-CrudeNN constant kappa and the sample sizes T they give.

kappa = C * (1 + (c - 1) * (S_J + B_J)), C = c for round-end stopping (the paper's Algorithm 1) and C = c^2 for
within-round stopping, S_J = sum_{j <= J} c^j Delta_j with Delta_j = Pr[Bin(m, rho_j) >= h], rho_j = 1 - p_col(c^-j),
h = m - l + 1, and B_J the analytic tail bound of Lemma 1 (increase J until B_J <= 1e-10).

The prefix is evaluated in mpmath interval arithmetic (mpmath.iv), so the upper endpoint encloses the exact value;
B_J is an upper bound by Lemma 1 and is also evaluated with outward rounding. The bucket width is the float32 value the
implementation uses. The final kappa is rounded up to six decimals.
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction

import mpmath
import numpy as np

iv = mpmath.iv
iv.dps = 60


def params(c: float, n_b: int, norm: int, beta: float | None = 1.0) -> tuple[float, int, int]:
    """QALSH parameters exactly as the implementation computes them (float32 width, delta_Q = 1/e).

    beta = 1 is the QALSH4C default (no candidate budget); beta = None gives QALSH's beta = min(100/n_B, 1).
    """
    beta = min(100.0 / n_b, 1.0) if beta is None else min(beta, 1.0)
    gb = math.sqrt(math.log(2.0 / beta))
    gq = 1.0
    xi = gb / gq
    if norm == 1:
        w = float(np.float32(2.0 * math.sqrt(c)))
        pc = lambda s: 2.0 / math.pi * math.atan(w / (2.0 * s))
    else:
        w = float(np.float32(math.sqrt(8.0 * c * c * math.log(c) / (c * c - 1.0))))
        pc = lambda s: math.erf(w / (2.0 * math.sqrt(2.0) * s))
    p1, p2 = pc(1.0), pc(c)
    m = math.ceil((gb + gq) ** 2 / (2.0 * (p1 - p2) ** 2))
    l = math.ceil(m * (xi * p1 + p2) / (1.0 + xi))
    return w, m, l


def erf_enclosure(x) -> "iv.mpf":
    """Rigorous interval enclosure of erf(x) for one point x >= 0 (an mpf), from the Maclaurin series.

    erf(x) = 2/sqrt(pi) * sum_k (-1)^k x^(2k+1) / (k! (2k+1)). Once the terms decrease, the series alternates with
    decreasing terms, so the remainder after term K lies between 0 and the next term (sign of that term). The terms are
    evaluated in interval arithmetic, so the enclosure is rigorous up to the interval rounding.
    """
    X = iv.mpf(x)
    total = iv.mpf(0)
    k = 0
    power = X  # x^(2k+1)
    factorial = iv.mpf(1)  # k!
    previous = None
    while True:
        term = power / (factorial * (2 * k + 1))
        if previous is not None and k > float(x) ** 2 + 2 and term.b < previous.a and term.b < mpmath.mpf("1e-55"):
            # Remainder of an alternating series with decreasing terms: between 0 and the first omitted term.
            remainder = iv.mpf([-term.b, term.b])
            return 2 / iv.sqrt(iv.pi) * (total + remainder)
        total += term if k % 2 == 0 else -term
        previous = term
        k += 1
        power *= X * X
        factorial *= k


def miss_probability(norm: int, w, cj):
    """rho_j = 1 - p_col(c^-j) as an interval.

    l1: rho = 2/pi * atan(2/(w c^j)) in interval arithmetic. l2: rho = erfc(w c^j / (2 sqrt 2)); erfc is decreasing,
    so the interval is [1 - erf(upper end), 1 - erf(lower end)] with erf enclosed by erf_enclosure.
    """
    if norm == 1:
        return 2 / iv.pi * iv.atan2(iv.mpf(2), w * cj)
    x = w * cj / (2 * iv.sqrt(2))
    if float(x.b) > 12:
        raise SystemExit("erf_enclosure is meant for moderate arguments; increase precision or use the tail bound")
    low = 1 - erf_enclosure(mpmath.mpf(x.b))
    high = 1 - erf_enclosure(mpmath.mpf(x.a))
    return iv.mpf([low.a, high.b])


def binomial_upper_tail(m: int, h: int, rho):
    total = iv.mpf(0)
    for k in range(h, m + 1):
        total += iv.mpf(math.comb(m, k)) * rho ** k * (1 - rho) ** (m - k)
    return total


def tail_bound(norm: int, m: int, h: int, w, c, J: int):
    binom = iv.mpf(math.comb(m, h))
    if norm == 1:
        return binom * (4 / (iv.pi * w)) ** h * c ** (-(J + 1) * (h - 1)) / (1 - c ** (-(h - 1)))
    a2 = 2 * iv.sqrt(2) / (iv.sqrt(iv.pi) * w)
    lam = w ** 2 / 8
    u = lambda j: binom * a2 ** h * c ** (-j * (h - 1)) * iv.exp(-lam * h * c ** (2 * j))
    u1, u2 = u(J + 1), u(J + 2)
    return u1 / (1 - u2 / u1)


def certify(c: float, n_b: int, norm: int, round_end: bool = True, beta: float | None = 1.0) -> dict:
    w_f, m, l = params(c, n_b, norm, beta)
    r = certify_params(c, w_f, m, l, norm, round_end)
    r["n_B"] = n_b
    return r


def certify_params(c: float, w_f: float, m: int, l: int, norm: int, round_end: bool = True) -> dict:
    """Certified kappa for explicit (c, w, m, l)."""
    n_b = None
    h = m - l + 1
    if h < 2:
        raise SystemExit(f"h = {h} < 2 for c={c}, n_B={n_b}, l{norm}")
    w = iv.mpf(w_f)
    C = iv.mpf(c)
    prefix = iv.mpf(0)
    J = 0
    while True:
        cj = C ** J
        prefix += cj * binomial_upper_tail(m, h, miss_probability(norm, w, cj))
        b = tail_bound(norm, m, h, w, C, J)
        if b.b <= 1e-10:
            break
        J += 1
    # Prefactor c for the hybrid rule (stop inside a round on a candidate within R, at round end on one within cR) and
    # for the pure round-end rule; c^2 for the legacy rule that accepts a candidate within cR inside a round.
    factor = C if round_end else C * C
    kappa = factor * (1 + (C - 1) * (prefix + b))
    # Round the upper endpoint up to whole millionths in exact arithmetic (no binary64 conversion before the ceiling).
    millionths = int(mpmath.ceil((kappa * 10**6).b))
    kappa_up = millionths / 1e6
    return {"c": c, "n_B": n_b, "norm": norm, "w": w_f, "m": m, "l": l, "h": h, "J": J,
            "kappa_up": kappa_up, "kappa_millionths": millionths,
            "kappa_hi": float(kappa.b), "kappa_lo": float(kappa.a)}


def sample_size(kappa_up: float, eps: float, delta: float = 0.1) -> int:
    """T = ceil((kappa_up - 1) / (delta eps^2)) in exact rational arithmetic (eps and delta read as decimals)."""
    k = Fraction(str(kappa_up))
    t = (k - 1) / (Fraction(str(delta)) * Fraction(str(eps)) ** 2)
    return max(1, math.ceil(t))


def main() -> None:
    datasets = {"Federalist": 1178, "MNIST": 67000, "Trevi": 96120, "GIST": 990000}
    print("iv.erfc available:", hasattr(iv, "erfc"), file=sys.stderr)
    print("== beta = 1 (QALSH4C default; m and l do not depend on n_B), round-end stopping")
    for c in (1.5, 2.0, 3.0):
        for norm in (1, 2):
            r = certify(c, 100, norm, True, 1.0)
            ts = [sample_size(r["kappa_up"], e) for e in (0.3, 0.1)]
            print(f"c={c} l{norm} w={r['w']:.7f} m={r['m']:3d} l={r['l']:3d} h={r['h']:2d} J={r['J']} "
                  f"kappa<={r['kappa_up']:.6f} [{r['kappa_lo']:.9f}, {r['kappa_hi']:.9f}] T(.3,.1)={ts}")
    for round_end in (True, False):
        print(f"== c = 2, beta = min(100/n_B, 1), {'round-end' if round_end else 'within-round'} stopping")
        for name, n_b in datasets.items():
            for norm in (1, 2):
                r = certify(2.0, n_b, norm, round_end, None)
                ts = [sample_size(r["kappa_up"], e) for e in (1.0, 0.3, 0.1, 0.01)]
                print(f"{name:10s} l{norm} w={r['w']:.7f} m={r['m']:3d} l={r['l']:3d} h={r['h']:2d} J={r['J']} "
                      f"kappa<={r['kappa_up']:.6f} [{r['kappa_lo']:.9f}, {r['kappa_hi']:.9f}] T(1,.3,.1,.01)={ts}")
    print("== c sweep, beta = min(100/n_B, 1), round-end stopping")
    for c in (1.5, 2.0, 3.0):
        for name in ("MNIST", "Trevi"):
            for norm in (1, 2):
                r = certify(c, datasets[name], norm, True, None)
                print(f"c={c} {name:6s} l{norm} m={r['m']:3d} l={r['l']:3d} kappa<={r['kappa_up']:.6f} "
                      f"T(eps=0.1)={sample_size(r['kappa_up'], 0.1)}")


if __name__ == "__main__":
    main()
