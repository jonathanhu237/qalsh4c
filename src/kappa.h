#pragma once

#include <cstddef>
#include <cstdint>

// Certified upper bound on the constant kappa of QALSH-CrudeNN (Theorem 2 of the paper) and the sample size it gives
// (Theorem 1).
//
// A search that returns only when its best distance is at most gamma times a radius up to which every projection line
// has been processed satisfies E[D_a] <= kappa d(a, B) with
//
//     kappa = 1 + gamma * int_{1/gamma}^inf F(x) dx,    F(x) = Pr[Binomial(m, rho(x)) >= h],    h = m - l + 1,
//
// where rho(x) = 1 - p_col(1/x) is the probability that a vector at distance s misses one line at radius x s:
// rho(x) = (2/pi) atan(2/(w x)) for l1 and erfc(w x / (2 sqrt 2)) for l2. The continuous front has gamma = c.
//
// CertifyKappa evaluates an upper Riemann sum of the non-increasing F on a grid of dyadic points plus the closed-form
// tail bound of Lemma 3, in interval arithmetic: every floating-point operation is widened by one unit in the last
// place, and atan, erf and exp are enclosed by series with explicit remainder bounds. The result is therefore a proven
// upper bound on kappa, not an estimate.
namespace kappa {

struct Parameters {
    int norm{2};                         // 1 or 2
    double bucket_width{0.0};            // w, the value the index uses
    unsigned int num_hash_tables{0};     // m
    unsigned int collision_threshold{0};  // l, with 1 <= l < m
    double gamma{2.0};                   // stopping factor, > 1
};

struct Bound {
    std::uint64_t millionths{0};  // kappa <= millionths / 10^6
    double upper{0.0};            // proven upper bound on kappa before rounding
    double lower{0.0};            // proven lower bound on kappa (lower Riemann sum), shows how tight the bound is
    std::size_t grid_points{0};
    double grid_end{0.0};         // where the Riemann sum ends and the tail bound takes over
};

// Proven upper bound on kappa. The grid is refined until the upper and the lower sum differ by about
// kKappaSlack in kappa, so the rounded bound exceeds kappa by little more than that.
inline constexpr double kKappaSlack = 1e-6;
Bound CertifyKappa(const Parameters& parameters);

// Smallest T with T >= (kappa_up - 1) / (delta epsilon^2), kappa_up = millionths / 10^6 (Theorem 1).
std::uint64_t SampleSize(std::uint64_t millionths, double epsilon, double delta);

}  // namespace kappa
