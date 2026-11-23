#include "kappa.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace kappa {
namespace {

// ---------------------------------------------------------------------------------------------------------------
// Interval arithmetic on doubles.
//
// Under round-to-nearest, every +, -, *, / and sqrt returns a result within half a unit in the last place of the exact
// value, so the exact value lies between the next double below and the next double above the computed one. Every
// operation below widens its result to those neighbors, so an interval always contains the exact value it stands for.
// This file is compiled with -ffp-contract=off, so the compiler does not fuse operations behind our back.
// ---------------------------------------------------------------------------------------------------------------

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// The next double above x (x itself for +inf and NaN).
double Up(double x) {
    if (!(x < kInfinity)) {
        return x;
    }
    if (x == 0.0) {
        return std::numeric_limits<double>::denorm_min();
    }
    auto bits = std::bit_cast<std::uint64_t>(x);
    bits = x > 0.0 ? bits + 1 : bits - 1;
    return std::bit_cast<double>(bits);
}

// The next double below x.
double Down(double x) { return -Up(-x); }

struct Interval {
    double lo;
    double hi;
};

Interval Exact(double x) { return {x, x}; }

Interval operator+(Interval a, Interval b) { return {Down(a.lo + b.lo), Up(a.hi + b.hi)}; }
Interval operator-(Interval a, Interval b) { return {Down(a.lo - b.hi), Up(a.hi - b.lo)}; }

// Products, quotients and square roots are only formed of quantities known to be non-negative, so negative lower
// ends (left by a subtraction such as 1 - rho) can be raised to zero.
Interval NonNegative(Interval a) { return {std::max(a.lo, 0.0), a.hi}; }

Interval operator*(Interval a, Interval b) {
    a = NonNegative(a);
    b = NonNegative(b);
    return {std::max(0.0, Down(a.lo * b.lo)), Up(a.hi * b.hi)};
}

Interval operator/(Interval a, Interval b) {
    a = NonNegative(a);
    if (!(b.lo > 0.0)) {
        throw std::logic_error("kappa: interval division by an interval that is not positive");
    }
    return {std::max(0.0, Down(a.lo / b.hi)), Up(a.hi / b.lo)};
}

Interval Sqrt(Interval a) {
    a = NonNegative(a);
    return {std::max(0.0, Down(std::sqrt(a.lo))), Up(std::sqrt(a.hi))};
}

Interval Pow(Interval a, unsigned int exponent) {
    Interval result = Exact(1.0);
    for (unsigned int i = 0; i < exponent; ++i) {
        result = result * a;
    }
    return result;
}

// pi lies between these two neighboring doubles.
constexpr Interval kPi{0x1.921fb54442d18p+1, 0x1.921fb54442d19p+1};

// atan(t) for a point 0 <= t <= 1 from its alternating Taylor series sum_k (-1)^k t^(2k+1) / (2k+1). The terms
// decrease, so after the terms k < K the remainder has the sign of term K and is at most its size.
Interval AtanSeries(double t) {
    const Interval x = Exact(t);
    const Interval x2 = x * x;
    Interval power = x;  // t^(2k+1)
    Interval added = Exact(0.0);
    Interval subtracted = Exact(0.0);
    for (unsigned int k = 0;; ++k) {
        const Interval term = power / Exact(2.0 * k + 1.0);
        if (term.hi <= 1e-20) {
            const Interval sum = added - subtracted;
            return k % 2 == 0 ? Interval{sum.lo, Up(sum.hi + term.hi)} : Interval{Down(sum.lo - term.hi), sum.hi};
        }
        if (k % 2 == 0) {
            added = added + term;
        } else {
            subtracted = subtracted + term;
        }
        power = power * x2;
    }
}

// atan(z) for a point z >= 0. Three halvings, atan u = 2 atan(u / (1 + sqrt(1 + u^2))), bring the argument below
// tan(pi/16) < 0.2, where the series converges fast; atan is increasing, so its values at the ends of the reduced
// interval enclose the reduced angle.
Interval AtanOfPoint(double z) {
    Interval u = Exact(z);
    for (int i = 0; i < 3; ++i) {
        u = u / (Exact(1.0) + Sqrt(Exact(1.0) + u * u));
    }
    return Exact(8.0) * Interval{AtanSeries(u.lo).lo, AtanSeries(u.hi).hi};
}

// e^t for a point t >= 0 from sum_k t^k / k!. Once k + 2 > 2t, the later terms shrink by factors of at most
// t / (k + 2) < 1/2, so they add at most next / (1 - t / (k + 2)).
Interval ExpOfPoint(double t) {
    const Interval x = Exact(t);
    Interval term = Exact(1.0);  // t^k / k!
    Interval sum = Exact(0.0);
    for (unsigned int k = 0;; ++k) {
        sum = sum + term;
        const Interval next = term * x / Exact(k + 1.0);
        if (k + 2.0 > 2.0 * t && next.hi <= 1e-20 * sum.lo) {
            const double ratio = Up(t / (k + 2.0));
            const double tail = Up(next.hi / Down(1.0 - ratio));
            return {sum.lo, Up(sum.hi + tail)};
        }
        term = next;
    }
}

// e^(-t) for t in a non-negative interval; e^(-t) decreases in t.
Interval ExpOfNegative(Interval t) {
    t = NonNegative(t);
    return Exact(1.0) / Interval{ExpOfPoint(t.lo).lo, ExpOfPoint(t.hi).hi};
}

// erf(y) for a point y >= 0 from erf(y) = (2 / sqrt(pi)) e^(-y^2) sum_k 2^k y^(2k+1) / (1 * 3 * ... * (2k+1)). The
// terms are positive and their ratios 2y^2 / (2k+3) decrease, so once a ratio r is below 1/2 the later terms add at
// most next / (1 - r).
Interval ErfOfPoint(double y) {
    const Interval x = Exact(y);
    const Interval x2 = x * x;
    const Interval two_x2 = Exact(2.0) * x2;
    Interval term = x;
    Interval sum = Exact(0.0);
    for (unsigned int k = 0;; ++k) {
        sum = sum + term;
        const Interval ratio = two_x2 / Exact(2.0 * k + 3.0);
        const Interval next = term * ratio;
        if (ratio.hi < 0.5 && next.hi <= 1e-20 * sum.lo) {
            sum = {sum.lo, Up(sum.hi + Up(next.hi / Down(1.0 - ratio.hi)))};
            break;
        }
        term = next;
    }
    Interval value = Exact(2.0) / Sqrt(kPi) * ExpOfNegative(x2) * sum;
    value.hi = std::min(value.hi, 1.0);
    return value;
}

Interval ErfcOfPoint(double y) {
    Interval value = NonNegative(Exact(1.0) - ErfOfPoint(y));
    value.hi = std::min(value.hi, 1.0);
    return value;
}

// rho(x), the probability that a vector at distance s misses one projection line at radius x s.
Interval MissProbability(int norm, double bucket_width, double x) {
    const Interval wx = Exact(bucket_width) * Exact(x);
    Interval rho{};
    if (norm == 1) {
        // (2 / pi) atan(2 / (w x)); atan is increasing.
        const Interval z = Exact(2.0) / wx;
        rho = Exact(2.0) * Interval{AtanOfPoint(z.lo).lo, AtanOfPoint(z.hi).hi} / kPi;
    } else {
        // erfc(w x / (2 sqrt 2)); erfc is decreasing.
        const Interval y = wx / Sqrt(Exact(8.0));
        rho = {ErfcOfPoint(y.hi).lo, ErfcOfPoint(y.lo).hi};
    }
    return {std::max(rho.lo, 0.0), std::min(rho.hi, 1.0)};
}

// F = Pr[Binomial(m, rho) >= h] = sum_{k >= h} C(m, k) rho^k (1 - rho)^(m - k). Every term increases in rho and in
// 1 - rho, so the ends of the enclosures of rho and 1 - rho give the ends of the enclosure of F.
Interval BinomialUpperTail(const std::vector<Interval>& coefficients, unsigned int m, unsigned int h, Interval rho,
                           std::vector<Interval>& complement_powers) {
    Interval complement = NonNegative(Exact(1.0) - rho);
    complement.hi = std::min(complement.hi, 1.0);
    complement_powers[0] = Exact(1.0);
    for (unsigned int i = 1; i <= m - h; ++i) {
        complement_powers[i] = complement_powers[i - 1] * complement;
    }
    Interval power = Pow(rho, h);
    Interval sum = Exact(0.0);
    for (unsigned int k = h; k <= m; ++k) {
        sum = sum + coefficients[k] * power * complement_powers[m - k];
        power = power * rho;
    }
    sum.hi = std::min(sum.hi, 1.0);
    return sum;
}

// Lemma 3: an upper bound on int_x^inf F(t) dt from the union bound F <= C(m, h) rho^h and
// rho(t) <= 4 / (pi w t) (l1) or rho(t) <= A2 t^-1 e^(-lambda t^2) with A2 = 2 sqrt 2 / (sqrt(pi) w), lambda = w^2 / 8
// (l2), using t^-h <= x^-h and int_x^inf e^(-h lambda t^2) dt <= e^(-h lambda x^2) / (2 h lambda x).
double TailBound(int norm, double bucket_width, unsigned int h, Interval coefficient, double x) {
    const Interval w = Exact(bucket_width);
    const Interval point = Exact(x);
    if (norm == 1) {
        const Interval a = Exact(4.0) / (kPi * w);
        return (coefficient * Pow(a, h) / (Pow(point, h - 1) * Exact(h - 1.0))).hi;
    }
    const Interval a2 = Exact(2.0) * Sqrt(Exact(2.0)) / (Sqrt(kPi) * w);
    const Interval lambda = w * w / Exact(8.0);
    const Interval exponent = Exact(static_cast<double>(h)) * lambda * point * point;
    return (coefficient * Pow(a2, h) / Pow(point, h) * ExpOfNegative(exponent) /
            (Exact(2.0 * h) * lambda * point))
        .hi;
}

// Double-precision F, used only to place the grid points; it does not enter the bound.
double ApproximateTail(int norm, double bucket_width, unsigned int m, unsigned int h,
                       const std::vector<double>& coefficients, double x) {
    const double rho = norm == 1 ? 2.0 / std::numbers::pi * std::atan(2.0 / (bucket_width * x))
                                 : std::erfc(bucket_width * x / (2.0 * std::numbers::sqrt2));
    double sum = 0.0;
    for (unsigned int k = h; k <= m; ++k) {
        sum += coefficients[k] * std::pow(rho, k) * std::pow(1.0 - rho, m - k);
    }
    return std::min(sum, 1.0);
}

constexpr int kGridBits = 22;   // grid points are integer multiples of 2^-22, so points and cell widths are exact
constexpr int kCellBits = 12;   // the grid is placed cell by cell, cells of width 2^-12 ...
constexpr unsigned int kMaxSplit = 1U << (kGridBits - kCellBits);  // ... each split into at most 2^10 pieces
constexpr std::uint64_t kCellsPerSixteenth = std::uint64_t{1} << (kCellBits - 4);
constexpr double kTailLimit = 1e-13;
constexpr std::size_t kMaxGridPoints = 100'000'000;

}  // namespace

Bound CertifyKappa(const Parameters& parameters) {
    const int norm = parameters.norm;
    const double w = parameters.bucket_width;
    const unsigned int m = parameters.num_hash_tables;
    const unsigned int l = parameters.collision_threshold;
    const double gamma = parameters.gamma;
    if (norm != 1 && norm != 2) {
        throw std::invalid_argument("kappa: the norm must be 1 or 2");
    }
    if (!(w > 0.0) || !std::isfinite(w)) {
        throw std::invalid_argument("kappa: the bucket width must be positive");
    }
    if (!(gamma > 1.0) || !std::isfinite(gamma)) {
        throw std::invalid_argument("kappa: gamma must be greater than 1");
    }
    if (l == 0 || l >= m) {
        // h = m - l + 1 >= 2 keeps the integral finite (Lemma 3).
        throw std::invalid_argument("kappa: the collision threshold l must satisfy 1 <= l < m");
    }
    const unsigned int h = m - l + 1;

    std::vector<Interval> coefficients(m + 1);
    std::vector<double> approximate_coefficients(m + 1);
    coefficients[0] = Exact(1.0);
    approximate_coefficients[0] = 1.0;
    for (unsigned int k = 1; k <= m; ++k) {
        coefficients[k] = coefficients[k - 1] * Exact(m - k + 1.0) / Exact(k);
        approximate_coefficients[k] = approximate_coefficients[k - 1] * (m - k + 1.0) / k;
    }

    // The grid starts at x_0 = i_0 2^-22 <= 1/gamma; the sign of the fused i_0 gamma - 2^22 is exact.
    const double scale = std::ldexp(1.0, kGridBits);
    auto first = static_cast<std::uint64_t>(std::floor(scale / gamma));
    while (std::fma(static_cast<double>(first), gamma, -scale) > 0.0) {
        --first;
    }
    const bool starts_at_one_over_gamma = std::fma(static_cast<double>(first), gamma, -scale) == 0.0;
    const auto point = [&](std::uint64_t index) { return std::ldexp(static_cast<double>(index), -kGridBits); };

    // The grid ends at the first multiple of 1/16 past x_0 whose tail bound is at most kTailLimit.
    std::uint64_t cells = 0;
    do {
        cells += kCellsPerSixteenth;
        if (cells > (std::uint64_t{1} << 30)) {
            throw std::runtime_error("kappa: the tail bound does not fall below its limit");
        }
    } while (TailBound(norm, w, h, coefficients[h], point(first + cells * kMaxSplit)) > kTailLimit);

    // A cell of width u where F falls by d adds about u d / n to the gap between the upper and the lower sum when it is
    // split into n pieces. Splitting cell j into n_j proportional to sqrt(u d_j) pieces gives every cell the same share
    // of a total gap of kKappaSlack / gamma with the fewest points; n_j is rounded up to a power of two (at most 2^10)
    // so that the points stay multiples of 2^-22.
    const double cell_width = std::ldexp(1.0, -kCellBits);
    std::vector<double> weight(cells);
    double total_weight = 0.0;
    double previous = ApproximateTail(norm, w, m, h, approximate_coefficients, point(first));
    for (std::uint64_t j = 0; j < cells; ++j) {
        const double next = ApproximateTail(norm, w, m, h, approximate_coefficients, point(first + (j + 1) * kMaxSplit));
        weight[j] = std::sqrt(cell_width * std::max(previous - next, 0.0));
        total_weight += weight[j];
        previous = next;
    }
    const double target_gap = kKappaSlack / gamma;
    std::vector<std::uint64_t> grid;
    for (std::uint64_t j = 0; j < cells; ++j) {
        const double wanted = total_weight > 0.0 ? weight[j] * total_weight / target_gap : 1.0;
        unsigned int split = 1;
        while (split < kMaxSplit && split < wanted) {
            split *= 2;
        }
        const unsigned int step = kMaxSplit / split;
        for (unsigned int k = 0; k < split; ++k) {
            grid.push_back(first + j * kMaxSplit + k * step);
        }
        if (grid.size() > kMaxGridPoints) {
            throw std::runtime_error("kappa: the grid grew too large");
        }
    }
    grid.push_back(first + cells * kMaxSplit);

    // F at every grid point, in interval arithmetic.
    const std::size_t count = grid.size();
    std::vector<double> upper_values(count);
    std::vector<double> lower_values(count);
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel
#endif
    {
        std::vector<Interval> complement_powers(m - h + 1);
#ifdef QALSH4C_USE_OPENMP
#pragma omp for schedule(static)
#endif
        for (std::size_t k = 0; k < count; ++k) {
            const Interval value = BinomialUpperTail(coefficients, m, h, MissProbability(norm, w, point(grid[k])),
                                                     complement_powers);
            upper_values[k] = value.hi;
            lower_values[k] = value.lo;
        }
    }

    // F does not increase, so on [x_k, x_k+1] it lies between F(x_k+1) and F(x_k). The upper sum (plus the tail bound)
    // is at least int_{1/gamma}^inf F; the lower sum, without the first cell when x_0 < 1/gamma, is at most it. The sums
    // run in a fixed order, so the result does not depend on the number of threads.
    double upper_sum = 0.0;
    double lower_sum = 0.0;
    for (std::size_t k = 0; k + 1 < count; ++k) {
        const Interval width = Exact(point(grid[k + 1] - grid[k]));
        upper_sum = (Exact(upper_sum) + width * Exact(upper_values[k])).hi;
        if (k > 0 || starts_at_one_over_gamma) {
            lower_sum = (Exact(lower_sum) + width * Exact(lower_values[k + 1])).lo;
        }
    }
    const double tail = TailBound(norm, w, h, coefficients[h], point(grid.back()));
    const double upper = (Exact(1.0) + Exact(gamma) * (Exact(upper_sum) + Exact(tail))).hi;
    const double lower = (Exact(1.0) + Exact(gamma) * Exact(lower_sum)).lo;
    if (!std::isfinite(upper)) {
        throw std::runtime_error("kappa: the bound is not finite");
    }

    Bound bound;
    bound.millionths = static_cast<std::uint64_t>(std::ceil(Up(upper * 1e6)));
    bound.upper = upper;
    bound.lower = lower;
    bound.grid_points = count;
    bound.grid_end = point(grid.back());
    return bound;
}

std::uint64_t SampleSize(std::uint64_t millionths, double epsilon, double delta) {
    if (!(epsilon > 0.0) || !(delta > 0.0) || !(delta < 1.0)) {
        throw std::invalid_argument("kappa: epsilon must be positive and delta must lie in (0, 1)");
    }
    if (millionths <= 1'000'000) {
        return 1;
    }
    // (millionths - 10^6) / (10^6 delta epsilon^2), rounded up; the numerator is an exact integer.
    const Interval denominator = Exact(1e6) * Exact(delta) * Exact(epsilon) * Exact(epsilon);
    const double value = (Exact(static_cast<double>(millionths - 1'000'000)) / denominator).hi;
    if (!(value < 1e18)) {
        throw std::runtime_error("kappa: the sample size is too large");
    }
    return std::max<std::uint64_t>(1, static_cast<std::uint64_t>(std::ceil(value)));
}

}  // namespace kappa
