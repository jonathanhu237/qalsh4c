#include "pool_plan.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace PoolPlan {

namespace {

// Budgets are clamped here; a run that needs more draws than this is not worth starting anyway.
constexpr double kMaxCount = 1e9;

unsigned int CeilCount(double value) {
    if (!(value > 1.0)) {  // Also maps NaN to one draw.
        return 1;
    }
    return static_cast<unsigned int>(std::ceil(std::min(value, kMaxCount)));
}

double Mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

struct PairStatistics {
    double mean_x{0.0};
    std::vector<double> h0;
    std::vector<double> h1;
};

// The mean of x over every draw, and H0 and H1 over the disjoint pairs (0, 1), (2, 3), ...; an odd last draw only
// enters the mean.
PairStatistics ComputePairStatistics(const std::vector<PilotDraw>& draws) {
    if (draws.size() < 2) {
        throw std::invalid_argument("A pool pilot needs at least two draws.");
    }
    PairStatistics statistics;
    for (const PilotDraw& draw : draws) {
        statistics.mean_x += draw.x;
    }
    statistics.mean_x /= static_cast<double>(draws.size());
    for (size_t index = 0; index + 1 < draws.size(); index += 2) {
        const PilotDraw& first = draws[index];
        const PilotDraw& second = draws[index + 1];
        const double x_gap = first.x - second.x;
        // z = 0 forces x = 0, and the pair's H1 vanishes whatever r is.
        const double first_ratio = first.z > 0.0 ? first.x / first.z : 0.0;
        const double second_ratio = second.z > 0.0 ? second.x / second.z : 0.0;
        const double r_gap = first_ratio - second_ratio;
        statistics.h0.push_back(0.5 * x_gap * x_gap);
        statistics.h1.push_back(0.5 * first.z * second.z * r_gap * r_gap);
    }
    return statistics;
}

}  // namespace

Coefficients EstimateCoefficients(const std::vector<PilotDraw>& draws) {
    const PairStatistics statistics = ComputePairStatistics(draws);
    const double mean_squared = statistics.mean_x * statistics.mean_x;
    return Coefficients{.lambda0 = Mean(statistics.h0) / mean_squared, .lambda1 = Mean(statistics.h1) / mean_squared};
}

std::optional<Coefficients> CertifyCoefficients(const std::vector<PilotDraw>& draws, double alpha) {
    const PairStatistics statistics = ComputePairStatistics(draws);
    if (statistics.h0.size() < 2) {
        throw std::invalid_argument("Certifying the pool coefficients needs at least four pilot draws.");
    }
    std::vector<double> ratios(draws.size());
    std::ranges::transform(draws, ratios.begin(), &PilotDraw::x);
    const double level = alpha / 3.0;
    const double mean_lower = statistics.mean_x - EmpiricalBernsteinRadius(ratios, 1.0, level);
    if (!(mean_lower > 0.0)) {
        return std::nullopt;
    }
    // H0 and H1 lie in [0, 1/2] because x, z and r lie in (0, 1].
    const double h0_upper = std::min(0.5, Mean(statistics.h0) + EmpiricalBernsteinRadius(statistics.h0, 0.5, level));
    const double h1_upper = std::min(0.5, Mean(statistics.h1) + EmpiricalBernsteinRadius(statistics.h1, 0.5, level));
    const double mean_squared = mean_lower * mean_lower;
    return Coefficients{.lambda0 = h0_upper / mean_squared, .lambda1 = h1_upper / mean_squared};
}

double EmpiricalBernsteinRadius(const std::vector<double>& values, double range, double failure_probability) {
    if (values.size() < 2) {
        throw std::invalid_argument("The empirical Bernstein bound needs at least two values.");
    }
    const auto count = static_cast<double>(values.size());
    const double mean = Mean(values);
    double sum_of_squares = 0.0;
    for (const double value : values) {
        sum_of_squares += (value - mean) * (value - mean);
    }
    const double sample_variance = sum_of_squares / (count - 1.0);
    const double log_term = std::log(4.0 / failure_probability);
    return std::sqrt(2.0 * sample_variance * log_term / count) + 7.0 * range * log_term / (3.0 * (count - 1.0));
}

std::optional<unsigned int> SamplesForPool(Coefficients coefficients, double eta, unsigned int pool_size) {
    const auto draws = static_cast<double>(pool_size);
    // K (eta - lambda0 / K): what the second stage may still add, scaled by K.
    const double slack = eta * draws - coefficients.lambda0;
    if (!(slack > 0.0)) {
        return std::nullopt;
    }
    return CeilCount(coefficients.lambda1 * (draws - 1.0) / slack);
}

unsigned int SquareRootPoolSize(Coefficients coefficients, double eta, double qalsh_cost, double exact_cost) {
    const double product = coefficients.lambda0 * coefficients.lambda1;
    return CeilCount((coefficients.lambda0 + std::sqrt(product * exact_cost / qalsh_cost)) / eta);
}

unsigned int OneStageSamples(double lambda, double eta) { return CeilCount(lambda / eta); }

}  // namespace PoolPlan
