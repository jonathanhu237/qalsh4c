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

Coefficients RelativeStandardErrors(const std::vector<PilotDraw>& draws) {
    const PairStatistics statistics = ComputePairStatistics(draws);
    const size_t num_pairs = statistics.h0.size();
    if (num_pairs < 2) {
        return Coefficients{.lambda0 = 0.0, .lambda1 = 0.0};
    }
    std::vector<double> pair_means(num_pairs);
    for (size_t pair = 0; pair < num_pairs; ++pair) {
        pair_means[pair] = 0.5 * (draws[2 * pair].x + draws[2 * pair + 1].x);
    }
    const double mean_ratio = Mean(pair_means);
    const auto relative_error = [&](const std::vector<double>& values) {
        const double mean = Mean(values);
        if (!(mean > 0.0)) {
            return 0.0;
        }
        // Linearization of mean(H) / mean(P)^2 around the pair means.
        std::vector<double> influence(num_pairs);
        for (size_t pair = 0; pair < num_pairs; ++pair) {
            influence[pair] = (values[pair] - mean) / mean - 2.0 * (pair_means[pair] - mean_ratio) / mean_ratio;
        }
        const double influence_mean = Mean(influence);
        double sum_of_squares = 0.0;
        for (const double value : influence) {
            sum_of_squares += (value - influence_mean) * (value - influence_mean);
        }
        const auto count = static_cast<double>(num_pairs);
        return std::sqrt(sum_of_squares / (count - 1.0) / count);
    };
    return Coefficients{.lambda0 = relative_error(statistics.h0), .lambda1 = relative_error(statistics.h1)};
}

std::optional<Coefficients> CertifyCoefficients(const std::vector<PilotDraw>& draws, double alpha) {
    const std::optional<StratumMoments> moments = CertifyStratumMoments(draws, alpha);
    if (!moments) {
        return std::nullopt;
    }
    const double mean_squared = moments->mean_ratio * moments->mean_ratio;
    return Coefficients{.lambda0 = moments->h0 / mean_squared, .lambda1 = moments->h1 / mean_squared};
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

ConfidenceSequence::ConfidenceSequence(double delta) : log_term_(std::log(2.0 / delta)) {
    if (!(delta > 0.0 && delta < 1.0)) {
        throw std::invalid_argument("A confidence sequence needs a failure probability in (0, 1).");
    }
}

void ConfidenceSequence::Add(double value) {
    ++count_;
    const auto t = static_cast<double>(count_);
    // The bet depends only on earlier values: lambda_t = sqrt(2 log(2 / delta) / (sigma^2_{t-1} t log(1 + t))), capped.
    const double bet = std::min(std::sqrt(2.0 * log_term_ / (previous_variance_ * t * std::log1p(t))), kMaxBet);
    const double deviation = value - previous_mean_;
    // v_t psi_E(lambda_t) with v_t = 4 (x_t - mu_hat_{t-1})^2 and psi_E(lambda) = (-log(1 - lambda) - lambda) / 4.
    sum_penalties_ += deviation * deviation * (-std::log1p(-bet) - bet);
    sum_bets_ += bet;
    sum_weighted_values_ += bet * value;
    sum_ += value;
    const double mean = (0.5 + sum_) / (t + 1.0);
    sum_squared_deviations_ += (value - mean) * (value - mean);
    previous_mean_ = mean;
    previous_variance_ = (0.25 + sum_squared_deviations_) / (t + 1.0);
    const double center = sum_weighted_values_ / sum_bets_;
    const double radius = (log_term_ + sum_penalties_) / sum_bets_;
    lower_ = std::max(lower_, center - radius);
    upper_ = std::min(upper_, center + radius);
}

double ConfidenceSequence::Mean() const { return count_ == 0 ? 0.0 : sum_ / static_cast<double>(count_); }

bool ConfidenceSequence::CertifiesMean(double epsilon) const {
    const double mean = Mean();
    return lower_ > 0.0 && mean - lower_ <= epsilon * lower_ && upper_ - mean <= epsilon * upper_;
}

StratumMoments EstimateStratumMoments(const std::vector<PilotDraw>& draws) {
    const PairStatistics statistics = ComputePairStatistics(draws);
    return StratumMoments{.mean_ratio = statistics.mean_x, .h0 = Mean(statistics.h0), .h1 = Mean(statistics.h1)};
}

std::optional<StratumMoments> CertifyStratumMoments(const std::vector<PilotDraw>& draws, double alpha) {
    const PairStatistics statistics = ComputePairStatistics(draws);
    if (statistics.h0.size() < 2) {
        throw std::invalid_argument("Certification needs at least four pilot draws.");
    }
    std::vector<double> ratios(draws.size());
    std::ranges::transform(draws, ratios.begin(), &PilotDraw::x);
    const double level = alpha / 3.0;
    const double mean_lower = statistics.mean_x - EmpiricalBernsteinRadius(ratios, 1.0, level);
    if (!(mean_lower > 0.0)) {
        return std::nullopt;
    }
    // H0 and H1 lie in [0, 1/2] because x, z and r lie in (0, 1].
    return StratumMoments{
        .mean_ratio = mean_lower,
        .h0 = std::min(0.5, Mean(statistics.h0) + EmpiricalBernsteinRadius(statistics.h0, 0.5, level)),
        .h1 = std::min(0.5, Mean(statistics.h1) + EmpiricalBernsteinRadius(statistics.h1, 0.5, level)),
    };
}

Coefficients PooledCoefficients(const std::vector<std::vector<PilotDraw>>& strata, const std::vector<double>& shares) {
    double mean_x = 0.0;
    double mean_x2 = 0.0;
    double mean_z = 0.0;
    double mean_x2_over_z = 0.0;
    for (size_t stratum = 0; stratum < strata.size(); ++stratum) {
        const double weight = shares[stratum] / static_cast<double>(strata[stratum].size());
        for (const PilotDraw& draw : strata[stratum]) {
            mean_x += weight * draw.x;
            mean_x2 += weight * draw.x * draw.x;
            mean_z += weight * draw.z;
            // z = 0 forces x = 0, which contributes nothing.
            mean_x2_over_z += draw.z > 0.0 ? weight * draw.x * draw.x / draw.z : 0.0;
        }
    }
    if (!(mean_x > 0.0)) {
        return Coefficients{.lambda0 = 0.0, .lambda1 = 0.0};
    }
    const double mean_squared = mean_x * mean_x;
    return Coefficients{.lambda0 = std::max(0.0, mean_x2 / mean_squared - 1.0),
                        .lambda1 = std::max(0.0, mean_z * mean_x2_over_z / mean_squared - 1.0)};
}

std::vector<unsigned int> StratifiedSamples(const std::vector<double>& sigmas, double eta, double total) {
    const double sigma_sum = std::accumulate(sigmas.begin(), sigmas.end(), 0.0);
    std::vector<unsigned int> samples(sigmas.size());
    for (size_t stratum = 0; stratum < sigmas.size(); ++stratum) {
        samples[stratum] = CeilCount(sigmas[stratum] * sigma_sum / (eta * total * total));
    }
    return samples;
}

}  // namespace PoolPlan
