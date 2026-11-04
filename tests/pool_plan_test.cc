// The pilot statistics and budgets of pool_plan.h: the pair identities and the pooled variance identity by exact
// enumeration over small populations, and the budgets against their defining conditions.
#include "pool_plan.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(bool value, const std::string& message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

void CheckClose(double actual, double expected, double tolerance, const std::string& message) {
    Check(std::abs(actual - expected) <= tolerance * std::max(1.0, std::abs(expected)),
          message + ": " + std::to_string(actual) + " instead of " + std::to_string(expected));
}

// A finite population of source points with 0 < v_a <= D_a <= U_a.
struct Population {
    std::vector<double> upper;  // U_a
    std::vector<double> crude;  // D_a
    std::vector<double> exact;  // v_a
};

Population RandomPopulation(std::mt19937& gen, unsigned int size) {
    std::uniform_real_distribution<double> bound(0.5, 3.0);
    std::uniform_real_distribution<double> fraction(0.05, 1.0);
    Population population;
    for (unsigned int point = 0; point < size; ++point) {
        const double upper = bound(gen);
        const double crude = upper * fraction(gen);
        population.upper.push_back(upper);
        population.crude.push_back(crude);
        population.exact.push_back(crude * fraction(gen));
    }
    return population;
}

struct Truth {
    double total;  // S
    double mu;     // S / U
    PoolPlan::Coefficients coefficients;
    std::vector<double> probability;  // p_a = U_a / U
};

// lambda0 and lambda1 straight from their definitions.
Truth TrueCoefficients(const Population& population) {
    double total_upper = 0.0;
    double total = 0.0;
    for (size_t point = 0; point < population.upper.size(); ++point) {
        total_upper += population.upper[point];
        total += population.exact[point];
    }
    Truth truth{.total = total, .mu = total / total_upper, .coefficients = {}, .probability = {}};
    double second_moment = 0.0;  // E_p[x^2]
    double mean_z = 0.0;         // E_p[z]
    double moment_over_z = 0.0;  // E_p[x^2 / z]
    for (size_t point = 0; point < population.upper.size(); ++point) {
        const double probability = population.upper[point] / total_upper;
        const double x = population.exact[point] / population.upper[point];
        const double z = population.crude[point] / population.upper[point];
        truth.probability.push_back(probability);
        second_moment += probability * x * x;
        mean_z += probability * z;
        moment_over_z += probability * x * x / z;
    }
    const double mu_squared = truth.mu * truth.mu;
    truth.coefficients = PoolPlan::Coefficients{.lambda0 = (second_moment - mu_squared) / mu_squared,
                                                .lambda1 = (mean_z * moment_over_z - mu_squared) / mu_squared};
    return truth;
}

// E[H0] = mu^2 lambda0 and E[H1] = mu^2 lambda1 over independent pairs of draws in proportion to U.
void CheckPairIdentities(const Population& population) {
    const Truth truth = TrueCoefficients(population);
    double h0 = 0.0;
    double h1 = 0.0;
    for (size_t first = 0; first < population.upper.size(); ++first) {
        for (size_t second = 0; second < population.upper.size(); ++second) {
            const double weight = truth.probability[first] * truth.probability[second];
            const double x_gap = population.exact[first] / population.upper[first] -
                                 population.exact[second] / population.upper[second];
            const double r_gap = population.exact[first] / population.crude[first] -
                                 population.exact[second] / population.crude[second];
            const double z_product = population.crude[first] / population.upper[first] * population.crude[second] /
                                     population.upper[second];
            h0 += weight * 0.5 * x_gap * x_gap;
            h1 += weight * 0.5 * z_product * r_gap * r_gap;
        }
    }
    const double mu_squared = truth.mu * truth.mu;
    CheckClose(h0, mu_squared * truth.coefficients.lambda0, 1e-12, "E[H0]");
    CheckClose(h1, mu_squared * truth.coefficients.lambda1, 1e-12, "E[H1]");
}

// The pooled estimate (U / K) sum_j z_j * (1 / T) sum_t r_t over every pool of K draws and every T resamples is
// unbiased, with relative variance lambda0 / K + (1 - 1 / K) lambda1 / T.
void CheckVarianceIdentity(const Population& population, unsigned int pool_size, unsigned int num_samples) {
    const Truth truth = TrueCoefficients(population);
    const auto num_points = static_cast<unsigned int>(population.upper.size());
    const double total_upper = truth.total / truth.mu;
    double mean = 0.0;
    double second_moment = 0.0;
    std::vector<unsigned int> pool(pool_size, 0);
    while (true) {
        double pool_probability = 1.0;
        double z_sum = 0.0;
        for (const unsigned int point : pool) {
            pool_probability *= truth.probability[point];
            z_sum += population.crude[point] / population.upper[point];
        }
        const double crude_total = total_upper / pool_size * z_sum;
        std::vector<unsigned int> picks(num_samples, 0);
        while (true) {
            double pick_probability = pool_probability;
            double r_sum = 0.0;
            for (const unsigned int pick : picks) {
                const unsigned int point = pool[pick];
                pick_probability *= population.crude[point] / population.upper[point] / z_sum;
                r_sum += population.exact[point] / population.crude[point];
            }
            const double estimate = crude_total * r_sum / num_samples;
            mean += pick_probability * estimate;
            second_moment += pick_probability * estimate * estimate;
            unsigned int position = 0;
            while (position < num_samples && ++picks[position] == pool_size) {
                picks[position++] = 0;
            }
            if (position == num_samples) {
                break;
            }
        }
        unsigned int position = 0;
        while (position < pool_size && ++pool[position] == num_points) {
            pool[position++] = 0;
        }
        if (position == pool_size) {
            break;
        }
    }
    const double relative_variance = (second_moment - mean * mean) / (truth.total * truth.total);
    const double expected = truth.coefficients.lambda0 / pool_size +
                            (1.0 - 1.0 / pool_size) * truth.coefficients.lambda1 / num_samples;
    CheckClose(mean, truth.total, 1e-12, "pooled mean");
    CheckClose(relative_variance, expected, 1e-9, "pooled relative variance");
}

double PooledVariance(PoolPlan::Coefficients coefficients, unsigned int pool_size, unsigned int num_samples) {
    return coefficients.lambda0 / pool_size + (1.0 - 1.0 / pool_size) * coefficients.lambda1 / num_samples;
}

}  // namespace

int main() {
    try {
        std::mt19937 gen(11);

        for (int trial = 0; trial < 50; ++trial) {
            const Population population = RandomPopulation(gen, 5);
            CheckPairIdentities(population);
            CheckVarianceIdentity(population, 3, 2);
        }
        CheckVarianceIdentity(RandomPopulation(gen, 3), 4, 3);
        CheckVarianceIdentity(RandomPopulation(gen, 4), 1, 2);

        // Point estimates from hand-computed pairs: (0.5, 0.25) differ in x only, and (1, 1) do not differ.
        {
            const std::vector<PoolPlan::PilotDraw> draws{{.x = 0.5, .z = 1.0}, {.x = 0.25, .z = 0.5},
                                                         {.x = 1.0, .z = 1.0}, {.x = 1.0, .z = 1.0}};
            const PoolPlan::Coefficients estimate = PoolPlan::EstimateCoefficients(draws);
            const double mean = 0.6875;
            CheckClose(estimate.lambda0, (0.5 * 0.25 * 0.25 / 2.0) / (mean * mean), 1e-12, "lambda0 estimate");
            CheckClose(estimate.lambda1, 0.0, 1e-12, "lambda1 estimate");
        }

        // Delta-method relative standard errors: the pair (x, z) = (1, 1), (0.5, 1) gives H0 = H1 = 1/8 and a pair of
        // equal draws gives zero, so two pairs of each kind have mean 1/16, and the pair means of x vary as well.
        {
            const std::vector<PoolPlan::PilotDraw> draws{{.x = 1.0, .z = 1.0}, {.x = 0.5, .z = 1.0},
                                                         {.x = 0.7, .z = 1.0}, {.x = 0.7, .z = 1.0},
                                                         {.x = 1.0, .z = 1.0}, {.x = 0.5, .z = 1.0},
                                                         {.x = 0.3, .z = 1.0}, {.x = 0.3, .z = 1.0}};
            const std::vector<double> pairs{0.125, 0.0, 0.125, 0.0};
            const std::vector<double> pair_means{0.75, 0.7, 0.75, 0.3};
            const double mean = 0.0625;
            const double mean_ratio = 0.625;
            std::vector<double> influence;
            for (size_t pair = 0; pair < pairs.size(); ++pair) {
                influence.push_back((pairs[pair] - mean) / mean - 2.0 * (pair_means[pair] - mean_ratio) / mean_ratio);
            }
            double influence_mean = 0.0;
            for (const double value : influence) {
                influence_mean += value / static_cast<double>(influence.size());
            }
            double sum_of_squares = 0.0;
            for (const double value : influence) {
                sum_of_squares += (value - influence_mean) * (value - influence_mean);
            }
            const double expected = std::sqrt(sum_of_squares / 3.0 / 4.0);
            const PoolPlan::Coefficients errors = PoolPlan::RelativeStandardErrors(draws);
            CheckClose(errors.lambda0, expected, 1e-12, "relative standard error of H0");
            CheckClose(errors.lambda1, expected, 1e-12, "relative standard error of H1");
            const std::vector<PoolPlan::PilotDraw> equal(6, PoolPlan::PilotDraw{.x = 0.4, .z = 0.8});
            Check(PoolPlan::RelativeStandardErrors(equal).lambda0 == 0.0, "equal draws have no relative error");
        }

        // The radius of Maurer and Pontil with both tails: values {0, 1} at level 0.1.
        {
            const double log_term = std::log(40.0);
            CheckClose(PoolPlan::EmpiricalBernsteinRadius({0.0, 1.0}, 1.0, 0.1),
                       std::sqrt(2.0 * 0.5 * log_term / 2.0) + 7.0 * log_term / 3.0, 1e-12, "Bernstein radius");
        }

        // Four draws cannot certify anything; a long constant pilot certifies exactly the radius terms.
        {
            const std::vector<PoolPlan::PilotDraw> few(4, PoolPlan::PilotDraw{.x = 0.5, .z = 1.0});
            Check(!PoolPlan::CertifyCoefficients(few, 0.05).has_value(), "four draws should not certify");
            const std::vector<PoolPlan::PilotDraw> constant(1000, PoolPlan::PilotDraw{.x = 0.5, .z = 1.0});
            const std::optional<PoolPlan::Coefficients> bounds = PoolPlan::CertifyCoefficients(constant, 0.05);
            Check(bounds.has_value(), "a long constant pilot should certify");
            const double log_term = std::log(4.0 / (0.05 / 3.0));
            const double mean_lower = 0.5 - 7.0 * log_term / (3.0 * 999.0);
            const double pair_upper = 7.0 * 0.5 * log_term / (3.0 * 499.0);
            CheckClose(bounds->lambda0, pair_upper / (mean_lower * mean_lower), 1e-12, "certified lambda0");
            CheckClose(bounds->lambda1, pair_upper / (mean_lower * mean_lower), 1e-12, "certified lambda1");
        }

        // Certified bounds cover the true coefficients at least as often as promised.
        {
            const Population population = RandomPopulation(gen, 40);
            const Truth truth = TrueCoefficients(population);
            std::discrete_distribution<unsigned int> draw(truth.probability.begin(), truth.probability.end());
            constexpr int kTrials = 200;
            constexpr double kAlpha = 0.2;
            int covered = 0;
            for (int trial = 0; trial < kTrials; ++trial) {
                std::vector<PoolPlan::PilotDraw> pilot;
                for (int index = 0; index < 400; ++index) {
                    const unsigned int point = draw(gen);
                    pilot.push_back({.x = population.exact[point] / population.upper[point],
                                     .z = population.crude[point] / population.upper[point]});
                }
                const std::optional<PoolPlan::Coefficients> bounds = PoolPlan::CertifyCoefficients(pilot, kAlpha);
                if (bounds && bounds->lambda0 >= truth.coefficients.lambda0 &&
                    bounds->lambda1 >= truth.coefficients.lambda1) {
                    ++covered;
                }
            }
            Check(covered >= static_cast<int>((1.0 - kAlpha) * kTrials), "certified coverage " + std::to_string(covered));
        }

        // The confidence sequence follows the predictable plug-in empirical Bernstein recursion of Waudby-Smith and
        // Ramdas; the reference values come from an independent transcription of their formulas.
        {
            PoolPlan::ConfidenceSequence sequence(0.1);
            const std::vector<double> pattern{0.9, 0.7, 0.95, 0.8, 0.85, 1.0, 0.6, 0.75};
            for (int round = 0; round < 25; ++round) {
                for (const double value : pattern) {
                    sequence.Add(value);
                    if (sequence.Count() == 8) {
                        CheckClose(sequence.Lower(), 0.05414423002353441, 1e-12, "lower bound after 8 values");
                        CheckClose(sequence.Upper(), 1.0, 1e-12, "upper bound after 8 values");
                    }
                    if (sequence.Count() == 50) {
                        CheckClose(sequence.Lower(), 0.6904084897577099, 1e-12, "lower bound after 50 values");
                        CheckClose(sequence.Upper(), 0.9455915102422903, 1e-12, "upper bound after 50 values");
                    }
                }
            }
            CheckClose(sequence.Lower(), 0.7834565550492422, 1e-12, "lower bound after 200 values");
            CheckClose(sequence.Upper(), 0.8552291020634225, 1e-12, "upper bound after 200 values");
            CheckClose(sequence.Mean(), 0.81875, 1e-12, "sample mean after 200 values");

            // Bernstein's fixed-size budget from the sample mean and variance of the first eight values.
            PoolPlan::ConfidenceSequence first_round(0.1);
            for (const double value : pattern) {
                first_round.Add(value);
            }
            CheckClose(first_round.BernsteinBudget(0.01, 0.1), 1835.9719654094658, 1e-9, "Bernstein budget");
        }

        // Constant values certify their own mean, which is then exact.
        {
            PoolPlan::ConfidenceSequence sequence(0.1);
            while (!sequence.CertifiesMean(0.05) && sequence.Count() < 100000) {
                sequence.Add(1.0);
            }
            Check(sequence.CertifiesMean(0.05), "constant values should certify");
            Check(sequence.Mean() == 1.0 && sequence.Upper() == 1.0, "constant values keep an exact mean");
            Check(sequence.Lower() >= 1.0 / 1.05, "a certified lower bound is within epsilon of the mean");
        }

        // The sequence holds the true mean at every count at once at least as often as promised.
        {
            std::uniform_int_distribution<int> pick(0, 3);
            const std::vector<double> support{0.1, 0.5, 0.9, 1.0};
            const double mean = (0.1 + 0.5 + 0.9 + 1.0) / 4.0;
            constexpr int kTrials = 400;
            constexpr double kDelta = 0.2;
            int escaped = 0;
            for (int trial = 0; trial < kTrials; ++trial) {
                PoolPlan::ConfidenceSequence sequence(kDelta);
                bool left = false;
                for (int index = 0; index < 400 && !left; ++index) {
                    sequence.Add(support[static_cast<size_t>(pick(gen))]);
                    left = mean < sequence.Lower() || mean > sequence.Upper();
                }
                escaped += left ? 1 : 0;
            }
            Check(escaped <= static_cast<int>(kDelta * kTrials), "confidence sequence misses " + std::to_string(escaped));
        }

        // SamplesForPool returns the smallest T that meets the exact identity, and nothing when K is too small.
        {
            std::uniform_real_distribution<double> coefficient(0.0, 2.0);
            std::uniform_real_distribution<double> target(1e-4, 1e-2);
            std::uniform_int_distribution<unsigned int> pool(1, 5000);
            for (int trial = 0; trial < 1000; ++trial) {
                const PoolPlan::Coefficients coefficients{.lambda0 = coefficient(gen), .lambda1 = coefficient(gen)};
                const double eta = target(gen);
                const unsigned int pool_size = pool(gen);
                const std::optional<unsigned int> samples = PoolPlan::SamplesForPool(coefficients, eta, pool_size);
                if (eta * pool_size <= coefficients.lambda0) {
                    Check(!samples.has_value(), "a pool that alone exceeds eta needs no T");
                    continue;
                }
                if (coefficients.lambda1 * (pool_size - 1.0) / (eta * pool_size - coefficients.lambda0) > 1e8) {
                    continue;  // Budgets this large are clamped.
                }
                Check(samples.has_value(), "a feasible pool needs a T");
                Check(PooledVariance(coefficients, pool_size, *samples) <= eta * (1.0 + 1e-12), "T meets eta");
                Check(*samples == 1 || PooledVariance(coefficients, pool_size, *samples - 1) > eta, "T is minimal");
            }
        }

        // The square-root pool size, with T from SamplesForPool, costs at most a little more than the best pool.
        {
            std::uniform_real_distribution<double> coefficient(0.01, 2.0);
            std::uniform_real_distribution<double> cost(0.05, 5.0);
            for (int trial = 0; trial < 200; ++trial) {
                const PoolPlan::Coefficients coefficients{.lambda0 = coefficient(gen), .lambda1 = coefficient(gen)};
                const double eta = 1e-4;
                const double qalsh_cost = cost(gen);
                const double exact_cost = cost(gen);
                const auto total_cost = [&](unsigned int pool_size) {
                    const std::optional<unsigned int> samples = PoolPlan::SamplesForPool(coefficients, eta, pool_size);
                    return samples ? pool_size * qalsh_cost + *samples * exact_cost
                                   : std::numeric_limits<double>::infinity();
                };
                const unsigned int square_root_size =
                    PoolPlan::SquareRootPoolSize(coefficients, eta, qalsh_cost, exact_cost);
                double best = std::numeric_limits<double>::infinity();
                for (unsigned int pool_size = 1; pool_size <= 4 * square_root_size; ++pool_size) {
                    best = std::min(best, total_cost(pool_size));
                }
                Check(total_cost(square_root_size) <= 1.01 * best + qalsh_cost + exact_cost,
                      "square-root pool size is near the best");
            }
        }

        Check(PoolPlan::OneStageSamples(0.3, 0.01) == 30, "one-stage samples");
        Check(PoolPlan::OneStageSamples(0.0, 0.01) == 1, "one-stage samples need at least one draw");

        // Stratum moments are the unnormalized pair means behind the coefficient estimates.
        {
            const std::vector<PoolPlan::PilotDraw> draws{{.x = 0.5, .z = 1.0}, {.x = 0.25, .z = 0.5},
                                                         {.x = 1.0, .z = 1.0}, {.x = 0.6, .z = 0.8}};
            const PoolPlan::StratumMoments moments = PoolPlan::EstimateStratumMoments(draws);
            const PoolPlan::Coefficients coefficients = PoolPlan::EstimateCoefficients(draws);
            const double mean_squared = moments.mean_ratio * moments.mean_ratio;
            CheckClose(moments.mean_ratio, 0.5875, 1e-12, "stratum mean");
            CheckClose(moments.h0 / mean_squared, coefficients.lambda0, 1e-12, "stratum H0");
            CheckClose(moments.h1 / mean_squared, coefficients.lambda1, 1e-12, "stratum H1");
        }

        // Pooled coefficients from stratified pilots keep the differences between strata: two strata with constant
        // but different ratios have no variance inside either stratum, yet drawing over both does.
        {
            const std::vector<std::vector<PoolPlan::PilotDraw>> strata{
                std::vector<PoolPlan::PilotDraw>(8, PoolPlan::PilotDraw{.x = 0.2, .z = 0.5}),
                std::vector<PoolPlan::PilotDraw>(8, PoolPlan::PilotDraw{.x = 0.8, .z = 1.0})};
            const PoolPlan::Coefficients pooled = PoolPlan::PooledCoefficients(strata, {0.25, 0.75});
            const double mean = 0.25 * 0.2 + 0.75 * 0.8;
            CheckClose(pooled.lambda0, (0.25 * 0.04 + 0.75 * 0.64) / (mean * mean) - 1.0, 1e-12,
                       "pooled lambda0 across strata");
            CheckClose(pooled.lambda1,
                       (0.25 * 0.5 + 0.75 * 1.0) * (0.25 * 0.04 / 0.5 + 0.75 * 0.64) / (mean * mean) - 1.0, 1e-12,
                       "pooled lambda1 across strata");
            Check(pooled.lambda0 > 0.0, "constant strata with different means still vary");
            Check(PoolPlan::EstimateStratumMoments(strata[0]).h0 == 0.0, "a constant stratum has no variance");
        }

        // Neyman allocation meets the target and stays within one draw of sigma_h * sum_g sigma_g / (eta S^2).
        {
            std::uniform_real_distribution<double> sigma(0.0, 5.0);
            constexpr double kEta = 1e-4;
            constexpr double kTotal = 100.0;
            for (int trial = 0; trial < 200; ++trial) {
                std::vector<double> sigmas(1 + trial % 6);
                for (double& value : sigmas) {
                    value = sigma(gen);
                }
                const std::vector<unsigned int> samples = PoolPlan::StratifiedSamples(sigmas, kEta, kTotal);
                double sigma_sum = 0.0;
                double variance = 0.0;
                for (size_t stratum = 0; stratum < sigmas.size(); ++stratum) {
                    sigma_sum += sigmas[stratum];
                    variance += sigmas[stratum] * sigmas[stratum] / samples[stratum];
                }
                Check(variance <= kEta * kTotal * kTotal * (1.0 + 1e-12), "stratified draws meet eta");
                for (size_t stratum = 0; stratum < sigmas.size(); ++stratum) {
                    const double ideal = sigmas[stratum] * sigma_sum / (kEta * kTotal * kTotal);
                    Check(samples[stratum] >= 1 && samples[stratum] < std::max(1.0, ideal) + 1.0,
                          "Neyman allocation rounds up by less than one draw");
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
