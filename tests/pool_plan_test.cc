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
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
