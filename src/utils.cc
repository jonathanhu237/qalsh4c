#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#include "global.h"
#include "log.h"

double Calc::LpDistance(const Point &pt1, const Point &pt2, double norm_order) {
    Eigen::Map<const Eigen::VectorXd> v1(pt1.data(), static_cast<Eigen::Index>(pt1.size()));
    Eigen::Map<const Eigen::VectorXd> v2(pt2.data(), static_cast<Eigen::Index>(pt2.size()));

    if (std::abs(norm_order - 1.0) < Global::kEpsilon) {
        return static_cast<double>((v1 - v2).lpNorm<1>());
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    if (std::abs(norm_order - 2.0) < Global::kEpsilon) {
        return static_cast<double>((v1 - v2).lpNorm<2>());
    }
    Log::error("Unsupported norm order: {}", norm_order);
    return 0.0;
}

double Calc::LpDistanceBounded(const Point &pt1, const Point &pt2, double norm_order, double upper_bound) {
    if (upper_bound < 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    if (std::abs(norm_order - 1.0) < Global::kEpsilon) {
        double sum = 0.0;
        for (size_t i = 0; i < pt1.size(); ++i) {
            sum += std::abs(pt1[i] - pt2[i]);
            if (sum > upper_bound) {
                return std::numeric_limits<double>::infinity();
            }
        }
        return sum;
    }

    // NOLINTNEXTLINE(readability-magic-numbers)
    if (std::abs(norm_order - 2.0) < Global::kEpsilon) {
        const double upper_bound_sq = upper_bound * upper_bound;
        double sum_sq = 0.0;
        for (size_t i = 0; i < pt1.size(); ++i) {
            const double diff = pt1[i] - pt2[i];
            sum_sq += diff * diff;
            if (sum_sq > upper_bound_sq) {
                return std::numeric_limits<double>::infinity();
            }
        }
        return std::sqrt(sum_sq);
    }

    Log::error("Unsupported norm order: {}", norm_order);
    return 0.0;
}

double Calc::DotProduct(const Point &pt1, const Point &pt2) {
    Eigen::Map<const Eigen::VectorXd> v1(pt1.data(), static_cast<Eigen::Index>(pt1.size()));
    Eigen::Map<const Eigen::VectorXd> v2(pt2.data(), static_cast<Eigen::Index>(pt2.size()));

    return v1.dot(v2);
}

double Calc::L1Probability(double x) { return 2.0 / std::numbers::pi_v<double> * atan(x); }  // 2 / pi * atan(x)

double Calc::L2Probability(double x) { return std::erf(x * M_SQRT1_2); }  // 2 * Phi(x) - 1

std::mt19937 Random::SeededGenerator() {
    if (Global::kUseFixedSeed) {
        return std::mt19937(Global::kDefaultSeed);
    }
    return std::mt19937(std::random_device{}());
}

double Memory::Usage() {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    size_t memory_usage = 0;

    while (std::getline(status_file, line)) {
        // NOLINTNEXTLINE: readability-magic-numbers
        if (line.substr(0, 6) == "VmHWM:") {
            std::istringstream iss(line);
            std::string key;
            iss >> key >> memory_usage;
            break;
        }
    }

    // Convert from KB to MB
    return static_cast<double>(memory_usage) / 1024.0;  // NOLINT: readability-magic-numbers
}

unsigned int Sample::FromWeights(const std::vector<double> &weights) {
    double total_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total_sum <= 0) {
        throw std::runtime_error("Total sum of weights must be positive.");
    }

    std::vector<double> cumulative_weights;
    cumulative_weights.reserve(weights.size());
    std::partial_sum(weights.begin(), weights.end(), std::back_inserter(cumulative_weights));

    static std::mt19937 gen = Random::SeededGenerator();
    std::uniform_real_distribution<> dis(0.0, total_sum);
    double random_value = dis(gen);

    auto it = std::ranges::upper_bound(cumulative_weights, random_value);

    return static_cast<unsigned int>(std::distance(cumulative_weights.begin(), it));
}

WeightedSampler::WeightedSampler(const std::vector<double> &weights) : gen_(Random::SeededGenerator()) {
    total_ = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total_ <= 0) {
        throw std::runtime_error("Total sum of weights must be positive.");
    }

    cumulative_weights_.reserve(weights.size());
    std::partial_sum(weights.begin(), weights.end(), std::back_inserter(cumulative_weights_));
}

unsigned int WeightedSampler::Sample() {
    std::uniform_real_distribution<double> dis(0.0, total_);
    const double random_value = dis(gen_);
    auto it = std::ranges::upper_bound(cumulative_weights_, random_value);
    if (it == cumulative_weights_.end()) {
        return static_cast<unsigned int>(cumulative_weights_.size() - 1);
    }
    return static_cast<unsigned int>(std::distance(cumulative_weights_.begin(), it));
}

double WeightedSampler::Total() const {
    return total_;
}
