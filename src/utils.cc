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

namespace {

constexpr size_t kDistanceBlockSize = 16;

template <bool Squared>
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
#if defined(__GNUC__) || defined(__clang__)
__attribute__((always_inline))
#endif
inline float AccumulateScalar(const Coordinate* lhs, const Coordinate* rhs, size_t dimensions, float bound) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    float sum = 0.0F;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (size_t i = 0; i < dimensions; ++i) {
        const float difference = lhs[i] - rhs[i];
        if constexpr (Squared) {
            sum += difference * difference;
        } else {
            sum += std::abs(difference);
        }
        if (sum > bound) {
            return std::numeric_limits<float>::infinity();
        }
    }
    return sum;
}

// Distances only accumulate nonnegative terms. Checking a small block at a
// time removes a branch per coordinate without changing the addition order.
// Keep the scalar fallback for NaNs: an earlier term might already have
// crossed the bound before a later NaN entered the same block.
template <bool Squared>
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
float AccumulateBounded(const Coordinate* lhs, const Coordinate* rhs, size_t dimensions, float bound) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    float sum = 0.0F;
    for (size_t begin = 0; begin < dimensions; begin += kDistanceBlockSize) {
        const size_t end = std::min(begin + kDistanceBlockSize, dimensions);
        // Preserve separately rounded float products and ordered additions,
        // including when the compiler vectorizes products or unrolls tails.
        for (size_t i = begin; i < end; ++i) {
            const float difference = lhs[i] - rhs[i];
            if constexpr (Squared) {
                sum += difference * difference;
            } else {
                sum += std::abs(difference);
            }
        }
        if (sum > bound) {
            return std::numeric_limits<float>::infinity();
        }
        if (std::isnan(sum)) {
            return AccumulateScalar<Squared>(lhs, rhs, dimensions, bound);
        }
    }
    return sum;
}

}  // namespace

Distance Calc::LpDistance(const Point &pt1, const Point &pt2, float norm_order) {
    Eigen::Map<const Eigen::VectorXf> v1(pt1.data(), static_cast<Eigen::Index>(pt1.size()));
    Eigen::Map<const Eigen::VectorXf> v2(pt2.data(), static_cast<Eigen::Index>(pt2.size()));

    if (std::abs(norm_order - 1.0F) < Global::kEpsilon) {
        return static_cast<Distance>((v1 - v2).lpNorm<1>());
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    if (std::abs(norm_order - 2.0F) < Global::kEpsilon) {
        return static_cast<Distance>((v1 - v2).lpNorm<2>());
    }
    Log::error("Unsupported norm order: {}", norm_order);
    return 0.0F;
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
Distance Calc::LpDistanceBounded(const Point &pt1, const Point &pt2, float norm_order, Distance upper_bound) {
    if (std::isinf(upper_bound)) {
        return LpDistance(pt1, pt2, norm_order);
    }
    // Keep the short-vector path in this entry point: forwarding each tiny
    // distance through another out-of-line function costs more than its math.
    if (pt1.size() <= kDistanceBlockSize) {
        if (upper_bound < 0.0F) {
            return std::numeric_limits<float>::infinity();
        }
        if (std::abs(norm_order - 1.0F) < Global::kEpsilon) {
            return AccumulateScalar<false>(pt1.data(), pt2.data(), pt1.size(), upper_bound);
        }
        if (std::abs(norm_order - 2.0F) < Global::kEpsilon) {
            return std::sqrt(AccumulateScalar<true>(pt1.data(), pt2.data(), pt1.size(),
                                                   upper_bound * upper_bound));
        }
    }
    return LpDistanceBoundedRaw(pt1.data(), pt2.data(), pt1.size(), norm_order, upper_bound);
}

// The original bounded loop rounds the square before adding it on the
// supported GCC build. Keep that behavior when loops are inlined/unrolled.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
Distance Calc::LpDistanceBoundedRaw(const Coordinate* pt1, const Coordinate* pt2, size_t num_dimensions,
                                    float norm_order, Distance upper_bound) {
    if (upper_bound < 0.0F) {
        return std::numeric_limits<float>::infinity();
    }

    if (std::abs(norm_order - 1.0F) < Global::kEpsilon) {
        // Short vectors benefit from the original early bound checks and
        // avoid the extra call/loop overhead of block accumulation.
        if (num_dimensions <= kDistanceBlockSize) {
            return AccumulateScalar<false>(pt1, pt2, num_dimensions, upper_bound);
        }
        return AccumulateBounded<false>(pt1, pt2, num_dimensions, upper_bound);
    }

    // NOLINTNEXTLINE(readability-magic-numbers)
    if (std::abs(norm_order - 2.0F) < Global::kEpsilon) {
        const float upper_bound_sq = upper_bound * upper_bound;
        const float sum_sq = num_dimensions <= kDistanceBlockSize
                                 ? AccumulateScalar<true>(pt1, pt2, num_dimensions, upper_bound_sq)
                                 : AccumulateBounded<true>(pt1, pt2, num_dimensions, upper_bound_sq);
        return std::sqrt(sum_sq);
    }

    Log::error("Unsupported norm order: {}", norm_order);
    return 0.0F;
}

Projection Calc::DotProduct(const Point &pt1, const Point &pt2) {
    Eigen::Map<const Eigen::VectorXf> v1(pt1.data(), static_cast<Eigen::Index>(pt1.size()));
    Eigen::Map<const Eigen::VectorXf> v2(pt2.data(), static_cast<Eigen::Index>(pt2.size()));

    return v1.dot(v2);
}

float Calc::L1Probability(float x) { return 2.0F / std::numbers::pi_v<float> * std::atan(x); }  // 2 / pi * atan(x)

float Calc::L2Probability(float x) {
    constexpr float kInverseSqrt2 = 0.7071067811865476F;
    return std::erf(x * kInverseSqrt2);
}  // 2 * Phi(x) - 1

std::mt19937 Random::SeededGenerator() {
    if (Global::kUseFixedSeed) {
        return std::mt19937(Global::kDefaultSeed);
    }
    return std::mt19937(std::random_device{}());
}

float Memory::Usage() {
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
    return static_cast<float>(memory_usage) / 1024.0F;  // NOLINT: readability-magic-numbers
}

unsigned int Sample::FromWeights(const std::vector<Weight> &weights) {
    Weight total_sum = std::accumulate(weights.begin(), weights.end(), 0.0F);
    if (total_sum <= 0) {
        throw std::runtime_error("Total sum of weights must be positive.");
    }

    std::vector<Weight> cumulative_weights;
    cumulative_weights.reserve(weights.size());
    std::partial_sum(weights.begin(), weights.end(), std::back_inserter(cumulative_weights));

    static std::mt19937 gen = Random::SeededGenerator();
    std::uniform_real_distribution<Weight> dis(0.0F, total_sum);
    Weight random_value = dis(gen);

    auto it = std::ranges::upper_bound(cumulative_weights, random_value);

    return static_cast<unsigned int>(std::distance(cumulative_weights.begin(), it));
}

WeightedSampler::WeightedSampler(const std::vector<Weight> &weights) : gen_(Random::SeededGenerator()) {
    total_ = std::accumulate(weights.begin(), weights.end(), 0.0F);
    if (total_ <= 0) {
        throw std::runtime_error("Total sum of weights must be positive.");
    }

    cumulative_weights_.reserve(weights.size());
    std::partial_sum(weights.begin(), weights.end(), std::back_inserter(cumulative_weights_));
}

unsigned int WeightedSampler::Sample() {
    std::uniform_real_distribution<Weight> dis(0.0F, total_);
    const Weight random_value = dis(gen_);
    auto it = std::ranges::upper_bound(cumulative_weights_, random_value);
    if (it == cumulative_weights_.end()) {
        return static_cast<unsigned int>(cumulative_weights_.size() - 1);
    }
    return static_cast<unsigned int>(std::distance(cumulative_weights_.begin(), it));
}

Weight WeightedSampler::Total() const {
    return total_;
}
