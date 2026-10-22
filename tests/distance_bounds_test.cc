#include "utils.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

// Frozen scalar behavior from dd5138f. Its per-coordinate bound check keeps
// this reference independent of the optimized block implementation.
float Reference(const Point& a, const Point& b, float norm, float bound) {
    if (bound < 0.0F) return std::numeric_limits<float>::infinity();
    float sum = 0.0F;
    if (norm == 1.0F) {
        for (size_t i = 0; i < a.size(); ++i) {
            sum += std::abs(a[i] - b[i]);
            if (sum > bound) return std::numeric_limits<float>::infinity();
        }
        return sum;
    }
    const float limit = bound * bound;
    for (size_t i = 0; i < a.size(); ++i) {
        const float difference = a[i] - b[i];
        sum += difference * difference;
        if (sum > limit) return std::numeric_limits<float>::infinity();
    }
    return std::sqrt(sum);
}

void Compare(const Point& a, const Point& b, float norm, float bound) {
    const float expected = Reference(a, b, norm, bound);
    const float actual = Calc::LpDistanceBoundedRaw(a.data(), b.data(), a.size(), norm, bound);
    // The vector API deliberately uses Eigen for infinite bounds; finite
    // bounds (including its short-vector fast path) share the scalar contract.
    const float vector_actual = std::isinf(bound) ? actual : Calc::LpDistanceBounded(a, b, norm, bound);
    const auto equal = [expected](float value) {
        return (std::isnan(expected) && std::isnan(value)) ||
               std::bit_cast<std::uint32_t>(expected) == std::bit_cast<std::uint32_t>(value);
    };
    if (equal(actual) && equal(vector_actual)) return;
    std::cerr << "dimensions=" << a.size() << " norm=" << norm << " bound=" << bound
              << " expected=" << expected << " actual=" << actual << " vector=" << vector_actual << '\n';
    throw std::runtime_error("bounded distance differs from scalar reference");
}

void CheckPair(const Point& a, const Point& b) {
    constexpr float infinity = std::numeric_limits<float>::infinity();
    for (const float norm : {1.0F, 2.0F}) {
        const float distance = Reference(a, b, norm, infinity);
        for (const float bound : {-1.0F, 0.0F, infinity, distance, distance * 0.1F,
                                 std::nextafter(distance, 0.0F), std::nextafter(distance, infinity),
                                 std::numeric_limits<float>::quiet_NaN()}) {
            Compare(a, b, norm, bound);
        }
    }
}

}  // namespace

int main() {
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1000.0F, 1000.0F);
    for (const size_t dimensions : {0U, 1U, 7U, 15U, 16U, 17U, 63U, 64U, 65U, 127U, 784U, 4096U}) {
        Point a(dimensions), b(dimensions);
        CheckPair(a, b);
        for (unsigned repeat = 0; repeat < 100; ++repeat) {
            for (size_t i = 0; i < dimensions; ++i) {
                a[i] = distribution(generator);
                b[i] = distribution(generator);
            }
            CheckPair(a, b);
        }
    }
    CheckPair(Point(33, 1e20F), Point(33, -1e20F));
    CheckPair(Point(33, 1e-30F), Point(33, 0.0F));
    Point nonfinite(32, 0.0F), zeros(32, 0.0F);
    nonfinite[0] = 100.0F;
    nonfinite[7] = std::numeric_limits<float>::quiet_NaN();
    for (const float norm : {1.0F, 2.0F}) Compare(nonfinite, zeros, norm, 1.0F);
    CheckPair(nonfinite, zeros);
    nonfinite[0] = std::numeric_limits<float>::infinity();
    CheckPair(nonfinite, zeros);
}
