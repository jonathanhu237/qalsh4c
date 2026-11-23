#pragma once

#include <cstddef>
#include <Eigen/Eigen>
#include <random>
#include <vector>

#include "point_set.h"

class Calc {
   public:
    static Distance LpDistance(const Point &pt1, const Point &pt2, float norm_order);
    static Distance LpDistanceBounded(const Point &pt1, const Point &pt2, float norm_order, Distance upper_bound);
    static Distance LpDistanceBoundedRaw(const Coordinate* pt1, const Coordinate* pt2, size_t num_dimensions,
                                         float norm_order, Distance upper_bound);
    static Projection DotProduct(const Point &pt1, const Point &pt2);
    static float L1Probability(float x);
    static float L2Probability(float x);
};

class Random {
   public:
    static std::mt19937 SeededGenerator();
};

class Memory {
   public:
    static float Usage();
};

// Draws index i with probability weights[i] / Total(). The cumulative sums and the uniform draws are kept in double
// precision, so small weights keep their probabilities even when the total is many orders of magnitude larger.
class WeightedSampler {
   public:
    explicit WeightedSampler(const std::vector<Weight> &weights);
    unsigned int Sample();
    double Total() const;

   private:
    std::vector<double> cumulative_weights_;
    double total_{0.0};
    std::mt19937 gen_;
};
