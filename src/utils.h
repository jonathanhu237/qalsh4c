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

class Sample {
   public:
    static unsigned int FromWeights(const std::vector<Weight> &weights);
};

class WeightedSampler {
   public:
    explicit WeightedSampler(const std::vector<Weight> &weights);
    unsigned int Sample();
    Weight Total() const;

   private:
    std::vector<Weight> cumulative_weights_;
    Weight total_{0.0};
    std::mt19937 gen_;
};
