#pragma once

#include <spdlog/spdlog.h>

#include <Eigen/Eigen>
#include <random>

#include "point_set.h"

class Calc {
   public:
    static double LpDistance(const Point &pt1, const Point &pt2, double norm_order);
    static double DotProduct(const Point &pt1, const Point &pt2);
    static double L1Probability(double x);
    static double L2Probability(double x);
};

class Random {
   public:
    static std::mt19937 SeededGenerator();
};

class Memory {
   public:
    static double Usage();
};

class Sample {
   public:
    static unsigned int FromWeights(const std::vector<double> &weights);
};
