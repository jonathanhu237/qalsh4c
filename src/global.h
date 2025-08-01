#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <cmath>
#include <numbers>

class Global {
   public:
    // Constants
    static constexpr double kEpsilon = 1e-6;
    static constexpr unsigned int kDefaultPageSize = 4096;
    static constexpr double kDefaultErrorProbability = 1.0 / std::numbers::e_v<double>;
    static constexpr double kDefaultApproximationRatio = 2.0;
    static constexpr unsigned int kNumCandidates = 100;

    // For dataset generation
    static constexpr unsigned int kDefaultNumPointsA = 1000;
    static constexpr unsigned int kDefaultNumPointsB = 1000;
    static constexpr unsigned int kDefaultNumDimensions = 256;
    static constexpr double kDefaultLeftBoundary = -1024;
    static constexpr double kDefaultRightBoundary = 1024;
};

#endif