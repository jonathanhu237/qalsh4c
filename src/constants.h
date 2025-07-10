#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#include <cmath>
#include <numbers>

class Constants {
   public:
    static constexpr double kEpsilon = 1e-6;
    static constexpr unsigned int kDefaultPageSize = 4096;

    static constexpr double kDefaultLeftBoundary = -128.0;                               // For dataset generation
    static constexpr double kDefaultRightBoundary = 128.0;                               // For dataset generation
    static constexpr double kDefaultErrorProbability = 1.0 / std::numbers::e_v<double>;  // For QALSH
};

#endif