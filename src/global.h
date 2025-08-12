#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <cmath>
#include <numbers>

class Global {
   public:
    static constexpr double kEpsilon = 1e-6;
    static constexpr unsigned int kDefaultPageSize = 4096;
    static constexpr double kQalshDefaultErrorProbability = 1.0 / std::numbers::e_v<double>;
    static constexpr double kSamplingDefaultErrorProbability = 0.1;
    static constexpr double kDefaultApproximationRatio = 2.0;
    static constexpr unsigned int kNumCandidates = 100;
    static constexpr unsigned int kScanSize = 128;

    static bool kUseFixedSeed;
    static constexpr unsigned int kDefaultSeed = 42;
};

#endif