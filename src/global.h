#pragma once

#include <cmath>
#include <numbers>

class Global {
   public:
    static constexpr double kEpsilon = 1e-6;
    static constexpr double kQalshDefaultErrorProbability = 1.0 / std::numbers::e_v<double>;
    static constexpr double kDefaultApproximationRatio = 2.0;
    static constexpr unsigned int kNumCandidates = 100;
    static constexpr unsigned int kScanSize = 128;

    static bool kUseFixedSeed;
    static constexpr unsigned int kDefaultSeed = 42;
    static constexpr unsigned int kDefaultDiskBlockSize = 4096;
    static constexpr unsigned int kDefaultDiskQueryBatchSize = 64;

    // qalsh4c
    static constexpr double kQalsh4cEpsilon = 1.0;
    static constexpr double kQalsh4cDelta = 0.1;
};
