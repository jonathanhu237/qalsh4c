#pragma once

#include <cmath>
#include <numbers>

class Global {
   public:
    static constexpr float kEpsilon = 1e-6F;
    static constexpr float kQalshDefaultErrorProbability = 1.0F / std::numbers::e_v<float>;
    static constexpr float kDefaultApproximationRatio = 2.0F;
    static constexpr unsigned int kNumCandidates = 100;
    static constexpr unsigned int kScanSize = 128;

    static bool kUseFixedSeed;
    static constexpr unsigned int kDefaultSeed = 42;
    static constexpr unsigned int kDefaultDiskBlockSize = 4096;
    static constexpr unsigned int kDefaultDiskQueryBatchSize = 64;
    static constexpr unsigned int kDefaultDiskQalshTableBlockSize = 262144;
    static constexpr unsigned int kDefaultDiskQalshTableCacheBlocks = 64;

    // qalsh4c
    static constexpr float kQalsh4cEpsilon = 1.0F;
    static constexpr float kQalsh4cDelta = 0.1F;
};
