#pragma once

#include <cmath>
#include <numbers>
#include <string>

class Global {
   public:
    static constexpr float kEpsilon = 1e-6F;
    static constexpr float kQalshDefaultErrorProbability = 1.0F / std::numbers::e_v<float>;
    static constexpr float kDefaultApproximationRatio = 2.0F;
    static constexpr unsigned int kNumCandidates = 100;
    static constexpr unsigned int kScanSize = 128;

    static bool kUseFixedSeed;
    // Find exact nearest neighbors with the scalar kernel instead of the batched ExactScan (ablation).
    static bool kUseScalarExactScan;
    // Check QALSH's stopping condition only when a radius round is complete, which the per-point overestimation
    // bound D_a / v_a <= max(1, c J_a) assumes; by default it is also checked after every evaluated candidate.
    static bool kQalshStopAtRoundBoundary;
    // Diagnostic: when not empty, pool plans write every pilot draw (point id, U_a, D_a, v_a) to this CSV file.
    static std::string kPilotDumpPath;
    // Diagnostic: when not empty, pool sampling writes every source point's final upper bound, the one its exact-scan
    // draws follow, to this file as float64 values.
    static std::string kBoundsDumpPath;
    static constexpr unsigned int kDefaultSeed = 42;
    static constexpr unsigned int kDefaultDiskBlockSize = 4096;
    static constexpr unsigned int kDefaultDiskQueryBatchSize = 64;
    static constexpr unsigned int kDefaultDiskQalshTableBlockSize = 65536;
    static constexpr unsigned int kDefaultDiskQalshTableCacheBlocks = 64;

    // qalsh4c
    static constexpr float kQalsh4cEpsilon = 1.0F;
    static constexpr float kQalsh4cDelta = 0.1F;
};
