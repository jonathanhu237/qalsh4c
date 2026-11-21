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
    // Find exact nearest neighbors with the scalar kernel instead of the batched ExactScan (ablation).
    static bool kUseScalarExactScan;
    // Starting search radius R_0 of QALSH queries; the analysis needs R_0 <= c * d(a, B) for every positive d(a, B).
    static float kInitialRadius;
    // Check the stopping condition T1 only after a whole radius round, as in the paper's Algorithm 1. This gives the
    // expected-overestimation constant c (1 + (c - 1) S) instead of c^2 (1 + (c - 1) S).
    static bool kQalshStopAtRoundBoundary;
    // With round-boundary checks, also stop inside a round once a candidate lies within this multiple of the radius.
    // A value of at most one keeps the constant c (1 + (c - 1) S); zero gives the pure round-end rule.
    static float kQalshMidRoundRatio;
    // Fraction beta in QALSH's table rule m = ceil((sqrt(ln(2/beta)) + sqrt(ln(1/delta)))^2 / (2 (p1 - p2)^2)).
    // QALSH sets beta = 100 / n so that at most 100 false candidates pass its candidate-budget rule; QALSH-CrudeNN
    // has no candidate budget and its constant kappa depends only on the nearest neighbor's miss probabilities, so
    // the default is beta = 1. Zero restores QALSH's beta = min(100 / n, 1).
    static float kQalshBeta;
    // Explicit table count m and collision threshold l (0 keeps QALSH's rule). The constant kappa is computable for
    // any pair, so these only trade crude-stage cost against kappa.
    static unsigned int kQalshTables;
    static unsigned int kQalshThreshold;
    // Explicit bucket width w (0 keeps QALSH's width, which maximizes p1 - p2).
    static float kQalshBucketWidth;
    // Entries an in-memory QALSH query scans in one table before it moves to the next table (both sides together).
    static unsigned int kQalshScanQuantum;
    static constexpr unsigned int kDefaultSeed = 42;
    static constexpr unsigned int kDefaultDiskBlockSize = 4096;
    static constexpr unsigned int kDefaultDiskQueryBatchSize = 64;
    static constexpr unsigned int kDefaultDiskQalshTableBlockSize = 65536;
    static constexpr unsigned int kDefaultDiskQalshTableCacheBlocks = 64;

    // qalsh4c
    static constexpr float kQalsh4cEpsilon = 1.0F;
    static constexpr float kQalsh4cDelta = 0.1F;
};
