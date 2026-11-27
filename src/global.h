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
    // Seed of the projections and the sampler when non-negative (paired runs of different rules); -1 draws one.
    static long long kSeed;
    // Test hook: find exact nearest neighbors with the scalar kernel, the reference of the batched ExactScan.
    static bool kUseScalarExactScan;
    // Starting radius R_0 of the round rules; their bound on kappa needs R_0 at most every positive d(a, B).
    static float kInitialRadius;
    // Round rule: test QALSH's stopping condition T1 at the end of a round (true) or on every new candidate (false,
    // QALSH's own test, gamma = c^2).
    static bool kQalshStopAtRoundBoundary;
    // Round rule with end-of-round tests: also stop inside a round on a candidate within this multiple of the radius
    // (1 by default, so gamma stays c; 0 stops only at the end of a round).
    static float kQalshMidRoundRatio;
    // Quanta order: entries one table visit scans before the next table (both sides together).
    static constexpr unsigned int kQalshScanQuantum = 2 * kScanSize;
    // Order of the window entries: "quanta" (the round rules), "distance" (all tables advance together in distance
    // from the query's projection, the continuous front) or "distance-reference" (the same order, one entry at a time,
    // for checks).
    static std::string kQalshScanOrder;
    // Slabs per round for the distance order; they only change how much work a stop wastes, not the results.
    static unsigned int kQalshSlabsPerRound;
    // Round rule: also test the stopping condition right after the radius grows, against the new radius (gamma = c^2).
    static bool kQalshCheckAfterRadiusAdvance;
    // "continuous" (the paper's QALSH-CrudeNN: stop once the best distance is at most c times the radius of the scan
    // front) or "round" (the round rules, compared in Sect. 4.2).
    static std::string kQalshStopRule;
    // Continuous rule: stop once the best distance is at most this multiple gamma of the front's radius (0 uses the
    // default c^1.5). The factor gamma of the kappa bound is this multiple, so c^2 gives the T of QALSH's own test.
    static float kQalshContinuousRatio;
    static constexpr unsigned int kDefaultSeed = 42;

    // Default tolerance and failure probability of the automatic sample size.
    static constexpr float kQalsh4cEpsilon = 0.1F;
    static constexpr float kQalsh4cDelta = 0.1F;
};
