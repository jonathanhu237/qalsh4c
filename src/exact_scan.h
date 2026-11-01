#pragma once

#include <vector>

#include "ann_searcher.h"
#include "point_set.h"

// Exact L1 or L2 nearest neighbors for a batch of queries. A fast pass over blocks of targets computes approximate
// distances: for L2, matrix products through ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y> with both sides centered on
// the queries' mean; for L1, vectorized sums of absolute differences in any order. A target is skipped only when a
// rounding-error bound proves it is farther than the best target found so far; every other target is rescored with
// Calc::LpDistanceBoundedRaw, so the distances equal those of a scalar scan.
class ExactScan {
   public:
    // Each query keeps its upper bound, with point id 0, unless a target is strictly closer, as in
    // LinearScanAnnSearcher::SearchWithUpperBound. The query coordinates are copied.
    ExactScan(const std::vector<const Coordinate*>& queries, unsigned int num_dimensions, float norm_order,
              const std::vector<Distance>& upper_bounds);

    // True for the norms this scan handles: L1 and L2.
    [[nodiscard]] static bool Supports(float norm_order);

    // Scans an in-memory target set whose point ids are its indices. Both scans run in parallel over blocks of
    // targets and must be called outside parallel regions.
    void ScanPoints(const std::vector<Point>& points);
    // Scans a contiguous row-major block of `count` targets whose point ids start at `first_id`.
    void ScanBlock(const Coordinate* block, unsigned int count, unsigned int first_id);

    // Best target per query over every scan so far.
    [[nodiscard]] std::vector<AnnResult> Results() const;

   private:
    struct ThreadState {
        std::vector<AnnResult> best;
        // Largest approximate value a target may have and still be no farther than `best`; see Threshold.
        std::vector<float> thresholds;
        std::vector<Coordinate> centered_targets;
        std::vector<float> adjusted_norms;
        std::vector<float> values;
    };

    template <typename TargetRow>
    void Scan(unsigned int first_id, unsigned int count, const TargetRow& target_row);
    template <typename TargetRow>
    void ScanTargets(ThreadState& state, unsigned int first_id, unsigned int count, const TargetRow& target_row) const;
    template <typename TargetRow>
    void ComputeL1Values(ThreadState& state, unsigned int chunk_begin, unsigned int chunk, unsigned int count,
                         const TargetRow& target_row) const;
    void ComputeL2Values(ThreadState& state, unsigned int chunk_begin, unsigned int chunk, unsigned int count) const;
    void Rescore(ThreadState& state, unsigned int query, unsigned int point_id, const Coordinate* target) const;
    [[nodiscard]] float Threshold(unsigned int query, Distance distance) const;

    unsigned int num_queries_{0};
    unsigned int num_dimensions_{0};
    float norm_order_{2.0F};
    bool l1_{false};
    // L2: relative error bound of the approximate values, relative to ||x||^2 + ||y||^2.
    double approximation_error_{0.0};
    // Relative error bound of a float sum of d nonnegative terms, as the scalar kernel and the L1 pass compute them.
    double summation_error_{0.0};
    std::vector<Coordinate> center_;
    std::vector<Coordinate> queries_;
    std::vector<Coordinate> centered_queries_;
    std::vector<double> query_norms_;
    std::vector<AnnResult> initial_results_;
    std::vector<ThreadState> states_;
};
