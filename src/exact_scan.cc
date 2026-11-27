#include "exact_scan.h"

#ifdef QALSH4C_USE_OPENMP
#include <omp.h>
#endif

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "global.h"
#include "log.h"
#include "utils.h"

namespace {

using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Unit roundoff of float arithmetic.
constexpr double kUnitRoundoff = 0x1p-24;
// Queries per pass over a block, which keeps a thread's block of approximate values in cache.
constexpr unsigned int kQueryChunk = 512;
// Coordinates per block of targets. Matrix products want large blocks; the L1 pass rereads its block from cache for
// every group of queries, so it keeps a smaller one.
constexpr std::size_t kL2BlockCoordinates = std::size_t{1} << 18;
constexpr std::size_t kL1BlockCoordinates = std::size_t{1} << 16;
constexpr std::size_t kMinBlockTargets = 16;
constexpr std::size_t kMaxBlockTargets = 1024;
// Queries per group in the L1 pass; every target coordinate loaded serves the whole group.
constexpr unsigned int kL1QueryGroup = 4;

bool IsNorm(float norm_order, float value) { return std::abs(norm_order - value) < Global::kEpsilon; }

bool IsBetter(const AnnResult& candidate, const AnnResult& best) {
    return candidate.distance < best.distance ||
           (candidate.distance == best.distance && candidate.point_id < best.point_id);
}

unsigned int MaxThreads() {
#ifdef QALSH4C_USE_OPENMP
    return static_cast<unsigned int>(std::max(1, omp_get_max_threads()));
#else
    return 1;
#endif
}

unsigned int ThreadNumber() {
#ifdef QALSH4C_USE_OPENMP
    return static_cast<unsigned int>(omp_get_thread_num());
#else
    return 0;
#endif
}

// Large enough for an efficient pass, small enough to give every thread a few blocks.
unsigned int BlockTargets(std::size_t block_coordinates, unsigned int num_dimensions, unsigned int count,
                          unsigned int num_threads) {
    const std::size_t by_size = block_coordinates / std::max(1U, num_dimensions);
    const std::size_t by_threads = (static_cast<std::size_t>(count) + 2 * num_threads - 1) / (2 * num_threads);
    return static_cast<unsigned int>(std::clamp(std::min(by_size, by_threads), kMinBlockTargets, kMaxBlockTargets));
}

}  // namespace

bool ExactScan::Supports(float norm_order) { return IsNorm(norm_order, 1.0F) || IsNorm(norm_order, 2.0F); }

ExactScan::ExactScan(const std::vector<const Coordinate*>& queries, unsigned int num_dimensions, float norm_order,
                     const std::vector<Distance>& upper_bounds)
    : num_queries_(static_cast<unsigned int>(queries.size())),
      num_dimensions_(num_dimensions),
      norm_order_(norm_order),
      l1_(IsNorm(norm_order, 1.0F)) {
    if (!Supports(norm_order)) {
        Log::error("Batched exact scans support only L1 and L2, got norm order {}.", norm_order);
    }
    if (queries.size() != upper_bounds.size()) {
        Log::error("Query size and upper-bound size mismatch.");
    }
    // L2 values are within approximation_error * (||x||^2 + ||y||^2) of their exact counterparts, covering rounding in
    // centering, norms, inner products, and the value itself. Both bounds include slack beyond the standard ones.
    const auto dimensions = static_cast<double>(num_dimensions_);
    approximation_error_ = 1.25 * (dimensions + 16.0) * kUnitRoundoff;
    summation_error_ = 1.01 * (dimensions + 2.0) * kUnitRoundoff;

    const std::size_t d = num_dimensions_;
    queries_.resize(num_queries_ * d);
    initial_results_.resize(num_queries_);
    for (unsigned int query = 0; query < num_queries_; ++query) {
        std::copy_n(queries[query], d, queries_.data() + query * d);
        // The hint is enlarged by twice the relative error of a float sum of d terms, which covers the gap between
        // this scan's kernel and the double kernel that produced a crude distance, then rounded up.
        const Distance bound = upper_bounds[query];
        Distance hint = std::numeric_limits<Distance>::infinity();
        if (bound < std::numeric_limits<Distance>::infinity()) {
            hint = std::nextafter(static_cast<float>(static_cast<double>(bound) * (1.0 + 2.0 * summation_error_)),
                                  std::numeric_limits<float>::infinity());
        }
        initial_results_[query] = AnnResult{.distance = hint, .point_id = kNoTarget};
    }
    if (l1_) {
        return;
    }

    std::vector<double> sums(d, 0.0);
    for (const Coordinate* query : queries) {
        for (std::size_t i = 0; i < d; ++i) {
            sums[i] += query[i];
        }
    }
    center_.resize(d);
    for (std::size_t i = 0; i < d; ++i) {
        center_[i] = num_queries_ == 0 ? 0.0F : static_cast<Coordinate>(sums[i] / num_queries_);
    }
    centered_queries_.resize(num_queries_ * d);
    query_norms_.resize(num_queries_);
    for (unsigned int query = 0; query < num_queries_; ++query) {
        const Coordinate* copy = queries_.data() + query * d;
        Coordinate* centered = centered_queries_.data() + query * d;
        double norm = 0.0;
        for (std::size_t i = 0; i < d; ++i) {
            centered[i] = copy[i] - center_[i];
            norm += static_cast<double>(centered[i]) * centered[i];
        }
        query_norms_[query] = norm;
    }
}

void ExactScan::ScanPoints(const std::vector<Point>& points) {
    Scan(0, static_cast<unsigned int>(points.size()), [&points](unsigned int index) { return points[index].data(); });
}

void ExactScan::ScanBlock(const Coordinate* block, unsigned int count, unsigned int first_id) {
    const std::size_t d = num_dimensions_;
    Scan(first_id, count, [block, d](unsigned int index) { return block + index * d; });
}

std::vector<AnnResult> ExactScan::Results() const {
    std::vector<AnnResult> results = initial_results_;
    for (const ThreadState& state : states_) {
        for (unsigned int query = 0; query < num_queries_; ++query) {
            if (IsBetter(state.best[query], results[query])) {
                results[query] = state.best[query];
            }
        }
    }
    return results;
}

template <typename TargetRow>
void ExactScan::Scan(unsigned int first_id, unsigned int count, const TargetRow& target_row) {
    if (num_queries_ == 0 || count == 0) {
        return;
    }
    const unsigned int num_threads = MaxThreads();
    // Each thread keeps its own best targets, merged by Results; a thread's threshold only gets looser than the
    // global one, which shortlists more targets but never skips one that could win.
    while (states_.size() < num_threads) {
        ThreadState& state = states_.emplace_back();
        state.best = initial_results_;
        state.thresholds.resize(num_queries_);
        for (unsigned int query = 0; query < num_queries_; ++query) {
            state.thresholds[query] = Threshold(query, initial_results_[query].distance);
        }
    }

    const unsigned int block_targets =
        BlockTargets(l1_ ? kL1BlockCoordinates : kL2BlockCoordinates, num_dimensions_, count, num_threads);
    const auto num_blocks = static_cast<long long>((static_cast<std::size_t>(count) + block_targets - 1) / block_targets);
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (long long block = 0; block < num_blocks; ++block) {
        const auto begin = static_cast<unsigned int>(block) * block_targets;
        ScanTargets(states_[ThreadNumber()], first_id + begin, std::min(block_targets, count - begin),
                    [&target_row, begin](unsigned int index) { return target_row(begin + index); });
    }
}

template <typename TargetRow>
void ExactScan::ScanTargets(ThreadState& state, unsigned int first_id, unsigned int count,
                            const TargetRow& target_row) const {
    const std::size_t d = num_dimensions_;
    if (!l1_) {
        state.centered_targets.resize(count * d);
        state.adjusted_norms.resize(count);
        for (unsigned int target = 0; target < count; ++target) {
            const Coordinate* row = target_row(target);
            Coordinate* centered = state.centered_targets.data() + target * d;
            double norm = 0.0;
            for (std::size_t i = 0; i < d; ++i) {
                centered[i] = row[i] - center_[i];
                norm += static_cast<double>(centered[i]) * centered[i];
            }
            state.adjusted_norms[target] = static_cast<float>(norm * (1.0 - approximation_error_));
        }
    }

    for (unsigned int chunk_begin = 0; chunk_begin < num_queries_; chunk_begin += kQueryChunk) {
        const unsigned int chunk = std::min(kQueryChunk, num_queries_ - chunk_begin);
        state.values.resize(static_cast<std::size_t>(chunk) * count);
        if (l1_) {
            ComputeL1Values(state, chunk_begin, chunk, count, target_row);
        } else {
            ComputeL2Values(state, chunk_begin, chunk, count);
        }

        for (unsigned int local = 0; local < chunk; ++local) {
            const unsigned int query = chunk_begin + local;
            const float* values = state.values.data() + static_cast<std::size_t>(local) * count;
            // The block's smallest value is rescored first, so the threshold is already tight when the rest of the
            // block is filtered.
            float smallest = std::numeric_limits<float>::infinity();
            unsigned int smallest_target = count;
            for (unsigned int target = 0; target < count; ++target) {
                if (values[target] < smallest) {
                    smallest = values[target];
                    smallest_target = target;
                }
            }
            if (smallest_target == count || smallest > state.thresholds[query]) {
                continue;
            }
            Rescore(state, query, first_id + smallest_target, target_row(smallest_target));
            for (unsigned int target = 0; target < count; ++target) {
                if (target != smallest_target && values[target] <= state.thresholds[query]) {
                    Rescore(state, query, first_id + target, target_row(target));
                }
            }
        }
    }
}

template <typename TargetRow>
void ExactScan::ComputeL1Values(ThreadState& state, unsigned int chunk_begin, unsigned int chunk, unsigned int count,
                                const TargetRow& target_row) const {
    const std::size_t d = num_dimensions_;
    float* values = state.values.data();
    unsigned int local = 0;
    for (; local + kL1QueryGroup <= chunk; local += kL1QueryGroup) {
        const Coordinate* first = queries_.data() + (chunk_begin + local) * d;
        const Coordinate* second = first + d;
        const Coordinate* third = second + d;
        const Coordinate* fourth = third + d;
        for (unsigned int target = 0; target < count; ++target) {
            const Coordinate* row = target_row(target);
            float sum_first = 0.0F;
            float sum_second = 0.0F;
            float sum_third = 0.0F;
            float sum_fourth = 0.0F;
#ifdef QALSH4C_USE_OPENMP
#pragma omp simd reduction(+ : sum_first, sum_second, sum_third, sum_fourth)
#endif
            for (std::size_t i = 0; i < d; ++i) {
                const Coordinate coordinate = row[i];
                sum_first += std::abs(first[i] - coordinate);
                sum_second += std::abs(second[i] - coordinate);
                sum_third += std::abs(third[i] - coordinate);
                sum_fourth += std::abs(fourth[i] - coordinate);
            }
            const std::size_t column = static_cast<std::size_t>(local) * count + target;
            values[column] = sum_first;
            values[column + count] = sum_second;
            values[column + 2 * static_cast<std::size_t>(count)] = sum_third;
            values[column + 3 * static_cast<std::size_t>(count)] = sum_fourth;
        }
    }
    for (; local < chunk; ++local) {
        const Coordinate* query = queries_.data() + (chunk_begin + local) * d;
        for (unsigned int target = 0; target < count; ++target) {
            const Coordinate* row = target_row(target);
            float sum = 0.0F;
#ifdef QALSH4C_USE_OPENMP
#pragma omp simd reduction(+ : sum)
#endif
            for (std::size_t i = 0; i < d; ++i) {
                sum += std::abs(query[i] - row[i]);
            }
            values[static_cast<std::size_t>(local) * count + target] = sum;
        }
    }
}

void ExactScan::ComputeL2Values(ThreadState& state, unsigned int chunk_begin, unsigned int chunk,
                                unsigned int count) const {
    const std::size_t d = num_dimensions_;
    const auto rows = static_cast<Eigen::Index>(count);
    const Eigen::Map<const RowMajorMatrix> targets(state.centered_targets.data(), rows, static_cast<Eigen::Index>(d));
    const Eigen::Map<const RowMajorMatrix> queries(centered_queries_.data() + chunk_begin * d,
                                                   static_cast<Eigen::Index>(chunk), static_cast<Eigen::Index>(d));
    Eigen::Map<RowMajorMatrix> products(state.values.data(), static_cast<Eigen::Index>(chunk), rows);
    products.noalias() = queries * targets.transpose();
    // Approximate value ||y||^2 (1 - e) - 2 <x, y>.
    for (unsigned int local = 0; local < chunk; ++local) {
        float* values = state.values.data() + static_cast<std::size_t>(local) * count;
        for (unsigned int target = 0; target < count; ++target) {
            values[target] = state.adjusted_norms[target] - 2.0F * values[target];
        }
    }
}

void ExactScan::Rescore(ThreadState& state, unsigned int query, unsigned int point_id, const Coordinate* target) const {
    AnnResult& best = state.best[query];
    const Distance distance = Calc::LpDistanceBoundedRaw(
        target, queries_.data() + static_cast<std::size_t>(query) * num_dimensions_, num_dimensions_, norm_order_,
        best.distance);
    const AnnResult candidate{.distance = distance, .point_id = point_id};
    if (IsBetter(candidate, best)) {
        best = candidate;
        state.thresholds[query] = Threshold(query, distance);
    }
}

float ExactScan::Threshold(unsigned int query, Distance distance) const {
    if (!(distance < std::numeric_limits<Distance>::infinity())) {
        return std::numeric_limits<float>::infinity();
    }
    double threshold = 0.0;
    if (l1_) {
        // Both the approximate value and the scalar sum are within summation_error of the exact sum, so a target
        // whose value exceeds this sums to more than `distance` and cannot win or tie.
        threshold = static_cast<double>(distance) * (1.0 + summation_error_) / (1.0 - summation_error_);
    } else {
        // A target whose float sum exceeds `squared` rounds to a distance above `distance`. Its exact squared distance
        // is at least ||x||^2 (1 - e) + value, and its float sum at least (1 - r) times that.
        const double squared = static_cast<double>(distance) * distance * (1.0 + 8.0 * kUnitRoundoff);
        threshold = squared / (1.0 - summation_error_) - query_norms_[query] * (1.0 - approximation_error_);
    }
    // Round up, so the conversion never drops a target the bound keeps.
    return std::nextafter(static_cast<float>(threshold), std::numeric_limits<float>::infinity());
}
