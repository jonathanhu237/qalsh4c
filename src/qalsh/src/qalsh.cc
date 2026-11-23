#include "qalsh/qalsh.h"
#include "window_clamp.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <thread>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qalsh {
namespace {

// Prevent a malformed configuration from requesting an unbounded vector/table
// allocation. These bounds are well above the supported consumer workloads.
constexpr std::uint32_t kMaxReasonableHashTables = 1U << 20;
constexpr std::uint32_t kMaxReasonableDimensions = 1U << 20;
constexpr std::uint32_t kMaxReasonablePoints = 1U << 30;

[[nodiscard]] bool IsFinite(float value) { return std::isfinite(value); }

#if defined(__GNUC__) || defined(__clang__)
[[gnu::always_inline]]
#endif
inline void Require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::invalid_argument(std::string(message));
    }
}

#if defined(__GNUC__) || defined(__clang__)
[[gnu::always_inline]]
#endif
inline void RequireFinite(float value, std::string_view name) {
    if (!IsFinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

float WindowBound(float width, float radius) {
    const double value = static_cast<double>(width) * radius / 2.0;
    Require(std::isfinite(value) && value >= 0 && value <= std::numeric_limits<float>::max(),
            "QALSH scan bound is not representable");
    return static_cast<float>(value);
}

[[nodiscard]] bool CheckedMultiply(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t& result) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

[[nodiscard]] std::uint64_t CheckedMultiplyOrThrow(std::uint64_t lhs, std::uint64_t rhs,
                                                    std::string_view description) {
    std::uint64_t result = 0;
    Require(CheckedMultiply(lhs, rhs, result), description);
    return result;
}

[[nodiscard]] std::size_t CheckedSize(std::uint64_t value, std::string_view description) {
    Require(value <= std::numeric_limits<std::size_t>::max(), description);
    return static_cast<std::size_t>(value);
}

[[nodiscard]] bool IsValidMetric(Metric metric) {
    return metric == Metric::l1 || metric == Metric::l2;
}

void ValidateQalsh(const QalshParameters& qalsh) {
    RequireFinite(qalsh.approximation_ratio, "approximation ratio");
    Require(qalsh.approximation_ratio > 1.0F, "approximation ratio must be greater than one");
    RequireFinite(qalsh.bucket_width, "bucket width");
    Require(qalsh.bucket_width > 0.0F, "bucket width must be positive");
    Require(IsFinite(qalsh.error_probability) && qalsh.error_probability > 0.0F &&
                qalsh.error_probability < 1.0F,
            "error probability must be in (0, 1)");
    Require(qalsh.num_hash_tables > 0, "the index must contain at least one projection table");
    Require(qalsh.candidate_budget > 0, "candidate budget must be positive");
    Require(qalsh.collision_threshold <= qalsh.num_hash_tables,
            "collision threshold exceeds the table count");
    Require(IsFinite(qalsh.initial_radius) && qalsh.initial_radius > 0.0F,
            "initial radius must be positive");
    Require(IsFinite(qalsh.radius_growth) && qalsh.radius_growth > 1.0F,
            "radius growth must be greater than one");
    Require(qalsh.scan_quantum > 0, "scan quantum must be positive");
}

void ValidateConfig(const IndexConfig& config) {
    Require(IsValidMetric(config.metric), "unsupported metric");
    Require(config.build_threads > 0 && config.build_threads <= 1024,
            "build_threads is outside the supported resource bounds");
    Require(config.num_points > 0 && config.num_points <= kMaxReasonablePoints,
            "num_points is outside the supported resource bounds");
    Require(config.num_dimensions > 0 && config.num_dimensions <= kMaxReasonableDimensions,
            "num_dimensions is outside the supported resource bounds");
    Require(config.qalsh.num_hash_tables > 0 &&
                config.qalsh.num_hash_tables <= kMaxReasonableHashTables,
            "num_hash_tables is outside the supported resource bounds");
    ValidateQalsh(config.qalsh);
    const std::uint64_t expected_projection_values = CheckedMultiplyOrThrow(
        config.qalsh.num_hash_tables, config.num_dimensions, "projection-vector count overflows");
    const std::size_t expected_size = CheckedSize(expected_projection_values,
                                                  "projection-vector count is too large");
    if (!config.projection_vectors.empty()) {
        Require(config.projection_vectors.size() == expected_size,
                "projection_vectors has the wrong size");
        for (float value : config.projection_vectors) {
            RequireFinite(value, "projection vector coordinate");
        }
    }
}

// Sorting only touches independent, fully populated projection tables.
// after_sort is called on each table after it is sorted, on the worker that sorted it.
template <typename Tables, typename Entries, typename AfterSort>
void SortTables(Tables& tables, std::uint32_t requested_threads, Entries entries, AfterSort after_sort) {
    const auto sort_one = [&](std::size_t id) {
        std::ranges::sort(entries(tables[id]), [](const auto& lhs, const auto& rhs) {
            return lhs.value != rhs.value ? lhs.value < rhs.value : lhs.point_id < rhs.point_id;
        });
        after_sort(tables[id]);
    };
    const auto workers = std::min<std::size_t>(requested_threads, tables.size());
    if (workers <= 1) {
        for (std::size_t id = 0; id < tables.size(); ++id) sort_one(id);
        return;
    }
    std::atomic<std::size_t> next{0};
    const auto work = [&] {
        for (;;) {
            const auto id = next.fetch_add(1, std::memory_order_relaxed);
            if (id >= tables.size()) return;
            sort_one(id);
        }
    };
    std::vector<std::jthread> threads;
    threads.reserve(workers - 1);
    try {
        for (std::size_t i = 1; i < workers; ++i) threads.emplace_back(work);
    } catch (...) {
        next.store(tables.size(), std::memory_order_relaxed);
        threads.clear();
        throw;
    }
    work();
    // jthread destruction joins all sorts before the builder consumes tables.
}

[[nodiscard]] std::vector<Coordinate> MakeProjectionVectors(const IndexConfig& config) {
    if (!config.projection_vectors.empty()) {
        return config.projection_vectors;
    }

    const std::uint64_t expected_values = CheckedMultiplyOrThrow(
        config.qalsh.num_hash_tables, config.num_dimensions, "projection-vector count overflows");
    const std::size_t expected_size = CheckedSize(expected_values, "projection-vector count is too large");
    std::mt19937 generator(config.seed);
    std::vector<Coordinate> vectors;
    vectors.reserve(expected_size);
    if (config.metric == Metric::l1) {
        std::cauchy_distribution<float> distribution(0.0F, 1.0F);
        for (std::size_t i = 0; i < expected_size; ++i) {
            const float value = distribution(generator);
            RequireFinite(value, "generated projection vector coordinate");
            vectors.push_back(value);
        }
    } else if (config.metric == Metric::l2) {
        std::normal_distribution<float> distribution(0.0F, 1.0F);
        for (std::size_t i = 0; i < expected_size; ++i) {
            const float value = distribution(generator);
            RequireFinite(value, "generated projection vector coordinate");
            vectors.push_back(value);
        }
    } else {
        throw std::invalid_argument("unsupported metric");
    }
    return vectors;
}

[[nodiscard]] IndexMetadata MetadataFromConfig(const IndexConfig& config) {
    return IndexMetadata{.metric = config.metric,
                         .num_points = config.num_points,
                         .num_dimensions = config.num_dimensions,
                         .num_hash_tables = config.qalsh.num_hash_tables,
                         .seed = config.seed,
                         .qalsh = config.qalsh};
}

void ValidatePoint(PointView point, std::uint32_t dimensions, PointId point_id) {
    if (point.size() != dimensions) {
        throw std::invalid_argument("point accessor returned the wrong dimensionality for point " +
                                    std::to_string(point_id));
    }
    unsigned finite = 1;
#ifdef QALSH_USE_OPENMP_SIMD
#pragma omp simd reduction(& : finite)
#endif
    for (std::size_t i = 0; i < point.size(); ++i) {
        finite &= static_cast<unsigned>(IsFinite(point[i]));
    }
    if (finite == 0) {
        throw std::invalid_argument("point accessor returned a non-finite coordinate");
    }
}

// Independent tables hide SIMD accumulation latency. The compiler chooses
// each instantiation's reduction tree. This is projection arithmetic, not
// candidate batching: accepted points are still evaluated immediately.
template <unsigned Count>
void DotGroup(PointView point, const Coordinate* vectors, std::size_t dimensions, float* output) {
    static_assert(Count == 1 || Count == 4);
    [[maybe_unused]] float a = 0, b = 0, c = 0, d = 0;
#ifdef QALSH_USE_OPENMP_SIMD
#pragma omp simd reduction(+ : a, b, c, d)
#endif
    for (std::size_t i = 0; i < dimensions; ++i) {
        const float x = point[i];
        a += x * vectors[i];
        if constexpr (Count == 4) {
            b += x * vectors[dimensions + i];
            c += x * vectors[2 * dimensions + i];
            d += x * vectors[3 * dimensions + i];
        }
    }
    output[0] = a;
    if constexpr (Count == 4) {
        output[1] = b; output[2] = c; output[3] = d;
    }
}

void ProjectPoint(PointView point, const Coordinate* vectors, std::size_t dimensions,
                  std::span<Projection> output) {
    std::size_t table = 0;
    for (; table + 4 <= output.size(); table += 4) {
        DotGroup<4>(point, vectors + table * dimensions, dimensions, output.data() + table);
    }
    for (; table < output.size(); ++table) {
        DotGroup<1>(point, vectors + table * dimensions, dimensions, output.data() + table);
    }
}

[[nodiscard]] float FromBfloat16(std::uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

// Writes the bfloat16 copy of a finite point, coordinate order[i] in place i (rounded to nearest, ties to even), and
// returns its kind for InMemoryIndex::compressed_kind_: 1 when every coordinate is exact, 0 when some are rounded,
// 2 when one overflows.
[[nodiscard]] std::uint8_t CompressPoint(PointView point, const std::uint32_t* order, std::uint16_t* row) noexcept {
    std::uint32_t rounded_bits = 0;
    std::uint32_t overflow = 0;
#ifdef QALSH_USE_OPENMP_SIMD
#pragma omp simd reduction(| : rounded_bits, overflow)
#endif
    for (std::size_t i = 0; i < point.size(); ++i) {
        const auto bits = std::bit_cast<std::uint32_t>(point[order[i]]);
        const std::uint32_t upper = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U;
        row[i] = static_cast<std::uint16_t>(upper);
        rounded_bits |= (upper << 16U) ^ bits;
        overflow |= static_cast<std::uint32_t>((upper & 0x7F80U) == 0x7F80U);
    }
    return overflow != 0 ? 2 : (rounded_bits == 0 ? 1 : 0);
}

// Coordinates in decreasing order of their variance over an evenly spaced sample of at most 4,096 points (ties by
// index). The order only decides how early the compressed lower bound can reject; any order gives the same results.
[[nodiscard]] std::vector<std::uint32_t> SpreadOrder(const IndexConfig& config, const PointAccessor& accessor) {
    const std::size_t dimensions = config.num_dimensions;
    const std::size_t samples = std::min<std::size_t>(config.num_points, 4096);
    std::vector<double> sum(dimensions, 0.0);
    std::vector<double> sum_squares(dimensions, 0.0);
    for (std::size_t sample = 0; sample < samples; ++sample) {
        const auto point_id = static_cast<PointId>(sample * config.num_points / samples);
        const PointView point = accessor(point_id);
        ValidatePoint(point, config.num_dimensions, point_id);
        for (std::size_t i = 0; i < dimensions; ++i) {
            const double value = point[i];
            sum[i] += value;
            sum_squares[i] += value * value;
        }
    }
    std::vector<double> variance(dimensions, 0.0);
    for (std::size_t i = 0; i < dimensions; ++i) {
        const double mean = sum[i] / static_cast<double>(samples);
        variance[i] = sum_squares[i] / static_cast<double>(samples) - mean * mean;
    }
    std::vector<std::uint32_t> order(dimensions);
    for (std::size_t i = 0; i < dimensions; ++i) order[i] = static_cast<std::uint32_t>(i);
    std::stable_sort(order.begin(), order.end(),
                     [&variance](std::uint32_t lhs, std::uint32_t rhs) { return variance[lhs] > variance[rhs]; });
    return order;
}

// visit_point is called for every point, with its projections, while the projection tables are built; build workers
// may call it for different points at the same time.
template <typename Tables, typename Entries, typename VisitPoint>
void BuildProjectionTables(Tables& tables, const IndexConfig& config, const PointAccessor& accessor,
                            std::span<const Coordinate> vectors, Entries entries, VisitPoint visit_point) {
    constexpr std::uint64_t kMinProjectionProductsPerWorker = 256U * 1024U;
    const std::uint64_t products_per_point =
        static_cast<std::uint64_t>(config.num_dimensions) * config.qalsh.num_hash_tables;
    const std::uint64_t min_points_per_worker =
        std::max<std::uint64_t>(1, (kMinProjectionProductsPerWorker + products_per_point - 1) /
                                      products_per_point);

    for (auto& table : tables) {
        entries(table).reserve(config.num_points);
    }
    // Every worker reads, checks and projects its own runs of points. Without a concurrent accessor the calling thread
    // is the only worker and reads the points in increasing id. Each point's projections do not depend on the worker,
    // so the tables are identical for every worker count.
    const std::size_t workers_wanted =
        config.concurrent_accessor
            ? static_cast<std::size_t>(std::min<std::uint64_t>(
                  config.build_threads, std::max<std::uint64_t>(1, config.num_points / min_points_per_worker)))
            : 1;
    for (auto& table : tables) {
        entries(table).resize(config.num_points);
    }
    constexpr std::size_t kPointsPerTask = 256;
    std::atomic<std::size_t> next_point{0};
    std::atomic<bool> stop{false};
    std::mutex failure_mutex;
    std::exception_ptr failure;
    const auto work = [&] {
        try {
            std::vector<Projection> projected(config.qalsh.num_hash_tables);
            for (;;) {
                const std::size_t begin = next_point.fetch_add(kPointsPerTask, std::memory_order_relaxed);
                if (begin >= config.num_points || stop.load(std::memory_order_relaxed)) return;
                const std::size_t end = std::min<std::size_t>(begin + kPointsPerTask, config.num_points);
                for (std::size_t index = begin; index < end; ++index) {
                    const auto point_id = static_cast<PointId>(index);
                    const PointView point = accessor(point_id);
                    ValidatePoint(point, config.num_dimensions, point_id);
                    ProjectPoint(point, vectors.data(), config.num_dimensions, projected);
                    for (std::uint32_t table_id = 0; table_id < config.qalsh.num_hash_tables; ++table_id) {
                        RequireFinite(projected[table_id], "projection value");
                        entries(tables[table_id])[point_id] = {.value = projected[table_id], .point_id = point_id};
                    }
                    visit_point(point_id, point, std::span<const Projection>(projected));
                }
            }
        } catch (...) {
            stop.store(true, std::memory_order_relaxed);
            std::lock_guard lock(failure_mutex);
            if (!failure) failure = std::current_exception();
        }
    };
    {
        std::vector<std::jthread> threads;
        threads.reserve(workers_wanted - 1);
        for (std::size_t worker = 1; worker < workers_wanted; ++worker) threads.emplace_back(work);
        work();
    }
    if (failure) std::rethrow_exception(failure);
}

[[nodiscard]] bool BetterNeighbor(const Neighbor& lhs, const Neighbor& rhs) {
    return lhs.distance < rhs.distance ||
           (lhs.distance == rhs.distance && lhs.point_id < rhs.point_id);
}

void InsertNeighborInto(std::vector<Neighbor>& neighbors, const Neighbor& candidate, std::uint32_t k) {
    const auto insertion = std::lower_bound(
        neighbors.begin(), neighbors.end(), candidate,
        [](const Neighbor& existing, const Neighbor& value) { return BetterNeighbor(existing, value); });
    if (neighbors.size() == k && insertion == neighbors.end()) {
        return;
    }
    neighbors.insert(insertion, candidate);
    if (neighbors.size() > k) {
        neighbors.pop_back();
    }
}

// The result of a bounded distance: exact=false and infinity when the bound proves the distance larger.
struct BoundedDistance {
    Distance distance{0.0F};
    bool exact{false};
};

// Bounded and unbounded calls use exactly the same accumulation order. A
// separate scalar bounded loop and SIMD unbounded loop can disagree by several
// float ULPs and incorrectly prune even identical high-dimensional vectors.
// Double intermediates also keep representable L2 distances from overflowing
// or underflowing merely because their squares do not fit in float32.
template <Metric Norm>
BoundedDistance DistanceKernel(PointView lhs, PointView rhs, Distance upper_bound) {
    constexpr auto infinity = std::numeric_limits<float>::infinity();
    const double expanded = std::nextafter(upper_bound, infinity);
    const double limit = Norm == Metric::l2 ? expanded * expanded : expanded;
    double sum = 0;
    for (std::size_t begin = 0; begin < lhs.size(); begin += 64) {
        const auto end = std::min(begin + 64, lhs.size());
        double block = 0;
#ifdef QALSH_USE_OPENMP_SIMD
#pragma omp simd reduction(+ : block)
#endif
        for (std::size_t i = begin; i < end; ++i) {
            const double difference = double(lhs[i]) - double(rhs[i]);
            if constexpr (Norm == Metric::l2) block += difference * difference;
            else block += std::abs(difference);
        }
        sum += block;
        if (sum > limit) return {.distance = infinity, .exact = false};
    }
    const double value = Norm == Metric::l2 ? std::sqrt(sum) : sum;
    if (value > std::numeric_limits<float>::max()) {
        if (std::isfinite(upper_bound)) return {.distance = infinity, .exact = false};
        throw std::overflow_error("distance is not representable as float32");
    }
    const auto distance = static_cast<float>(value);
    if (distance > upper_bound) return {.distance = infinity, .exact = false};
    return {.distance = distance, .exact = true};
}

[[nodiscard]] BoundedDistance BoundedDistanceValueUnchecked(PointView lhs, PointView rhs,
                                                            Metric metric, Distance upper_bound) {
    return metric == Metric::l1 ? DistanceKernel<Metric::l1>(lhs, rhs, upper_bound)
                               : DistanceKernel<Metric::l2>(lhs, rhs, upper_bound);
}

// True when a lower bound computed from the bfloat16 copy of a target proves that DistanceKernel<Norm>(target, query,
// upper_bound) is not exact, where `limit` is the kernel's limit for that upper bound.
//
// For every coordinate, |a - b| >= |a - b'| - e with b' the rounded copy and e >= |b - b'|: zero for an exact copy,
// otherwise e = 2^-8 |b'| + 2^-133, at least twice the rounding error of round to nearest. So the sum over coordinates
// of max(0, |a - b'| - e), squared for L2, is at most the exact sum S. In float arithmetic (unit roundoff u = 2^-24),
// fl(a - b') is exact up to a factor 1 + u and the slack in e absorbs that error, so each computed term is at most
// (1 + 5u) times its exact counterpart for |a - b|; any summation order adds at most a factor 1 + (d - 1) u. The
// kernel's double sum is at least S (1 - (d + 2) 2^-53). A computed bound above limit (1 + (d + 16) 2^-23) therefore
// implies that the kernel's sum exceeds `limit`, and then the kernel rejects. Very large limits are left to the kernel,
// so the float sums cannot overflow.
template <Metric Norm>
[[nodiscard]] bool CompressedBoundRejects(const float* query, const std::uint16_t* target, std::size_t dimensions,
                                          bool exact_copy, double limit) {
    const double margin = 1.0 + (static_cast<double>(dimensions) + 16.0) * 0x1p-23;
    const double threshold_double = limit * margin;
    if (!(threshold_double < 0x1p100)) return false;
    // Round the threshold up so the float comparison never rejects below it.
    auto threshold = static_cast<float>(threshold_double);
    if (static_cast<double>(threshold) < threshold_double) {
        threshold = std::nextafter(threshold, std::numeric_limits<float>::infinity());
    }
    const float scale = exact_copy ? 0.0F : 0x1p-8F;
    const float floor = exact_copy ? 0.0F : 0x1p-133F;
    float sum = 0.0F;
    for (std::size_t begin = 0; begin < dimensions; begin += 128) {
        const std::size_t end = std::min<std::size_t>(begin + 128, dimensions);
        float block = 0.0F;
#ifdef QALSH_USE_OPENMP_SIMD
#pragma omp simd reduction(+ : block)
#endif
        for (std::size_t i = begin; i < end; ++i) {
            const float rounded = FromBfloat16(target[i]);
            const float gap = std::abs(query[i] - rounded) - (std::abs(rounded) * scale + floor);
            const float term = gap > 0.0F ? gap : 0.0F;
            if constexpr (Norm == Metric::l2) block += term * term;
            else block += term;
        }
        sum += block;
        if (sum > threshold) return true;
    }
    return false;
}

// Scratch space of SearchEngine::search. With the quanta scan order, in each radius round the tables that still have
// window entries are visited in increasing order, cycle after cycle, and a visit scans up to ceil(Q / 2) entries on the
// left side and floor(Q / 2) on the right side of the query's projection, counting each hit and evaluating a point
// when its counter runs out. The projection-distance orders are described in SearchEngine::search.
constexpr std::size_t kNoEntry = std::numeric_limits<std::size_t>::max();

struct SearchScratch {
    std::vector<std::uint8_t> counters;
    std::vector<Projection> query_projections;
    std::vector<std::size_t> left;
    std::vector<std::size_t> right;
    std::vector<std::uint8_t> left_blocked;
    std::vector<std::uint8_t> right_blocked;
    std::vector<float> last_bound;
    // Tables whose cursors ran out of entries on both sides.
    std::vector<std::uint8_t> exhausted;
    std::vector<std::uint32_t> active;
    std::vector<std::uint32_t> next_active;
    std::vector<Neighbor> neighbors;
    std::vector<float> query_ordered;
    // Projection-distance order: the slab's new candidates (collision radius, id) and per-table distances.
    std::vector<std::pair<float, PointId>> distance_events;
    std::vector<float> gaps;
};

thread_local SearchScratch search_scratch;

// Ids per 64-byte cache line of an id column.
constexpr std::size_t kIdsPerLine = 64U / sizeof(PointId);

}  // namespace

std::shared_ptr<InMemoryIndex> InMemoryIndex::Build(IndexConfig config, const PointAccessor& accessor) {
    ValidateConfig(config);
    Require(static_cast<bool>(accessor), "point accessor is required");
    auto index = std::shared_ptr<InMemoryIndex>(new InMemoryIndex());
    index->metadata_ = MetadataFromConfig(config);
    index->projection_vectors_ = MakeProjectionVectors(config);
    index->tables_.resize(config.qalsh.num_hash_tables);
    if (config.compressed_filter) {
        index->compressed_order_ = SpreadOrder(config, accessor);
        index->compressed_points_ = std::make_unique_for_overwrite<std::uint16_t[]>(
            static_cast<std::size_t>(config.num_points) * config.num_dimensions);
        index->compressed_kind_.assign(config.num_points, 0);
    }
    if (config.point_projections) {
        index->point_projections_ = std::make_unique_for_overwrite<Projection[]>(
            static_cast<std::size_t>(config.num_points) * config.qalsh.num_hash_tables);
    }
    // The compressed copy and the point-major projections are written while the points are projected.
    const auto visit = [&index, dimensions = std::size_t{config.num_dimensions},
                        tables = std::size_t{config.qalsh.num_hash_tables}](PointId point_id, PointView point,
                                                                             std::span<const Projection> projected) {
        if (index->compressed_points_) {
            index->compressed_kind_[point_id] =
                CompressPoint(point, index->compressed_order_.data(),
                              index->compressed_points_.get() + static_cast<std::size_t>(point_id) * dimensions);
        }
        if (index->point_projections_) {
            std::copy(projected.begin(), projected.end(),
                      index->point_projections_.get() + static_cast<std::size_t>(point_id) * tables);
        }
    };
    BuildProjectionTables(index->tables_, config, accessor, index->projection_vectors_,
                            [](auto& table) -> auto& { return table.entries; }, visit);
    // Each sorted table also gets its id column, on the worker that sorted it.
    SortTables(index->tables_, config.build_threads, [](auto& table) -> auto& { return table.entries; },
               [](Table& table) {
                   table.ids.resize(table.entries.size());
                   for (std::size_t i = 0; i < table.entries.size(); ++i) table.ids[i] = table.entries[i].point_id;
               });
    return index;
}

const IndexMetadata& InMemoryIndex::metadata() const noexcept { return metadata_; }
std::span<const Coordinate> InMemoryIndex::projection_vectors() const noexcept {
    return projection_vectors_;
}

DefaultQalshStrategy::DefaultQalshStrategy(std::uint32_t collision_threshold)
    : collision_threshold_(collision_threshold) {}

void DefaultQalshStrategy::set_candidate_budget(std::optional<std::size_t> budget) {
    if (budget.has_value()) {
        Require(*budget > 0, "candidate budget must be positive");
    }
    candidate_budget_override_set_ = true;
    candidate_budget_override_ = budget;
}

void DefaultQalshStrategy::set_check_after_evaluation(bool enabled) {
    check_after_evaluation_ = enabled;
}

void DefaultQalshStrategy::set_mid_round_ratio(float ratio) {
    mid_round_ratio_ = ratio;
}

void DefaultQalshStrategy::set_check_after_radius_advance(bool enabled) {
    check_after_radius_advance_ = enabled;
}

void DefaultQalshStrategy::set_continuous_stop(bool enabled) {
    continuous_stop_ = enabled;
}

SearchEngine::SearchEngine(std::shared_ptr<const ProjectionIndex> index,
                           PointAccessor accessor, SearchOptions options)
    : index_(std::dynamic_pointer_cast<const InMemoryIndex>(index)), accessor_(std::move(accessor)),
      options_(options) {
    if (!index) {
        throw std::invalid_argument("search engine requires an index");
    }
    if (!index_) {
        throw std::invalid_argument("search engine requires an in-memory index");
    }
    if (!accessor_) {
        throw std::invalid_argument("search engine requires a point accessor");
    }
}

SearchResult SearchEngine::search(PointView query, std::uint32_t k, const DefaultQalshStrategy& strategy) const {
    const InMemoryIndex& index = *index_;
    const IndexMetadata& metadata = index.metadata();
    Require(IsValidMetric(metadata.metric), "unsupported index metric");
    Require(k > 0, "k must be positive");
    Require(k <= metadata.num_points, "k exceeds the number of indexed points");
    Require(query.size() == metadata.num_dimensions, "query has the wrong dimensionality");
    for (float value : query) {
        RequireFinite(value, "query coordinate");
    }
    const std::uint32_t table_count = options_.num_hash_tables.value_or(metadata.num_hash_tables);
    Require(table_count > 0 && table_count <= metadata.num_hash_tables, "invalid query table count");
    const std::uint32_t scan_quantum =
        options_.scan_quantum == 0 ? metadata.qalsh.scan_quantum : options_.scan_quantum;
    const std::uint64_t projection_values = CheckedMultiplyOrThrow(
        table_count, metadata.num_dimensions, "query projection layout overflows");
    Require(index.projection_vectors().size() >=
                CheckedSize(projection_values, "query projection layout is too large"),
            "projection index has insufficient projection vectors");

    // The strategy's settings.
    const std::uint32_t threshold = strategy.collision_threshold_ == 0 ? metadata.qalsh.collision_threshold
                                                                       : strategy.collision_threshold_;
    Require(threshold > 0 && threshold <= table_count, "default QALSH strategy has an invalid collision threshold");
    Require(threshold <= std::numeric_limits<std::uint8_t>::max(),
            "collision thresholds above 255 are not supported: the search counts collisions in one byte");
    std::optional<std::size_t> budget;
    if (strategy.candidate_budget_override_set_) {
        budget = strategy.candidate_budget_override_;
    } else {
        const std::size_t base_budget = metadata.qalsh.candidate_budget;
        const std::size_t additional = static_cast<std::size_t>(k - 1U);
        const std::size_t uncapped = base_budget > std::numeric_limits<std::size_t>::max() - additional
                                         ? std::numeric_limits<std::size_t>::max()
                                         : base_budget + additional;
        budget = std::min<std::size_t>(metadata.num_points, uncapped);
    }
    const bool check_after_evaluation = strategy.check_after_evaluation_;
    const bool continuous = strategy.continuous_stop_;
    Require(!continuous || options_.scan_order != SearchOptions::ScanOrder::quanta,
            "the continuous stop needs a projection-distance scan order");
    const float mid_round_ratio = strategy.mid_round_ratio_;
    const float approximation_ratio = metadata.qalsh.approximation_ratio;
    const std::uint32_t num_points = metadata.num_points;
    const std::uint32_t num_dimensions = metadata.num_dimensions;

    SearchScratch& scratch = search_scratch;
    auto& projections = scratch.query_projections;
    projections.resize(table_count);
    ProjectPoint(query, index.projection_vectors().data(), num_dimensions, projections);

    // Each table has two cursors, one on each side of the query's projection (at its lower bound). A round whose window
    // holds no entry in any table scans nothing and cannot end the search, since no point has collided yet. The search
    // therefore starts at the first radius of the grid R0 * growth^j whose window reaches the nearest entry of some
    // table; stepping through the skipped rounds gives the same result.
    auto& left = scratch.left;
    auto& right = scratch.right;
    left.assign(table_count, kNoEntry);
    right.assign(table_count, kNoEntry);
    scratch.left_blocked.assign(table_count, 0);
    scratch.right_blocked.assign(table_count, 0);
    scratch.last_bound.assign(table_count, -std::numeric_limits<float>::infinity());
    scratch.exhausted.assign(table_count, 0);
    float gap = std::numeric_limits<float>::infinity();
    for (std::uint32_t table = 0; table < table_count; ++table) {
        const float query_value = projections[table];
        RequireFinite(query_value, "query projection");
        const auto& entries = index.tables_[table].entries;
        Require(index.tables_[table].ids.size() == entries.size(), "in-memory table has no id column");
        const auto it = std::lower_bound(entries.begin(), entries.end(), query_value,
                                         [](const InMemoryIndex::Entry& entry, float value) {
                                             return entry.value < value;
                                         });
        const auto position = static_cast<std::size_t>(std::distance(entries.begin(), it));
        if (position != 0) {
            left[table] = position - 1;
            gap = std::min(gap, std::abs(query_value - entries[position - 1].value));
        }
        if (position != entries.size()) {
            right[table] = position;
            gap = std::min(gap, std::abs(query_value - entries[position].value));
        }
    }
    float radius = metadata.qalsh.initial_radius;
    float bound = WindowBound(metadata.qalsh.bucket_width, radius);
    while (bound < gap) {
        const float next_radius = radius * metadata.qalsh.radius_growth;
        if (!IsFinite(next_radius) || !(next_radius > radius)) break;
        radius = next_radius;
        bound = WindowBound(metadata.qalsh.bucket_width, radius);
    }

    scratch.counters.assign(num_points, static_cast<std::uint8_t>(threshold));
    std::uint8_t* const counters = scratch.counters.data();
    auto& neighbors = scratch.neighbors;
    neighbors.clear();
    neighbors.reserve(k);
    std::size_t evaluated = 0;

    const auto should_terminate = [&] {
        if (budget.has_value() && evaluated >= *budget) return true;
        return neighbors.size() >= k && neighbors.back().distance <= approximation_ratio * radius;
    };
    // With a compressed copy, candidates whose lower bound already exceeds the k-th distance skip the exact kernel. The
    // bound reads the query's coordinates in the copy's order.
    const bool compressed = !index.compressed_kind_.empty();
    auto& query_ordered = scratch.query_ordered;
    if (compressed) {
        query_ordered.resize(num_dimensions);
        for (std::size_t i = 0; i < num_dimensions; ++i) query_ordered[i] = query[index.compressed_order_[i]];
    }
    const auto compressed_rejects = [&](PointId point_id, Distance upper_bound) {
        const std::uint8_t kind = index.compressed_kind_[point_id];
        if (kind == 2) return false;
        // DistanceKernel's limit for this upper bound.
        const double expanded = std::nextafter(upper_bound, std::numeric_limits<float>::infinity());
        const double limit = metadata.metric == Metric::l2 ? expanded * expanded : expanded;
        const std::uint16_t* const target =
            index.compressed_points_.get() + static_cast<std::size_t>(point_id) * num_dimensions;
        return metadata.metric == Metric::l2
                   ? CompressedBoundRejects<Metric::l2>(query_ordered.data(), target, num_dimensions, kind == 1, limit)
                   : CompressedBoundRejects<Metric::l1>(query_ordered.data(), target, num_dimensions, kind == 1, limit);
    };
    // Evaluates a candidate, then runs the checks that follow an evaluation; true asks the search to finish.
    const auto evaluate = [&](PointId point_id) {
        const PointView point = accessor_(point_id);
        if (options_.validate_points) {
            ValidatePoint(point, num_dimensions, point_id);
        }
        Distance upper_bound = std::numeric_limits<float>::infinity();
        if (options_.bounded_distance && neighbors.size() == k) {
            upper_bound = neighbors.back().distance;
        }
        const BoundedDistance distance =
            compressed && upper_bound < std::numeric_limits<float>::infinity() &&
                    compressed_rejects(point_id, upper_bound)
                ? BoundedDistance{.distance = std::numeric_limits<float>::infinity(), .exact = false}
                : BoundedDistanceValueUnchecked(point, query, metadata.metric, upper_bound);
        ++evaluated;
        if (distance.exact) {
            InsertNeighborInto(neighbors, Neighbor{.distance = distance.distance, .point_id = point_id}, k);
        }
        if (continuous) return false;
        if (check_after_evaluation && should_terminate()) return true;
        return mid_round_ratio > 0.0F && neighbors.size() >= k &&
               neighbors.back().distance <= mid_round_ratio * radius;
    };

    std::uint64_t hits = 0;
    std::uint32_t exhausted_tables = 0;
    bool finished = false;
    const bool check_after_radius_advance = strategy.check_after_radius_advance_;
    // Between two rounds: the round-end test, the end of the search once every table is exhausted, then the radius
    // grows. With check_after_radius_advance the search tests again, against the new radius, before the next round
    // scans anything. True when the search finishes.
    const auto end_round = [&] {
        if (!continuous && should_terminate()) return true;
        if (exhausted_tables == table_count) return true;
        const float next_radius = radius * metadata.qalsh.radius_growth;
        Require(IsFinite(next_radius) && next_radius > radius, "strategy requested an invalid radius progression");
        radius = next_radius;
        bound = WindowBound(metadata.qalsh.bucket_width, radius);
        return !continuous && check_after_radius_advance && should_terminate();
    };
    // The continuous stop: the k-th distance is at most c times the radius of a front at `front` from the query's
    // projection.
    const float front_to_radius = 2.0F / metadata.qalsh.bucket_width;
    const auto front_reached = [&](float front) {
        return neighbors.size() >= k && neighbors.back().distance <= approximation_ratio * (front * front_to_radius);
    };

    switch (options_.scan_order) {
        case SearchOptions::ScanOrder::quanta: {
            Require(scan_quantum > 1, "the quanta scan order needs a scan quantum of at least two");
            const std::size_t left_limit = scan_quantum / 2U + scan_quantum % 2U;
            const std::size_t right_limit = scan_quantum / 2U;
            // Counts `count` ids of a table from `start`, downward when step is -1, and evaluates each point whose
            // counter runs out; true asks the search to finish.
            const auto count_entries = [&](std::uint32_t table, std::size_t start, std::ptrdiff_t step,
                                           std::size_t count) {
                const PointId* ids = index.tables_[table].ids.data() + start;
                for (std::size_t offset = 0; offset < count; ++offset, ids += step) {
                    const PointId point_id = *ids;
                    const std::uint8_t remaining = counters[point_id];
                    if (remaining > 1U) {
                        counters[point_id] = static_cast<std::uint8_t>(remaining - 1U);
                        continue;
                    }
                    if (remaining == 0U) continue;
                    counters[point_id] = 0;
                    if (evaluate(point_id)) return true;
                }
                return false;
            };
            for (;;) {
                auto& active = scratch.active;
                active.clear();
                for (std::uint32_t table = 0; table < table_count; ++table) {
                    if (scratch.exhausted[table] != 0) continue;
                    active.push_back(table);
                    // On the table's first visit of the round, the larger window unblocks both sides.
                    Require(bound >= scratch.last_bound[table], "scan bound cannot decrease for an existing cursor");
                    if (bound > scratch.last_bound[table]) {
                        scratch.left_blocked[table] = 0;
                        scratch.right_blocked[table] = 0;
                        scratch.last_bound[table] = bound;
                    }
                }
                while (!active.empty() && !finished) {
                    auto& next_active = scratch.next_active;
                    next_active.clear();
                    for (const std::uint32_t table : active) {
                        // One visit: up to left_limit entries on the left side, then up to right_limit on the right.
                        const auto& entries = index.tables_[table].entries;
                        const PointId* const table_ids = index.tables_[table].ids.data();
                        const auto entry_bytes = std::as_bytes(std::span<const InMemoryIndex::Entry>(entries));
                        const float query_value = projections[table];
                        std::size_t& left_position = left[table];
                        std::size_t& right_position = right[table];
                        std::size_t left_start = 0;
                        std::size_t left_count = 0;
                        std::size_t right_start = 0;
                        std::size_t right_count = 0;
                        if (left_position != kNoEntry && scratch.left_blocked[table] == 0) {
                            const std::size_t available = std::min(left_limit, left_position + 1U);
                            detail::HitRange range{entry_bytes.data() + left_position * sizeof(InMemoryIndex::Entry),
                                                   available, true, table, query_value};
                            window_detail::ClampToWindow(range, bound);
                            scratch.left_blocked[table] = range.count < available ? 1 : 0;
                            left_start = left_position;
                            left_count = range.count;
                            left_position =
                                range.count == left_position + 1U ? kNoEntry : left_position - range.count;
                        }
                        if (right_position != kNoEntry && scratch.right_blocked[table] == 0) {
                            const std::size_t available = std::min(right_limit, entries.size() - right_position);
                            detail::HitRange range{entry_bytes.data() + right_position * sizeof(InMemoryIndex::Entry),
                                                   available, false, table, query_value};
                            window_detail::ClampToWindow(range, bound);
                            scratch.right_blocked[table] = range.count < available ? 1 : 0;
                            right_start = right_position;
                            right_count = range.count;
                            right_position = right_position + range.count == entries.size()
                                                 ? kNoEntry
                                                 : right_position + range.count;
                        }
                        const bool table_exhausted = left_position == kNoEntry && right_position == kNoEntry;
                        const bool window_exhausted =
                            (left_position == kNoEntry || scratch.left_blocked[table] != 0) &&
                            (right_position == kNoEntry || scratch.right_blocked[table] != 0);
                        // The table's next visit, one cycle later, reads the next ids of each side that can still
                        // advance in this round and probes the window at their far end. Interleaved over all tables,
                        // these short runs escape the hardware prefetcher once the id columns outgrow the caches.
                        if (left_position != kNoEntry && scratch.left_blocked[table] == 0) {
                            const std::size_t span = std::min(left_limit, left_position + 1U);
                            const std::size_t far = left_position + 1U - span;
                            for (std::size_t position = far; position <= left_position; position += kIdsPerLine) {
                                __builtin_prefetch(table_ids + position);
                            }
                            __builtin_prefetch(table_ids + left_position);
                            __builtin_prefetch(entries.data() + far);
                        }
                        if (right_position != kNoEntry && scratch.right_blocked[table] == 0) {
                            const std::size_t span = std::min(right_limit, entries.size() - right_position);
                            const std::size_t last = right_position + span - 1U;
                            for (std::size_t position = right_position; position <= last; position += kIdsPerLine) {
                                __builtin_prefetch(table_ids + position);
                            }
                            __builtin_prefetch(table_ids + last);
                            __builtin_prefetch(entries.data() + last);
                        }
                        if (table_exhausted && scratch.exhausted[table] == 0) {
                            scratch.exhausted[table] = 1;
                            ++exhausted_tables;
                        }
                        if (!window_exhausted && !table_exhausted) {
                            next_active.push_back(table);
                        }
                        // Every entry of the visit counts as a hit, including those after a stop.
                        hits += left_count + right_count;
                        if ((left_count != 0 && count_entries(table, left_start, -1, left_count)) ||
                            (right_count != 0 && count_entries(table, right_start, 1, right_count))) {
                            finished = true;
                            break;
                        }
                    }
                    std::swap(active, next_active);
                }
                if (finished || end_round()) break;
            }
            break;
        }
        case SearchOptions::ScanOrder::projection_distance: {
            // Every table side advances to the same distance from the query's projection. A round's distance range
            // is cut into equal slabs, and a slab's entries on one table side form a single run, counted in one pass.
            // A point whose counter runs out in a slab has its collision radius, the threshold-th smallest distance
            // between its projections and the query's, inside that slab. Evaluating the slab's new candidates in
            // increasing collision radius, ties by id, is the order of projection_distance_reference, which processes
            // the entries one distance at a time and evaluates the candidates of one distance in increasing id.
            Require(index.point_projections_ != nullptr,
                    "the projection-distance scan order needs an index built with point projections");
            const std::uint32_t slabs = std::max<std::uint32_t>(1, options_.slabs_per_round);
            const std::size_t projection_stride = metadata.num_hash_tables;
            auto& events = scratch.distance_events;
            auto& gaps = scratch.gaps;
            gaps.resize(table_count);
            // Entries from `position` outward (downward when `reverse`) within `limit` of the query's projection:
            // ClampToWindow, the window test of the quanta order, on ranges that double until one ends inside.
            // ClampToWindow first reads a range's far end, so the first range is short: most runs of a slab are,
            // and a far end in a cold cache line would cost a miss for nothing.
            const auto run_length = [&](std::uint32_t table, std::size_t position, bool reverse, float limit) {
                const auto& entries = index.tables_[table].entries;
                const auto entry_bytes = std::as_bytes(std::span<const InMemoryIndex::Entry>(entries));
                std::size_t total = 0;
                std::size_t step = 8;
                for (;;) {
                    const std::size_t remaining = reverse ? position + 1U : entries.size() - position;
                    const std::size_t available = std::min(step, remaining);
                    detail::HitRange range{entry_bytes.data() + position * sizeof(InMemoryIndex::Entry), available,
                                           reverse, table, projections[table]};
                    window_detail::ClampToWindow(range, limit);
                    total += range.count;
                    if (range.count < available || range.count == remaining) return total;
                    position = reverse ? position - range.count : position + range.count;
                    step *= 2;
                }
            };
            // Counts a run and records each point whose counter runs out.
            const auto count_run = [&](const PointId* ids, std::ptrdiff_t step, std::size_t count) {
                for (std::size_t offset = 0; offset < count; ++offset, ids += step) {
                    const PointId point_id = *ids;
                    const std::uint8_t remaining = counters[point_id];
                    if (remaining > 1U) {
                        counters[point_id] = static_cast<std::uint8_t>(remaining - 1U);
                        continue;
                    }
                    if (remaining == 0U) continue;
                    counters[point_id] = 0;
                    events.emplace_back(0.0F, point_id);
                }
            };
            float done = 0.0F;  // distances up to here belong to earlier rounds
            for (;;) {
                float slab_bound = done;
                for (std::uint32_t slab = 1; slab <= slabs && !finished; ++slab) {
                    const float next = slab == slabs ? bound
                                                     : done + (bound - done) * (static_cast<float>(slab) /
                                                                                static_cast<float>(slabs));
                    const float previous_bound = slab_bound;
                    slab_bound = std::clamp(next, slab_bound, bound);
                    // Under the continuous stop the search ends once the front reaches the k-th distance over c, so
                    // the slab need not reach past that front. Entries beyond it would only add hits, since their
                    // candidates are never evaluated, and a later slab takes any entry that rounding leaves out.
                    if (continuous && neighbors.size() >= k) {
                        const float stop_front = neighbors.back().distance / (approximation_ratio * front_to_radius);
                        if (stop_front > previous_bound && stop_front < slab_bound) slab_bound = stop_front;
                    }
                    events.clear();
                    for (std::uint32_t table = 0; table < table_count; ++table) {
                        if (scratch.exhausted[table] != 0) continue;
                        const PointId* const table_ids = index.tables_[table].ids.data();
                        std::size_t& left_position = left[table];
                        std::size_t& right_position = right[table];
                        if (left_position != kNoEntry) {
                            const std::size_t count = run_length(table, left_position, true, slab_bound);
                            count_run(table_ids + left_position, -1, count);
                            hits += count;
                            left_position = count == left_position + 1U ? kNoEntry : left_position - count;
                        }
                        if (right_position != kNoEntry) {
                            const std::size_t count = run_length(table, right_position, false, slab_bound);
                            count_run(table_ids + right_position, 1, count);
                            hits += count;
                            right_position = right_position + count == index.tables_[table].entries.size()
                                                 ? kNoEntry
                                                 : right_position + count;
                        }
                        if (left_position == kNoEntry && right_position == kNoEntry) {
                            scratch.exhausted[table] = 1;
                            ++exhausted_tables;
                        }
                    }
                    if (events.empty()) {
                        if (continuous && front_reached(slab_bound)) finished = true;
                        continue;
                    }
                    for (auto& [collision_radius, point_id] : events) {
                        const Projection* const row =
                            index.point_projections_.get() + static_cast<std::size_t>(point_id) * projection_stride;
                        for (std::uint32_t table = 0; table < table_count; ++table) {
                            gaps[table] = std::abs(projections[table] - row[table]);
                        }
                        std::nth_element(gaps.begin(), gaps.begin() + (threshold - 1U), gaps.end());
                        collision_radius = gaps[threshold - 1U];
                    }
                    std::sort(events.begin(), events.end());
                    for (const auto& event : events) {
                        // Under the continuous stop, the front may already have reached the stopping radius.
                        if ((continuous && front_reached(event.first)) || evaluate(event.second) ||
                            (continuous && front_reached(event.first))) {
                            finished = true;
                            break;
                        }
                    }
                    if (!finished && continuous && front_reached(slab_bound)) finished = true;
                }
                if (finished || end_round()) break;
                done = slab_bound;
            }
            break;
        }
        case SearchOptions::ScanOrder::projection_distance_reference: {
            // One distance at a time: all entries at the smallest distance from the query's projection among the 2m
            // cursors, then the points that became candidates at that distance, in increasing id.
            struct Front {
                float distance;
                PointId point_id;
                std::uint32_t table;
                bool reverse;
                std::size_t position;
            };
            const auto later = [](const Front& lhs, const Front& rhs) {
                return lhs.distance != rhs.distance ? lhs.distance > rhs.distance : lhs.point_id > rhs.point_id;
            };
            std::vector<Front> heap;
            const auto push = [&](std::uint32_t table, bool reverse, std::size_t position) {
                if (position == kNoEntry) return;
                const auto& entries = index.tables_[table].entries;
                heap.push_back(Front{.distance = std::abs(projections[table] - entries[position].value),
                                     .point_id = index.tables_[table].ids[position],
                                     .table = table,
                                     .reverse = reverse,
                                     .position = position});
                std::push_heap(heap.begin(), heap.end(), later);
            };
            for (std::uint32_t table = 0; table < table_count; ++table) {
                push(table, true, left[table]);
                push(table, false, right[table]);
            }
            std::vector<PointId> level_candidates;
            for (;;) {
                while (!finished && !heap.empty() && heap.front().distance <= bound) {
                    const float level = heap.front().distance;
                    if (continuous && front_reached(level)) {
                        finished = true;
                        break;
                    }
                    level_candidates.clear();
                    while (!heap.empty() && heap.front().distance == level) {
                        std::pop_heap(heap.begin(), heap.end(), later);
                        const Front front = heap.back();
                        heap.pop_back();
                        ++hits;
                        const std::size_t size = index.tables_[front.table].entries.size();
                        push(front.table, front.reverse,
                             front.reverse ? (front.position == 0 ? kNoEntry : front.position - 1U)
                                           : (front.position + 1U == size ? kNoEntry : front.position + 1U));
                        const std::uint8_t remaining = counters[front.point_id];
                        if (remaining > 1U) {
                            counters[front.point_id] = static_cast<std::uint8_t>(remaining - 1U);
                            continue;
                        }
                        if (remaining == 0U) continue;
                        counters[front.point_id] = 0;
                        level_candidates.push_back(front.point_id);
                    }
                    std::sort(level_candidates.begin(), level_candidates.end());
                    for (const PointId point_id : level_candidates) {
                        if (evaluate(point_id) || (continuous && front_reached(level))) {
                            finished = true;
                            break;
                        }
                    }
                }
                if (heap.empty()) exhausted_tables = table_count;
                if (finished || end_round()) break;
            }
            break;
        }
    }
    finished = true;

    const TerminationReason reason =
        exhausted_tables == table_count ? TerminationReason::scan_exhausted : TerminationReason::strategy;
    return SearchResult{.neighbors = neighbors,
                        .reason = reason,
                        .evaluated_candidates = evaluated,
                        .projection_hits = static_cast<std::size_t>(hits),
                        .complete = finished && neighbors.size() == k};
}

const char* ToString(TerminationReason reason) noexcept {
    switch (reason) {
        case TerminationReason::strategy:
            return "strategy";
        case TerminationReason::scan_exhausted:
            return "scan_exhausted";
    }
    return "unknown";
}

}  // namespace qalsh
