#include "ann_searcher.h"

#include <atomic>
#include <cstdio>

#ifdef QALSH4C_USE_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <stdexcept>
#include <vector>

#include "exact_scan.h"
#include "global.h"
#include "log.h"
#include "utils.h"

namespace {

bool IsBetterAnnResult(const AnnResult& candidate, const AnnResult& best) {
    return candidate.distance < best.distance ||
           (candidate.distance == best.distance && candidate.point_id < best.point_id);
}

qalsh::Metric ToLibraryMetric(float norm_order) {
    if (std::abs(norm_order - 1.0F) < Global::kEpsilon) {
        return qalsh::Metric::l1;
    }
    if (std::abs(norm_order - 2.0F) < Global::kEpsilon) {
        return qalsh::Metric::l2;
    }
    throw std::invalid_argument("unsupported norm order");
}

// The bfloat16 candidate filter leaves results unchanged; QALSH4C_COMPRESSED_FILTER=0 switches it off for checks.
bool UseCompressedFilter() {
    const char* value = std::getenv("QALSH4C_COMPRESSED_FILTER");
    return value == nullptr || std::string(value) != "0";
}

qalsh::IndexConfig MakeLibraryConfig(const QalshConfig& config, float norm_order, unsigned int num_points,
                                     unsigned int num_dimensions, unsigned int seed) {
    const qalsh::Metric metric = ToLibraryMetric(norm_order);
    unsigned int build_threads = 1;
#ifdef QALSH4C_USE_OPENMP
    build_threads = static_cast<unsigned int>(std::max(1, omp_get_max_threads()));
#endif
    return qalsh::IndexConfig{.metric = metric,
                              .num_points = num_points,
                              .num_dimensions = num_dimensions,
                              .qalsh = qalsh::QalshParameters{.approximation_ratio = config.approximation_ratio,
                                                               .bucket_width = config.bucket_width,
                                                               .error_probability = config.error_probability,
                                                               .num_hash_tables = config.num_hash_tables,
                                                               .collision_threshold = config.collision_threshold,
                                                               .candidate_budget = Global::kNumCandidates,
                                                               .initial_radius = Global::kInitialRadius,
                                                               .radius_growth = config.approximation_ratio,
                                                               // The legacy scan budget applies independently to both sides;
                                                               // the library's quantum is the combined logical budget.
                                                               .scan_quantum = Global::kQalshScanQuantum},
                              .seed = seed,
                              .projection_vectors = {},
                              .build_threads = build_threads,
                              // The in-memory searcher's accessor returns views of immutable loaded vectors.
                              .concurrent_accessor = true,
                              .compressed_filter = UseCompressedFilter(),
                              .point_projections = Global::kQalshScanOrder != "quanta"};
}


qalsh::SearchOptions::ScanOrder ToScanOrder(const std::string& order) {
    if (order == "distance") return qalsh::SearchOptions::ScanOrder::projection_distance;
    if (order == "distance-reference") return qalsh::SearchOptions::ScanOrder::projection_distance_reference;
    return qalsh::SearchOptions::ScanOrder::quanta;
}

AnnResult ToAnnResult(const qalsh::SearchResult& result) {
    if (result.neighbors.empty()) {
        throw std::runtime_error(std::string("QALSH returned no result; termination reason: ") +
                                 qalsh::ToString(result.reason));
    }
    return AnnResult{.distance = result.neighbors.front().distance,
                     .point_id = result.neighbors.front().point_id};
}

}  // namespace

// ---------------------------------------------
// LinearScanAnnSearcher Implementation
// ---------------------------------------------
bool LinearScanAnnSearcher::RequiresIndex() const { return false; }

std::unique_ptr<AnnSearcher> LinearScanAnnSearcher::Clone() const {
    auto clone = std::make_unique<LinearScanAnnSearcher>();
    clone->norm_order_ = norm_order_;
    clone->num_points_ = num_points_;
    clone->num_dimensions_ = num_dimensions_;
    clone->base_points_ = base_points_;
    return clone;
}

void LinearScanAnnSearcher::Init(const PointSetMetadata& metadata, float norm_order) {
    auto points = std::make_shared<std::vector<Point>>(
        PointSet::LoadPointsFromFile(metadata.file_path, metadata.num_points, metadata.num_dimensions));
    if (points->size() != metadata.num_points) {
        Log::error("Failed to load all target points for in-memory linear scan.");
    }
    InitLoaded(std::move(points), metadata.num_dimensions, norm_order);
}

void LinearScanAnnSearcher::InitLoaded(std::shared_ptr<const std::vector<Point>> base_points,
                                       unsigned int num_dimensions, float norm_order) {
    if (!base_points) {
        Log::error("Loaded target points are not set.");
    }
    norm_order_ = norm_order;
    num_points_ = static_cast<unsigned int>(base_points->size());
    num_dimensions_ = num_dimensions;
    base_points_ = std::move(base_points);
}

AnnResult LinearScanAnnSearcher::Search(const Point& query_point) {
    return SearchWithUpperBound(query_point, std::numeric_limits<float>::infinity());
}

AnnResult LinearScanAnnSearcher::SearchWithUpperBound(const Point& query_point, Distance initial_upper_bound) {
    if (!base_points_) {
        Log::error("Linear scan searcher is not initialized.");
    }
    AnnResult result{.distance = initial_upper_bound, .point_id = 0};
    for (unsigned int point_id = 0; point_id < num_points_; ++point_id) {
        const float distance = Calc::LpDistanceBounded((*base_points_)[point_id], query_point, norm_order_, result.distance);
        const AnnResult candidate{.distance = distance, .point_id = point_id};
        if (IsBetterAnnResult(candidate, result)) {
            result = candidate;
        }
    }
    return result;
}

std::vector<AnnResult> LinearScanAnnSearcher::SearchBatchWithUpperBounds(
    const std::vector<Point>& query_points, const std::vector<Distance>& initial_upper_bounds) {
    if (query_points.size() != initial_upper_bounds.size()) {
        Log::error("Query size and upper-bound size mismatch.");
    }
    if (!base_points_) {
        Log::error("Linear scan searcher is not initialized.");
    }
    std::vector<AnnResult> results(query_points.size());
    for (std::size_t query_index = 0; query_index < query_points.size(); ++query_index) {
        results[query_index] = AnnResult{.distance = initial_upper_bounds[query_index], .point_id = 0};
    }
    for (unsigned int point_id = 0; point_id < num_points_; ++point_id) {
        for (std::size_t query_index = 0; query_index < query_points.size(); ++query_index) {
            const float distance = Calc::LpDistanceBounded((*base_points_)[point_id], query_points[query_index],
                                                            norm_order_, results[query_index].distance);
            const AnnResult candidate{.distance = distance, .point_id = point_id};
            if (IsBetterAnnResult(candidate, results[query_index])) {
                results[query_index] = candidate;
            }
        }
    }
    return results;
}

bool LinearScanAnnSearcher::UsesBatchedScan() const {
    return !Global::kUseScalarExactScan && ExactScan::Supports(norm_order_);
}

std::vector<AnnResult> LinearScanAnnSearcher::SearchBatched(
    const std::vector<Point>& query_points, const std::vector<Distance>& initial_upper_bounds) const {
    if (!base_points_) {
        Log::error("Linear scan searcher is not initialized.");
    }
    std::vector<const Coordinate*> queries;
    queries.reserve(query_points.size());
    for (const Point& query_point : query_points) {
        if (query_point.size() != num_dimensions_) {
            Log::error("Query point dimension {} does not match target dimension {}.", query_point.size(),
                       num_dimensions_);
        }
        queries.push_back(query_point.data());
    }
    ExactScan scan(queries, num_dimensions_, norm_order_, initial_upper_bounds);
    scan.ScanPoints(*base_points_);
    return scan.Results();
}

// ---------------------------------------------
// QalshAnnSearcher Implementation
// ---------------------------------------------
QalshAnnSearcher::QalshAnnSearcher(float approximation_ratio)
    : QalshAnnSearcher(approximation_ratio, Random::SeededGenerator()()) {}

QalshAnnSearcher::QalshAnnSearcher(float approximation_ratio, std::mt19937::result_type random_seed)
    : approximation_ratio_(approximation_ratio), random_seed_(random_seed) {}

bool QalshAnnSearcher::RequiresIndex() const { return false; }

std::unique_ptr<AnnSearcher> QalshAnnSearcher::Clone() const {
    auto clone = std::make_unique<QalshAnnSearcher>(approximation_ratio_, random_seed_);
    clone->index_data_ = index_data_;
    if (clone->index_data_) {
        qalsh::SearchOptions search_options;
        // Build checked every target point, and the accessor returns those same immutable vectors.
        search_options.validate_points = false;
        search_options.scan_order = ToScanOrder(Global::kQalshScanOrder);
        search_options.slabs_per_round = Global::kQalshSlabsPerRound;
        clone->engine_ = std::make_unique<qalsh::SearchEngine>(
            clone->index_data_->index,
            [points = clone->index_data_->base_points](qalsh::PointId point_id) -> qalsh::PointView {
                return (*points)[point_id];
            },
            search_options);
    }
    return clone;
}

namespace {
// Projection hits and evaluated candidates summed over all in-memory QALSH queries, printed to stderr at exit.
struct QalshHitCounter {
    std::atomic<unsigned long long> queries{0}, hits{0}, evaluated{0};
    std::atomic<unsigned long long> tables{0}, points{0};
    ~QalshHitCounter() {
        const unsigned long long q = queries.load();
        if (q == 0) return;
        const double per_query = static_cast<double>(hits.load()) / static_cast<double>(q);
        const double entries = static_cast<double>(tables.load()) * static_cast<double>(points.load());
        std::fprintf(stderr, "QALSH stats: queries=%llu m=%llu n=%llu hits_per_query=%.0f fraction_of_entries=%.4f "
                     "evaluated_per_query=%.1f\n", q, tables.load(), points.load(), per_query, per_query / entries,
                     static_cast<double>(evaluated.load()) / static_cast<double>(q));
    }
};
QalshHitCounter g_qalsh_hit_counter;
}  // namespace

std::shared_ptr<const std::vector<Point>> QalshAnnSearcher::TargetPoints() const {
    return index_data_ ? index_data_->base_points : std::shared_ptr<const std::vector<Point>>{};
}

void QalshAnnSearcher::Init(const PointSetMetadata& metadata, float norm_order) {
    auto data = std::make_shared<QalshAnnIndexData>();
    data->norm_order = norm_order;
    data->num_points = metadata.num_points;
    data->num_dimensions = metadata.num_dimensions;
    if (data->num_points == 0 || data->num_dimensions == 0) {
        Log::error("QALSH requires a non-empty point set with non-zero dimensions.");
    }
    data->qalsh_config = QalshConfig{.approximation_ratio = approximation_ratio_};
    data->qalsh_config.Regularize(data->num_points, norm_order);
    const auto load_start = std::chrono::steady_clock::now();
    auto points = std::make_shared<std::vector<Point>>(
        PointSet::LoadPointsFromFile(metadata.file_path, data->num_points, data->num_dimensions));
    if (points->size() != data->num_points) {
        Log::error("Failed to load all target points for in-memory QALSH index.");
    }
    data->base_points = points;
    const auto build_start = std::chrono::steady_clock::now();
    const qalsh::IndexConfig config = MakeLibraryConfig(data->qalsh_config, norm_order, data->num_points,
                                                         data->num_dimensions, random_seed_);
    const qalsh::PointAccessor accessor = [points](qalsh::PointId point_id) -> qalsh::PointView {
        return (*points)[point_id];
    };
    data->index = qalsh::InMemoryIndex::Build(config, accessor);
    Log::info("QALSH index times: load_ms={:.3f} build_ms={:.3f}",
              std::chrono::duration<double, std::milli>(build_start - load_start).count(),
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_start).count());
    index_data_ = std::move(data);
    qalsh::SearchOptions search_options;
    // Build checked every target point, and the accessor returns those same immutable vectors.
    search_options.validate_points = false;
    search_options.scan_order = ToScanOrder(Global::kQalshScanOrder);
    search_options.slabs_per_round = Global::kQalshSlabsPerRound;
    engine_ = std::make_unique<qalsh::SearchEngine>(
        index_data_->index,
        [points = index_data_->base_points](qalsh::PointId point_id) -> qalsh::PointView {
            return (*points)[point_id];
        },
        search_options);
}

AnnResult QalshAnnSearcher::Search(const Point& query_point) {
    if (!index_data_) {
        Log::error("QALSH searcher is not initialized.");
    }
    if (!engine_) {
        Log::error("QALSH searcher is not initialized.");
    }
    qalsh::DefaultQalshStrategy strategy(index_data_->qalsh_config.collision_threshold);
    strategy.set_candidate_budget(std::nullopt);
    strategy.set_check_after_evaluation(!Global::kQalshStopAtRoundBoundary);
    strategy.set_mid_round_ratio(Global::kQalshStopAtRoundBoundary ? Global::kQalshMidRoundRatio : 0.0F);
    strategy.set_check_after_radius_advance(Global::kQalshCheckAfterRadiusAdvance);
    strategy.set_continuous_stop(Global::kQalshStopRule == "continuous");
    const qalsh::SearchResult result = engine_->search(query_point, 1, strategy);
    g_qalsh_hit_counter.queries += 1;
    g_qalsh_hit_counter.hits += result.projection_hits;
    g_qalsh_hit_counter.evaluated += result.evaluated_candidates;
    g_qalsh_hit_counter.tables = index_data_->index->metadata().num_hash_tables;
    g_qalsh_hit_counter.points = index_data_->num_points;
    return ToAnnResult(result);
}

double QalshStoppingFactor(float approximation_ratio) {
    const double c = approximation_ratio;
    if (Global::kQalshStopRule == "continuous") {
        // The continuous front returns once the best distance is at most c times the radius of the front.
        return c;
    }
    // Round rules with radii R_t = R_0 c^t. A test right after the radius grows, or on every new candidate, may accept a
    // candidate within c R_t while only round t - 1, of radius R_t / c, is complete.
    if (!Global::kQalshStopAtRoundBoundary || Global::kQalshCheckAfterRadiusAdvance) {
        return c * c;
    }
    // A test at the end of round t accepts candidates within c R_t once round t is complete; a stop inside round t
    // accepts a candidate within ratio * R_t while round t - 1 is complete.
    return c * std::max(1.0, static_cast<double>(Global::kQalshMidRoundRatio));
}

kappa::Parameters MakeKappaParameters(const QalshConfig& config, float norm_order) {
    if (Global::kQalshStopRule != "continuous" && !(Global::kInitialRadius <= 0x1p-149F)) {
        // The round rules need R_0 at most every positive d(a, B); 2^-149 is at most any positive float32 distance.
        Log::error("The kappa bound of the round rules needs --initial-radius at most 2^-149 (1.4e-45).");
    }
    return kappa::Parameters{.norm = std::abs(norm_order - 1.0F) < Global::kEpsilon ? 1 : 2,
                             .bucket_width = static_cast<double>(config.bucket_width),
                             .num_hash_tables = config.num_hash_tables,
                             .collision_threshold = config.collision_threshold,
                             .gamma = QalshStoppingFactor(config.approximation_ratio)};
}
