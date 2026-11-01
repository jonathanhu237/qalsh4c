#include "ann_searcher.h"

#ifdef QALSH4C_USE_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "disk_qalsh_index.h"
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
                                                               .initial_radius = 1.0F,
                                                               .radius_growth = config.approximation_ratio,
                                                               // The legacy scan budget applies independently to both sides;
                                                               // the library's quantum is the combined logical budget.
                                                               .scan_quantum = Global::kScanSize * 2},
                              .seed = seed,
                              .projection_vectors = {},
                              .build_threads = build_threads};
}

void ReadPointAt(int fd, Point& buffer, unsigned int point_id, unsigned int dimensions) {
    const std::uint64_t offset = static_cast<std::uint64_t>(point_id) * dimensions * sizeof(Coordinate);
    auto* destination = reinterpret_cast<char*>(buffer.data());
    std::size_t total = 0;
    const std::size_t bytes = static_cast<std::size_t>(dimensions) * sizeof(Coordinate);
    while (total < bytes) {
        const ssize_t count = ::pread(fd, destination + total, bytes - total,
                                      static_cast<off_t>(offset + total));
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            throw std::runtime_error("failed to read target point");
        }
        total += static_cast<std::size_t>(count);
    }
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
// DiskBasedLinearScanAnnSearcher Implementation
// ---------------------------------------------
DiskBasedLinearScanAnnSearcher::DiskBasedLinearScanAnnSearcher(unsigned int block_size) : block_size_(block_size) {
    if (block_size_ == 0) {
        Log::error("Disk block size must be positive.");
    }
}

bool DiskBasedLinearScanAnnSearcher::RequiresIndex() const { return false; }

std::unique_ptr<AnnSearcher> DiskBasedLinearScanAnnSearcher::Clone() const {
    auto clone = std::make_unique<DiskBasedLinearScanAnnSearcher>(block_size_);
    if (initialized_) {
        clone->Init(metadata_, norm_order_);
    }
    return clone;
}

void DiskBasedLinearScanAnnSearcher::Init(const PointSetMetadata& metadata, float norm_order) {
    if (metadata.num_points == 0 || metadata.num_dimensions == 0) {
        Log::error("Disk-based linear scan requires a non-empty point set with non-zero dimensions.");
    }
    metadata_ = metadata;
    norm_order_ = norm_order;
    initialized_ = true;
    block_buffer_.clear();
    block_buffer_.reserve(static_cast<std::size_t>(block_size_) * metadata_.num_dimensions);
}

AnnResult DiskBasedLinearScanAnnSearcher::Search(const Point& query_point) {
    return SearchWithUpperBound(query_point, std::numeric_limits<float>::infinity());
}

AnnResult DiskBasedLinearScanAnnSearcher::SearchWithUpperBound(const Point& query_point,
                                                               Distance initial_upper_bound) {
    if (!initialized_) {
        Log::error("Disk-based linear scan searcher is not initialized.");
    }
    if (query_point.size() != metadata_.num_dimensions) {
        Log::error("Query point dimension {} does not match target dimension {}.", query_point.size(),
                   metadata_.num_dimensions);
    }
    std::ifstream input(metadata_.file_path, std::ios::binary);
    if (!input.is_open()) {
        Log::error("Could not open point set file for disk-based linear scan: {}.", metadata_.file_path.string());
    }
    AnnResult best{.distance = initial_upper_bound, .point_id = 0};
    for (unsigned int block_begin = 0; block_begin < metadata_.num_points; block_begin += block_size_) {
        const unsigned int count = std::min(block_size_, metadata_.num_points - block_begin);
        const std::size_t coordinates = static_cast<std::size_t>(count) * metadata_.num_dimensions;
        block_buffer_.resize(coordinates);
        input.read(reinterpret_cast<char*>(block_buffer_.data()),
                   static_cast<std::streamsize>(coordinates * sizeof(Coordinate)));
        if (input.gcount() != static_cast<std::streamsize>(coordinates * sizeof(Coordinate))) {
            Log::error("Failed to read target block starting at point {} from {}.", block_begin,
                       metadata_.file_path.string());
        }
        for (unsigned int local = 0; local < count; ++local) {
            const qalsh::PointView point(block_buffer_.data() + static_cast<std::size_t>(local) * metadata_.num_dimensions,
                                  metadata_.num_dimensions);
            const float distance = qalsh::BoundedDistanceValue(point, query_point, ToLibraryMetric(norm_order_),
                                                                best.distance)
                                       .distance;
            const AnnResult candidate{.distance = distance, .point_id = block_begin + local};
            if (IsBetterAnnResult(candidate, best)) {
                best = candidate;
            }
        }
    }
    return best;
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
        clone->engine_ = std::make_unique<qalsh::SearchEngine>(
            clone->index_data_->index,
            [points = clone->index_data_->base_points](qalsh::PointId point_id) -> qalsh::PointView {
                return (*points)[point_id];
            },
            search_options);
    }
    return clone;
}

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
    auto points = std::make_shared<std::vector<Point>>(
        PointSet::LoadPointsFromFile(metadata.file_path, data->num_points, data->num_dimensions));
    if (points->size() != data->num_points) {
        Log::error("Failed to load all target points for in-memory QALSH index.");
    }
    data->base_points = points;
    const qalsh::IndexConfig config = MakeLibraryConfig(data->qalsh_config, norm_order, data->num_points,
                                                         data->num_dimensions, random_seed_);
    const qalsh::PointAccessor accessor = [points](qalsh::PointId point_id) -> qalsh::PointView {
        return (*points)[point_id];
    };
    data->index = qalsh::InMemoryIndex::Build(config, accessor);
    index_data_ = std::move(data);
    qalsh::SearchOptions search_options;
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
    const qalsh::SearchResult result = engine_->search(query_point, 1, strategy);
    return ToAnnResult(result);
}

// ---------------------------------------------
// DiskBasedQalshAnnSearcher Implementation
// ---------------------------------------------
void DiskQalshProfileSnapshot::Merge(const DiskQalshProfileSnapshot& other) {
    queries += other.queries;
    table_entry_reads += other.table_entry_reads;
    table_block_reads += other.table_block_reads;
    table_cache_hits += other.table_cache_hits;
    table_cache_misses += other.table_cache_misses;
    candidate_point_reads += other.candidate_point_reads;
    candidates_evaluated += other.candidates_evaluated;
    total_search_time_ms += other.total_search_time_ms;
}

DiskBasedQalshAnnSearcher::DiskBasedQalshAnnSearcher(std::optional<float> expected_approximation_ratio,
                                                     DiskQalshSearchOptions options)
    : expected_approximation_ratio_(expected_approximation_ratio), options_(options) {
    if (options_.table_block_size == 0 || options_.table_cache_blocks == 0) {
        Log::error("Disk QALSH table options must be positive.");
    }
    if (options_.table_block_size != Global::kDefaultDiskQalshTableBlockSize ||
        options_.table_cache_blocks != Global::kDefaultDiskQalshTableCacheBlocks) {
        Log::info("Disk QALSH table block/cache options are legacy flat-table settings; "
                  "the selected library backend uses its own immutable layout and does not apply them.");
    }
}

DiskBasedQalshAnnSearcher::~DiskBasedQalshAnnSearcher() {
    if (target_fd_ >= 0) {
        ::close(target_fd_);
    }
}

bool DiskBasedQalshAnnSearcher::RequiresIndex() const { return true; }

std::unique_ptr<AnnSearcher> DiskBasedQalshAnnSearcher::Clone() const {
    auto clone = std::make_unique<DiskBasedQalshAnnSearcher>(expected_approximation_ratio_, options_);
    if (index_data_) {
        // The persistent index and its immutable projection vectors are shared. Only
        // the caller-owned point file descriptor, scratch buffer, and query
        // engine are per-clone state; reopening would repeat full validation
        // and mapping for every worker.
        clone->InitializeFromIndexData(index_data_);
    }
    return clone;
}

void DiskBasedQalshAnnSearcher::PreflightIndex(const PointSetMetadata& metadata, float norm_order) const {
    // Metadata/preflight checks stay outside the measured query setup.  The
    // immutable index is opened and structurally validated once by Init;
    // doing a second full traversal here only duplicated work.
    (void)DiskQalsh::ValidateIndex(metadata, norm_order, expected_approximation_ratio_, false);
}

bool DiskBasedQalshAnnSearcher::ProfilingEnabled() const { return options_.profile; }
void DiskBasedQalshAnnSearcher::ResetProfile() {
    std::lock_guard lock(profile_mutex_);
    profile_ = DiskQalshProfileSnapshot{};
}
void DiskBasedQalshAnnSearcher::MergeProfile(const DiskQalshProfileSnapshot& snapshot) {
    std::lock_guard lock(profile_mutex_);
    profile_.Merge(snapshot);
}
DiskQalshProfileSnapshot DiskBasedQalshAnnSearcher::ProfileSnapshot() const {
    std::lock_guard lock(profile_mutex_);
    return profile_;
}

void DiskBasedQalshAnnSearcher::PrintProfile() const {
    if (!options_.profile) {
        return;
    }
    const DiskQalshProfileSnapshot snapshot = ProfileSnapshot();
    Log::info("Disk QALSH profile: queries={}, total_search_time_ms={:.3f}, table_entry_reads={}, "
              "table_block_reads=0, table_cache_hits=0, table_cache_misses=0, "
              "candidate_point_reads={}, candidates_evaluated={} (layout traversal is library-owned; "
              "legacy flat-table block/cache counters are unsupported and intentionally not physical I/O).",
              snapshot.queries, snapshot.total_search_time_ms, snapshot.table_entry_reads,
              snapshot.candidate_point_reads, snapshot.candidates_evaluated);
}

void DiskBasedQalshAnnSearcher::InitializeFromIndexData(
    std::shared_ptr<const DiskBasedQalshIndexData> data) {
    if (!data || !data->index) {
        throw std::invalid_argument("disk QALSH index data is not initialized");
    }
    if (target_fd_ >= 0) {
        (void)::close(target_fd_);
        target_fd_ = -1;
    }
    target_fd_ = ::open(data->target_metadata.file_path.c_str(), O_RDONLY);
    if (target_fd_ < 0) {
        Log::error("Failed to open target point set for disk QALSH: {} ({})",
                   data->target_metadata.file_path.string(), std::strerror(errno));
    }
    index_data_ = std::move(data);

    qalsh::SearchOptions search_options;
    search_options.scan_quantum = index_data_->index->metadata().qalsh.scan_quantum;
    engine_ = std::make_unique<qalsh::SearchEngine>(
        index_data_->index,
        [this](qalsh::PointId point_id) -> qalsh::PointView {
            thread_local Point candidate_buffer;
            candidate_buffer.resize(index_data_->target_metadata.num_dimensions);
            ReadPointAt(target_fd_, candidate_buffer, point_id,
                        index_data_->target_metadata.num_dimensions);
            return candidate_buffer;
        },
        search_options);
}

void DiskBasedQalshAnnSearcher::Init(const PointSetMetadata& metadata, float norm_order) {
    if (metadata.num_points == 0 || metadata.num_dimensions == 0) {
        Log::error("Disk QALSH requires a non-empty target point set with non-zero dimensions.");
    }
    const DiskQalsh::IndexMetadata disk_metadata =
        DiskQalsh::ValidateIndex(metadata, norm_order, expected_approximation_ratio_, false);
    auto data = std::make_shared<DiskBasedQalshIndexData>();
    data->norm_order = norm_order;
    data->target_metadata = metadata;
    data->metadata = disk_metadata;
    data->qalsh_config = disk_metadata.ToConfig();
    data->index_path = DiskQalsh::IndexPath(metadata.file_path.parent_path(), norm_order,
                                             DiskQalsh::ParseTargetFromPointSetPath(metadata.file_path));
    data->index = qalsh::PersistentIndex::Open(data->index_path.string());
    if (data->index->metadata().num_points != metadata.num_points ||
        data->index->metadata().num_dimensions != metadata.num_dimensions) {
        throw std::runtime_error("library index metadata does not match the target point set");
    }
    // ValidateIndex(false) deliberately avoids opening the tree twice.  The
    // one structural open above still needs the same metadata compatibility
    // checks that the full preflight path performs.
    DiskQalsh::ValidateLibraryMetadata(data->index->metadata(), disk_metadata, norm_order,
                                       data->index_path.parent_path());
    InitializeFromIndexData(std::move(data));
}

AnnResult DiskBasedQalshAnnSearcher::Search(const Point& query_point) {
    if (!index_data_) {
        Log::error("Disk QALSH searcher is not initialized.");
    }
    if (query_point.size() != index_data_->target_metadata.num_dimensions) {
        Log::error("Query point dimension {} does not match disk QALSH target dimension {}.", query_point.size(),
                   index_data_->target_metadata.num_dimensions);
    }
    const auto search_start = std::chrono::steady_clock::now();
    if (!engine_) {
        Log::error("Disk QALSH searcher is not initialized.");
    }
    qalsh::DefaultQalshStrategy strategy(index_data_->qalsh_config.collision_threshold);
    strategy.set_candidate_budget(std::nullopt);
    const qalsh::SearchResult result = engine_->search(query_point, 1, strategy);
    if (options_.profile) {
        const auto search_end = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(search_end - search_start).count();
        std::lock_guard lock(profile_mutex_);
        ++profile_.queries;
        profile_.table_entry_reads += result.projection_hits;
        profile_.candidate_point_reads += result.evaluated_candidates;
        profile_.candidates_evaluated += result.evaluated_candidates;
        profile_.total_search_time_ms += elapsed_ms;
    }
    return ToAnnResult(result);
}
