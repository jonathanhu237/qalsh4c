#include "ann_searcher.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#ifdef QALSH4C_USE_OPENMP
#include <omp.h>
#endif

#include "global.h"
#include "log.h"
#include "point_set.h"
#include "utils.h"

namespace {

float DotProductRawData(const Coordinate* lhs_data, const Coordinate* rhs_data, size_t num_dimensions) {
    float result = 0.0;
    const auto dimension_count = static_cast<long long>(num_dimensions);

#ifdef QALSH4C_USE_OPENMP
#pragma omp simd reduction(+ : result)
#endif
    for (long long dimension = 0; dimension < dimension_count; ++dimension) {
        result += lhs_data[dimension] * rhs_data[dimension];
    }

    return result;
}

float DotProductRaw(const Point& lhs, const Point& rhs) {
    return DotProductRawData(lhs.data(), rhs.data(), lhs.size());
}

bool IsBetterAnnResult(const AnnResult& candidate, const AnnResult& best) {
    return candidate.distance < best.distance ||
           (candidate.distance == best.distance && candidate.point_id < best.point_id);
}

int OpenReadOnlyFd(const std::filesystem::path& path, const std::string& label) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        Log::error("Failed to open {} for disk QALSH search: {} ({})", label, path.string(), std::strerror(errno));
    }
    return fd;
}

void CloseFd(int& fd) {
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

void ReadExactAt(int fd, void* output, size_t bytes, std::uint64_t offset, const std::string& label) {
    auto* data = static_cast<char*>(output);
    size_t total_read = 0;
    while (total_read < bytes) {
        const ssize_t read_count =
            ::pread(fd, data + total_read, bytes - total_read, static_cast<off_t>(offset + total_read));
        if (read_count < 0) {
            if (errno == EINTR) {
                continue;
            }
            Log::error("Failed to read {} for disk QALSH search at offset {}: {}", label, offset + total_read,
                       std::strerror(errno));
        }
        if (read_count == 0) {
            Log::error("Unexpected EOF while reading {} for disk QALSH search at offset {}.", label,
                       offset + total_read);
        }
        total_read += static_cast<size_t>(read_count);
    }
}

std::shared_ptr<const DiskQalshTableMapping> MapQalshTables(const DiskBasedQalshIndexData& index_data) {
    const std::uint64_t expected_entries = static_cast<std::uint64_t>(index_data.metadata.num_hash_tables) *
                                           static_cast<std::uint64_t>(index_data.metadata.num_points);
    if (expected_entries == 0) {
        Log::error("Disk QALSH tables.bin mapping requires a non-empty index.");
    }
    if (expected_entries >
        std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(sizeof(DiskQalsh::DiskQalshEntry))) {
        Log::error("Disk QALSH tables.bin entry count is too large to size: {} entries.", expected_entries);
    }
    const std::uint64_t expected_bytes =
        expected_entries * static_cast<std::uint64_t>(sizeof(DiskQalsh::DiskQalshEntry));
    if (expected_entries > static_cast<std::uint64_t>(std::numeric_limits<size_t>::max()) ||
        expected_bytes > static_cast<std::uint64_t>(std::numeric_limits<size_t>::max())) {
        Log::error("Disk QALSH tables.bin is too large to map on this platform: {} bytes.", expected_bytes);
    }

    const std::filesystem::path table_path = DiskQalsh::TablesPath(index_data.index_directory);
    int fd = OpenReadOnlyFd(table_path, "tables.bin");

    struct stat table_stat {};
    if (::fstat(fd, &table_stat) != 0) {
        const std::string message = std::strerror(errno);
        CloseFd(fd);
        Log::error("Failed to stat tables.bin for disk QALSH search: {} ({})", table_path.string(), message);
    }
    if (table_stat.st_size < 0 || static_cast<std::uint64_t>(table_stat.st_size) != expected_bytes) {
        const std::uint64_t actual_bytes =
            table_stat.st_size < 0 ? 0 : static_cast<std::uint64_t>(table_stat.st_size);
        CloseFd(fd);
        Log::error("Disk QALSH tables.bin size mismatch: expected {} bytes, found {} bytes in {}.", expected_bytes,
                   actual_bytes, table_path.string());
    }

    const size_t mapping_bytes = static_cast<size_t>(expected_bytes);
    void* mapping = ::mmap(nullptr, mapping_bytes, PROT_READ, MAP_SHARED, fd, 0);
    const std::string mmap_error = mapping == MAP_FAILED ? std::strerror(errno) : "";
    CloseFd(fd);
    if (mapping == MAP_FAILED) {
        Log::error("Failed to mmap tables.bin for disk QALSH search: {} ({})", table_path.string(), mmap_error);
    }

    return std::make_shared<DiskQalshTableMapping>(
        static_cast<const DiskQalsh::DiskQalshEntry*>(mapping), static_cast<size_t>(expected_entries), mapping_bytes);
}

struct DiskQalshBatchQueryState {
    const Coordinate* query_data{nullptr};
    std::vector<InMemoryQalshTableCursor> cursors;
    std::vector<QalshSearchState> search_states;
    std::vector<unsigned char> table_done;
    float radius{1.0F};
    float bound{0.0F};
    bool has_candidate{false};
    bool should_terminate{false};
    bool done{false};
    AnnResult best_candidate{.distance = std::numeric_limits<float>::infinity(), .point_id = 0};

    void Reset(const Coordinate* data, unsigned int num_hash_tables, unsigned int num_points, float initial_bound) {
        query_data = data;
        cursors.assign(num_hash_tables, InMemoryQalshTableCursor{});
        search_states.assign(num_points, QalshSearchState{});
        table_done.assign(num_hash_tables, 0);
        radius = 1.0F;
        bound = initial_bound;
        has_candidate = false;
        should_terminate = false;
        done = false;
        best_candidate = AnnResult{.distance = std::numeric_limits<float>::infinity(), .point_id = 0};
    }

    [[nodiscard]] bool T1Satisfied(float approximation_ratio) const {
        return has_candidate && best_candidate.distance <= approximation_ratio * radius;
    }

    [[nodiscard]] bool AllTablesDone() const {
        return std::ranges::all_of(table_done, [](unsigned char done_value) { return done_value != 0; });
    }

    [[nodiscard]] bool CursorsExhausted() const {
        return std::ranges::all_of(cursors, [](const InMemoryQalshTableCursor& cursor) {
            return !cursor.left.has_value() && !cursor.right.has_value();
        });
    }
};

}  // namespace

DiskQalshTableMapping::DiskQalshTableMapping(const DiskQalsh::DiskQalshEntry* entries, size_t num_entries,
                                             size_t byte_size)
    : entries_(entries),
      num_entries_(num_entries),
      byte_size_(byte_size) {}

DiskQalshTableMapping::~DiskQalshTableMapping() {
    if (entries_ != nullptr && byte_size_ > 0) {
        (void)::munmap(const_cast<DiskQalsh::DiskQalshEntry*>(entries_), byte_size_);
    }
}

const DiskQalsh::DiskQalshEntry* DiskQalshTableMapping::entries() const {
    return entries_;
}

size_t DiskQalshTableMapping::num_entries() const {
    return num_entries_;
}

// ---------------------------------------------
// LinearScanAnnSearcher Implementation
// ---------------------------------------------
bool LinearScanAnnSearcher::RequiresIndex() const {
    return false;
}

std::unique_ptr<AnnSearcher> LinearScanAnnSearcher::Clone() const {
    return std::make_unique<LinearScanAnnSearcher>(*this);
}

void LinearScanAnnSearcher::Init(const PointSetMetadata& metadata, float norm_order) {
    auto base_points = std::make_shared<std::vector<Point>>(
        PointSet::LoadPointsFromFile(metadata.file_path, metadata.num_points, metadata.num_dimensions));
    if (base_points->size() != metadata.num_points) {
        Log::error("Failed to load all target points for in-memory linear scan.");
    }
    InitLoaded(std::move(base_points), metadata.num_dimensions, norm_order);
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

AnnResult LinearScanAnnSearcher::SearchWithUpperBound(const Point& query_point, float initial_upper_bound) {
    if (!base_points_) {
        Log::error("Linear scan searcher is not initialized.");
    }

    AnnResult result{.distance = initial_upper_bound, .point_id = 0};

    for (unsigned int i = 0; i < num_points_; i++) {
        float distance = Calc::LpDistanceBounded((*base_points_)[i], query_point, norm_order_, result.distance);
        if (distance < result.distance) {
            result.point_id = i;
            result.distance = distance;
        }
    }

    return result;
}

std::vector<AnnResult> LinearScanAnnSearcher::SearchBatchWithUpperBounds(
    const std::vector<Point>& query_points, const std::vector<float>& initial_upper_bounds) {
    if (query_points.size() != initial_upper_bounds.size()) {
        Log::error("Query size and upper-bound size mismatch.");
    }
    if (!base_points_) {
        Log::error("Linear scan searcher is not initialized.");
    }

    std::vector<AnnResult> results(query_points.size());
    for (size_t query_index = 0; query_index < query_points.size(); ++query_index) {
        results[query_index] = AnnResult{.distance = initial_upper_bounds[query_index], .point_id = 0};
    }

    for (unsigned int point_id = 0; point_id < num_points_; ++point_id) {
        for (size_t query_index = 0; query_index < query_points.size(); ++query_index) {
            const float distance = Calc::LpDistanceBounded((*base_points_)[point_id], query_points[query_index],
                                                            norm_order_, results[query_index].distance);
            if (distance < results[query_index].distance) {
                results[query_index].distance = distance;
                results[query_index].point_id = point_id;
            }
        }
    }

    return results;
}

// ---------------------------------------------
// DiskBasedLinearScanAnnSearcher Implementation
// ---------------------------------------------
DiskBasedLinearScanAnnSearcher::DiskBasedLinearScanAnnSearcher(unsigned int block_size) : block_size_(block_size) {
    if (block_size_ == 0) {
        Log::error("Disk block size must be positive.");
    }
}

bool DiskBasedLinearScanAnnSearcher::RequiresIndex() const {
    return false;
}

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
    block_buffer_.reserve(static_cast<size_t>(block_size_) * metadata_.num_dimensions);
}

AnnResult DiskBasedLinearScanAnnSearcher::Search(const Point& query_point) {
    return SearchWithUpperBound(query_point, std::numeric_limits<float>::infinity());
}

AnnResult DiskBasedLinearScanAnnSearcher::SearchWithUpperBound(const Point& query_point,
                                                               float initial_upper_bound) {
    if (!initialized_) {
        Log::error("Disk-based linear scan searcher is not initialized.");
    }
    if (query_point.size() != metadata_.num_dimensions) {
        Log::error("Query point dimension {} does not match target dimension {}.", query_point.size(),
                   metadata_.num_dimensions);
    }

    std::ifstream input(metadata_.file_path, std::ios::binary);
    if (!input.is_open()) {
        Log::error("Could not open point set file for disk-based linear scan: {}", metadata_.file_path.string());
    }

    AnnResult best{.distance = initial_upper_bound, .point_id = 0};
    const Coordinate* query_data = query_point.data();

    for (unsigned int block_begin = 0; block_begin < metadata_.num_points; block_begin += block_size_) {
        const unsigned int block_point_count = std::min(block_size_, metadata_.num_points - block_begin);
        const size_t coordinate_count = static_cast<size_t>(block_point_count) * metadata_.num_dimensions;
        block_buffer_.resize(coordinate_count);

        const auto bytes_to_read =
            static_cast<std::streamsize>(coordinate_count * static_cast<size_t>(sizeof(Coordinate)));
        input.read(reinterpret_cast<char*>(block_buffer_.data()), bytes_to_read);
        if (input.gcount() != bytes_to_read) {
            Log::error("Failed to read target block starting at point {} from {}.", block_begin,
                       metadata_.file_path.string());
        }

        AnnResult block_best = best;

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel
        {
            AnnResult local_best = block_best;

#pragma omp for nowait schedule(static)
            for (long long local_index = 0; local_index < static_cast<long long>(block_point_count); ++local_index) {
                const unsigned int point_id = block_begin + static_cast<unsigned int>(local_index);
                const Coordinate* point_data =
                    block_buffer_.data() + static_cast<size_t>(local_index) * metadata_.num_dimensions;
                const float distance = Calc::LpDistanceBoundedRaw(point_data, query_data, metadata_.num_dimensions,
                                                                   norm_order_, local_best.distance);
                const AnnResult candidate{.distance = distance, .point_id = point_id};
                if (IsBetterAnnResult(candidate, local_best)) {
                    local_best = candidate;
                }
            }

#pragma omp critical
            {
                if (IsBetterAnnResult(local_best, block_best)) {
                    block_best = local_best;
                }
            }
        }
#else
        for (unsigned int local_index = 0; local_index < block_point_count; ++local_index) {
            const unsigned int point_id = block_begin + local_index;
            const Coordinate* point_data =
                block_buffer_.data() + static_cast<size_t>(local_index) * metadata_.num_dimensions;
            const float distance = Calc::LpDistanceBoundedRaw(point_data, query_data, metadata_.num_dimensions,
                                                               norm_order_, block_best.distance);
            const AnnResult candidate{.distance = distance, .point_id = point_id};
            if (IsBetterAnnResult(candidate, block_best)) {
                block_best = candidate;
            }
        }
#endif

        best = block_best;
    }

    return best;
}

// ---------------------------------------------
// QalshAnnSearcher Implementation
// ---------------------------------------------
QalshAnnSearcher::QalshAnnSearcher(float approximation_ratio)
    : QalshAnnSearcher(approximation_ratio, Random::SeededGenerator()()) {}

QalshAnnSearcher::QalshAnnSearcher(float approximation_ratio, std::mt19937::result_type random_seed)
    : approximation_ratio_(approximation_ratio),
      random_seed_(random_seed) {}

bool QalshAnnSearcher::RequiresIndex() const {
    return false;
}

std::unique_ptr<AnnSearcher> QalshAnnSearcher::Clone() const {
    auto clone = std::make_unique<QalshAnnSearcher>(approximation_ratio_, random_seed_);
    clone->index_data_ = index_data_;
    clone->ResetSearchState();
    return clone;
}

std::shared_ptr<const std::vector<Point>> QalshAnnSearcher::TargetPoints() const {
    if (!index_data_) {
        return {};
    }
    return std::shared_ptr<const std::vector<Point>>(index_data_, &index_data_->base_points);
}

void QalshAnnSearcher::Init(const PointSetMetadata& metadata, float norm_order) {
    auto index_data = std::make_shared<QalshAnnIndexData>();
    index_data->norm_order = norm_order;
    index_data->num_points = metadata.num_points;
    index_data->num_dimensions = metadata.num_dimensions;

    if (index_data->num_points == 0 || index_data->num_dimensions == 0) {
        Log::error("QALSH requires a non-empty point set with non-zero dimensions.");
    }

    index_data->qalsh_config = QalshConfig{.approximation_ratio = approximation_ratio_};
    index_data->qalsh_config.Regularize(index_data->num_points, norm_order);
    Log::info(
        "Building in-memory QALSH index with approximation ratio {}, bucket width {}, {} hash tables, collision "
        "threshold {}.",
        index_data->qalsh_config.approximation_ratio, index_data->qalsh_config.bucket_width,
        index_data->qalsh_config.num_hash_tables, index_data->qalsh_config.collision_threshold);

    index_data->base_points =
        PointSet::LoadPointsFromFile(metadata.file_path, index_data->num_points, index_data->num_dimensions);
    if (index_data->base_points.size() != index_data->num_points) {
        Log::error("Failed to load all target points for in-memory QALSH index.");
    }

    std::function<float()> coordinate_generator;
    std::mt19937 gen(random_seed_);
    if (std::abs(norm_order - 1.0) < Global::kEpsilon) {
        std::cauchy_distribution<float> dist(0.0, 1.0);
        coordinate_generator = [dist, &gen]() mutable { return dist(gen); };
    } else if (std::abs(norm_order - 2.0) < Global::kEpsilon) {
        std::normal_distribution<float> dist(0.0, 1.0);
        coordinate_generator = [dist, &gen]() mutable { return dist(gen); };
    } else {
        Log::error("Unsupported norm order: {}", norm_order);
    }

    index_data->dot_vectors.clear();
    index_data->dot_vectors.reserve(index_data->qalsh_config.num_hash_tables);
    for (unsigned int table_id = 0; table_id < index_data->qalsh_config.num_hash_tables; ++table_id) {
        Point dot_vector;
        dot_vector.reserve(index_data->num_dimensions);
        for (unsigned int dimension = 0; dimension < index_data->num_dimensions; ++dimension) {
            dot_vector.push_back(coordinate_generator());
        }
        index_data->dot_vectors.push_back(std::move(dot_vector));
    }

    index_data->searchers.clear();
    index_data->searchers.resize(index_data->qalsh_config.num_hash_tables);
    std::vector<std::vector<DotProductPointIdPair>> entries_by_table(index_data->qalsh_config.num_hash_tables);
    for (std::vector<DotProductPointIdPair>& entries : entries_by_table) {
        entries.resize(index_data->num_points);
    }

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long point_index = 0; point_index < static_cast<long long>(index_data->num_points); ++point_index) {
        const auto point_id = static_cast<unsigned int>(point_index);
        const Point& base_point = index_data->base_points[point_id];
        for (unsigned int table_id = 0; table_id < index_data->qalsh_config.num_hash_tables; ++table_id) {
            entries_by_table[table_id][point_id] =
                DotProductPointIdPair{.dot_product = DotProductRaw(base_point, index_data->dot_vectors[table_id]),
                                      .point_id = point_id};
        }
    }

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (long long table_index = 0; table_index < static_cast<long long>(index_data->qalsh_config.num_hash_tables);
         ++table_index) {
        const auto table_id = static_cast<unsigned int>(table_index);
        index_data->searchers[table_id].Build(std::move(entries_by_table[table_id]));
    }

    index_data_ = std::move(index_data);
    ResetSearchState();
}

void QalshAnnSearcher::ResetSearchState() {
    if (!index_data_) {
        search_cursors_.clear();
        search_states_.clear();
        search_epoch_ = 0;
        return;
    }

    search_cursors_.assign(index_data_->qalsh_config.num_hash_tables, InMemoryQalshTableCursor{});
    search_states_.assign(index_data_->num_points, QalshSearchState{});
    search_epoch_ = 0;
}

unsigned int QalshAnnSearcher::NextSearchEpoch() {
    ++search_epoch_;
    if (search_epoch_ == 0) {
        std::ranges::fill(search_states_, QalshSearchState{});
        search_epoch_ = 1;
    }
    return search_epoch_;
}

AnnResult QalshAnnSearcher::Search(const Point& query_point) {
    if (!index_data_) {
        Log::error("QALSH searcher is not initialized.");
    }

    const QalshAnnIndexData& index_data = *index_data_;
    const unsigned int epoch = NextSearchEpoch();
    bool has_candidate = false;
    AnnResult best_candidate{.distance = std::numeric_limits<float>::infinity(), .point_id = 0};

    unsigned int num_hash_tables = index_data.qalsh_config.num_hash_tables;
    unsigned int collision_threshold = index_data.qalsh_config.collision_threshold;
    float bucket_width = index_data.qalsh_config.bucket_width;
    float approximation_ratio = index_data.qalsh_config.approximation_ratio;

    for (unsigned int i = 0; i < num_hash_tables; i++) {
        search_cursors_[i] =
            index_data.searchers[i].InitCursor(DotProductRaw(query_point, index_data.dot_vectors[i]));
    }

    // c-ANN search
    float radius = 1.0F;
    float bound = bucket_width * radius / 2.0F;
    bool shouldTerminate = false;

    auto t1_satisfied = [&]() {
        return has_candidate && best_candidate.distance <= approximation_ratio * radius;
    };

    while (!shouldTerminate) {
        if (t1_satisfied()) {
            break;
        }

        std::queue<unsigned int> active_tables;
        for (unsigned int i = 0; i < num_hash_tables; i++) {
            active_tables.push(i);
        }

        while (!active_tables.empty() && !shouldTerminate) {
            unsigned int i = active_tables.front();
            active_tables.pop();

            bool table_done = index_data.searchers[i].IncrementalSearch(
                search_cursors_[i], bound, [&](unsigned int point_id, [[maybe_unused]] float dot_product) {
                    if (shouldTerminate) {
                        return;
                    }

                    QalshSearchState& state = search_states_[point_id];
                    if (state.visited_stamp == epoch) {
                        return;
                    }

                    if (state.collision_stamp != epoch) {
                        state.collision_stamp = epoch;
                        state.collision_count = 0;
                    }

                    if (++state.collision_count >= collision_threshold) {
                        state.visited_stamp = epoch;
                        const Point& candidate_point = index_data.base_points[point_id];
                        float distance = Calc::LpDistanceBounded(candidate_point, query_point, index_data.norm_order,
                                                                  best_candidate.distance);
                        if (distance < best_candidate.distance) {
                            has_candidate = true;
                            best_candidate = AnnResult{.distance = distance, .point_id = point_id};
                        }

                        // Termination condition 1
                        if (t1_satisfied()) {
                            shouldTerminate = true;
                            return;
                        }
                    }
                });

            if (!table_done) {
                active_tables.push(i);
            }
        }

        if (t1_satisfied()) {
            break;
        }

        // Update radius and bound for next iteration
        radius *= approximation_ratio;
        bound = bucket_width * radius / 2.0F;
    }

    return best_candidate;
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
    : expected_approximation_ratio_(expected_approximation_ratio),
      options_(options) {
    ValidateOptions();
    table_entries_per_block_ = options_.table_block_size / sizeof(DiskQalsh::DiskQalshEntry);
}

DiskBasedQalshAnnSearcher::~DiskBasedQalshAnnSearcher() {
    CloseFiles();
}

bool DiskBasedQalshAnnSearcher::RequiresIndex() const {
    return true;
}

std::unique_ptr<AnnSearcher> DiskBasedQalshAnnSearcher::Clone() const {
    auto clone = std::make_unique<DiskBasedQalshAnnSearcher>(expected_approximation_ratio_, options_);
    clone->index_data_ = index_data_;
    if (clone->index_data_) {
        clone->OpenFiles();
    }
    clone->ResetSearchState();
    return clone;
}

void DiskBasedQalshAnnSearcher::PreflightIndex(const PointSetMetadata& metadata, float norm_order) const {
    (void)DiskQalsh::ValidateIndex(metadata, norm_order, expected_approximation_ratio_);
}

bool DiskBasedQalshAnnSearcher::ProfilingEnabled() const {
    return options_.profile;
}

void DiskBasedQalshAnnSearcher::ResetProfile() {
    profile_ = DiskQalshProfileSnapshot{};
}

void DiskBasedQalshAnnSearcher::MergeProfile(const DiskQalshProfileSnapshot& snapshot) {
    profile_.Merge(snapshot);
}

DiskQalshProfileSnapshot DiskBasedQalshAnnSearcher::ProfileSnapshot() const {
    return profile_;
}

void DiskBasedQalshAnnSearcher::PrintProfile() const {
    if (!options_.profile) {
        return;
    }

    const std::uint64_t cache_lookups = profile_.table_cache_hits + profile_.table_cache_misses;
    const double cache_hit_rate =
        cache_lookups == 0 ? 0.0 : (100.0 * static_cast<double>(profile_.table_cache_hits)) /
                                      static_cast<double>(cache_lookups);
    Log::info(
        "Disk QALSH profile: queries={}, total_search_time_ms={:.3f}, table_entry_reads={}, table_block_reads={}, "
        "table_cache_hits={}, table_cache_misses={}, table_cache_hit_rate={:.2f}%, candidate_point_reads={}, "
        "candidates_evaluated={}.",
        profile_.queries, profile_.total_search_time_ms, profile_.table_entry_reads, profile_.table_block_reads,
        profile_.table_cache_hits, profile_.table_cache_misses, cache_hit_rate, profile_.candidate_point_reads,
        profile_.candidates_evaluated);
}

void DiskBasedQalshAnnSearcher::ValidateOptions() const {
    if (options_.table_block_size < sizeof(DiskQalsh::DiskQalshEntry)) {
        Log::error("Disk QALSH table block size must be at least {} bytes.", sizeof(DiskQalsh::DiskQalshEntry));
    }
    if (options_.table_cache_blocks == 0) {
        Log::error("Disk QALSH table cache block count must be positive.");
    }
}

void DiskBasedQalshAnnSearcher::ResetTableCache() {
    table_cache_.clear();
    table_cache_.reserve(options_.table_cache_blocks);
    table_cache_positions_.clear();
    table_cache_positions_.reserve(options_.table_cache_blocks);
    table_cache_clock_ = 0;
    last_table_cache_slot_ = std::nullopt;
}

void DiskBasedQalshAnnSearcher::Init(const PointSetMetadata& metadata, float norm_order) {
    if (metadata.num_points == 0 || metadata.num_dimensions == 0) {
        Log::error("Disk QALSH requires a non-empty target point set with non-zero dimensions.");
    }

    const DiskQalsh::Target target = DiskQalsh::ParseTargetFromPointSetPath(metadata.file_path);
    const std::filesystem::path dataset_directory = metadata.file_path.parent_path();
    auto index_data = std::make_shared<DiskBasedQalshIndexData>();
    index_data->norm_order = norm_order;
    index_data->target_metadata = metadata;
    index_data->index_directory = DiskQalsh::IndexDirectory(dataset_directory, norm_order, target);
    index_data->metadata = DiskQalsh::ValidateIndex(metadata, norm_order, expected_approximation_ratio_);
    index_data->qalsh_config = index_data->metadata.ToConfig();
    index_data->table_mapping = MapQalshTables(*index_data);

    CloseFiles();
    index_data_ = index_data;
    OpenFiles();

    index_data->dot_vectors.resize(index_data->metadata.num_hash_tables);
    for (unsigned int table_id = 0; table_id < index_data->metadata.num_hash_tables; ++table_id) {
        Point& dot_vector = index_data->dot_vectors[table_id];
        dot_vector.resize(index_data->metadata.num_dimensions);
        const std::uint64_t offset = static_cast<std::uint64_t>(table_id) *
                                     static_cast<std::uint64_t>(index_data->metadata.num_dimensions) *
                                     static_cast<std::uint64_t>(sizeof(Coordinate));
        ReadExactAt(dot_vectors_fd_, dot_vector.data(),
                    static_cast<size_t>(index_data->metadata.num_dimensions) * sizeof(Coordinate), offset,
                    "dot_vectors.bin");
    }

    ResetSearchState();
    Log::info("Using disk QALSH index at {}.", index_data_->index_directory.string());
}

void DiskBasedQalshAnnSearcher::OpenFiles() {
    if (!index_data_) {
        Log::error("Disk QALSH index data is not initialized.");
    }

    CloseFiles();
    ResetTableCache();
    dot_vectors_fd_ = OpenReadOnlyFd(DiskQalsh::DotVectorsPath(index_data_->index_directory), "dot_vectors.bin");
    target_fd_ = OpenReadOnlyFd(index_data_->target_metadata.file_path, "target point set");
}

void DiskBasedQalshAnnSearcher::CloseFiles() {
    CloseFd(dot_vectors_fd_);
    CloseFd(target_fd_);
}

void DiskBasedQalshAnnSearcher::ResetSearchState() {
    if (!index_data_) {
        search_cursors_.clear();
        search_states_.clear();
        candidate_buffer_.clear();
        ResetTableCache();
        search_epoch_ = 0;
        return;
    }

    search_cursors_.assign(index_data_->qalsh_config.num_hash_tables, InMemoryQalshTableCursor{});
    search_states_.assign(index_data_->metadata.num_points, QalshSearchState{});
    candidate_buffer_.assign(index_data_->metadata.num_dimensions, 0.0F);
    ResetTableCache();
    search_epoch_ = 0;
}

unsigned int DiskBasedQalshAnnSearcher::NextSearchEpoch() {
    ++search_epoch_;
    if (search_epoch_ == 0) {
        std::ranges::fill(search_states_, QalshSearchState{});
        search_epoch_ = 1;
    }
    return search_epoch_;
}

std::uint64_t DiskBasedQalshAnnSearcher::TableLinearIndex(unsigned int table_id, size_t entry_index) const {
    if (!index_data_) {
        Log::error("Disk QALSH searcher is not initialized.");
    }
    if (table_id >= index_data_->metadata.num_hash_tables || entry_index >= index_data_->metadata.num_points) {
        Log::error("Disk QALSH table read is out of range: table {}, entry {}.", table_id, entry_index);
    }

    return static_cast<std::uint64_t>(table_id) * static_cast<std::uint64_t>(index_data_->metadata.num_points) +
           static_cast<std::uint64_t>(entry_index);
}

DiskBasedQalshAnnSearcher::TableBlockView DiskBasedQalshAnnSearcher::ReadTableBlock(std::uint64_t block_id) const {
    if (!index_data_ || !index_data_->table_mapping) {
        Log::error("Disk QALSH table mapping is not initialized.");
    }

    if (options_.profile) {
        auto touch_cached_block = [&](size_t cache_slot) {
            TableBlockCacheEntry& cached_block = table_cache_[cache_slot];
            cached_block.last_used = ++table_cache_clock_;
            last_table_cache_slot_ = cache_slot;
        };

        bool cache_hit = false;
        if (last_table_cache_slot_.has_value()) {
            const size_t cache_slot = *last_table_cache_slot_;
            if (cache_slot < table_cache_.size() && table_cache_[cache_slot].block_id == block_id) {
                cache_hit = true;
                touch_cached_block(cache_slot);
            }
        }

        if (!cache_hit) {
            const auto cache_position = table_cache_positions_.find(block_id);
            if (cache_position != table_cache_positions_.end()) {
                cache_hit = true;
                touch_cached_block(cache_position->second);
            }
        }

        if (cache_hit) {
            ++profile_.table_cache_hits;
        } else {
            ++profile_.table_cache_misses;
            ++profile_.table_block_reads;

            TableBlockCacheEntry* cache_slot = nullptr;
            size_t cache_slot_index = 0;
            if (table_cache_.size() < options_.table_cache_blocks) {
                table_cache_.push_back(TableBlockCacheEntry{});
                cache_slot_index = table_cache_.size() - 1;
                cache_slot = &table_cache_[cache_slot_index];
            } else {
                auto evict_it = std::min_element(table_cache_.begin(), table_cache_.end(),
                                                 [](const TableBlockCacheEntry& lhs, const TableBlockCacheEntry& rhs) {
                                                     return lhs.last_used < rhs.last_used;
                                                 });
                cache_slot_index = static_cast<size_t>(std::distance(table_cache_.begin(), evict_it));
                cache_slot = &table_cache_[cache_slot_index];
                table_cache_positions_.erase(cache_slot->block_id);
            }

            cache_slot->block_id = block_id;
            cache_slot->last_used = ++table_cache_clock_;
            table_cache_positions_[block_id] = cache_slot_index;
            last_table_cache_slot_ = cache_slot_index;
        }
    }

    const std::uint64_t total_entries = static_cast<std::uint64_t>(index_data_->table_mapping->num_entries());
    const std::uint64_t block_start_index = block_id * static_cast<std::uint64_t>(table_entries_per_block_);
    if (block_start_index >= total_entries) {
        Log::error("Disk QALSH table block {} is out of range.", block_id);
    }
    const std::uint64_t remaining_entries = total_entries - block_start_index;
    const size_t entries_in_block =
        static_cast<size_t>(std::min<std::uint64_t>(table_entries_per_block_, remaining_entries));
    if (entries_in_block == 0) {
        Log::error("Disk QALSH mapped table block is empty: block {}.", block_id);
    }

    return TableBlockView{.entries = index_data_->table_mapping->entries() + static_cast<size_t>(block_start_index),
                          .size = entries_in_block};
}

void DiskBasedQalshAnnSearcher::CountTableEntryRead() const {
    if (options_.profile) {
        ++profile_.table_entry_reads;
    }
}

void DiskBasedQalshAnnSearcher::ValidateTableEntry(const DiskQalsh::DiskQalshEntry& entry) const {
    if (entry.point_id >= index_data_->metadata.num_points) {
        Log::error("Disk QALSH table entry has invalid point id {}.", entry.point_id);
    }
}

DiskQalsh::DiskQalshEntry DiskBasedQalshAnnSearcher::ReadTableEntry(unsigned int table_id,
                                                                     size_t entry_index) const {
    CountTableEntryRead();
    const std::uint64_t linear_index = TableLinearIndex(table_id, entry_index);
    const std::uint64_t block_id = linear_index / static_cast<std::uint64_t>(table_entries_per_block_);
    const size_t block_entry_index = static_cast<size_t>(linear_index %
                                                        static_cast<std::uint64_t>(table_entries_per_block_));
    const TableBlockView block = ReadTableBlock(block_id);
    if (block_entry_index >= block.size) {
        Log::error("Disk QALSH table mmap entry is out of range: block {}, entry {}.", block_id, block_entry_index);
    }
    const DiskQalsh::DiskQalshEntry entry = block.entries[block_entry_index];
    ValidateTableEntry(entry);
    return entry;
}

InMemoryQalshTableCursor DiskBasedQalshAnnSearcher::InitCursor(unsigned int table_id, Projection query_key) const {
    size_t left = 0;
    size_t right = index_data_->metadata.num_points;
    while (left < right) {
        const size_t mid = left + (right - left) / 2;
        const DiskQalsh::DiskQalshEntry entry = ReadTableEntry(table_id, mid);
        if (entry.projection < query_key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return InMemoryQalshTableCursor{
        .query_key = query_key,
        .left = left == 0 ? std::nullopt : std::optional<size_t>{left - 1},
        .right = left == index_data_->metadata.num_points ? std::nullopt : std::optional<size_t>{left},
    };
}

void DiskBasedQalshAnnSearcher::SearchBatchContiguous(const Coordinate* query_data, unsigned int query_count,
                                                       unsigned int num_dimensions,
                                                       std::vector<AnnResult>& results) {
    if (!index_data_) {
        Log::error("Disk QALSH searcher is not initialized.");
    }
    if (query_count == 0) {
        results.clear();
        return;
    }
    if (query_data == nullptr) {
        Log::error("Disk QALSH batch search requires non-null query data.");
    }
    if (num_dimensions != index_data_->metadata.num_dimensions) {
        Log::error("Query point dimension {} does not match disk QALSH target dimension {}.", num_dimensions,
                   index_data_->metadata.num_dimensions);
    }

#ifndef QALSH4C_USE_OPENMP
    results.resize(query_count);
    Point query_point(num_dimensions);
    for (unsigned int query_index = 0; query_index < query_count; ++query_index) {
        const Coordinate* local_query_data = query_data + static_cast<size_t>(query_index) * num_dimensions;
        std::copy_n(local_query_data, num_dimensions, query_point.begin());
        results[query_index] = Search(query_point);
    }
    return;
#else
    const auto batch_start = std::chrono::steady_clock::now();
    const unsigned int num_hash_tables = index_data_->qalsh_config.num_hash_tables;
    const unsigned int num_points = index_data_->metadata.num_points;
    const unsigned int collision_threshold = index_data_->qalsh_config.collision_threshold;
    const float bucket_width = index_data_->qalsh_config.bucket_width;
    const float approximation_ratio = index_data_->qalsh_config.approximation_ratio;
    constexpr unsigned int kBatchSearchEpoch = 1;

    results.assign(query_count, AnnResult{.distance = std::numeric_limits<float>::infinity(), .point_id = 0});
    std::vector<DiskQalshBatchQueryState> query_states(query_count);

    bool all_queries_done = false;

#pragma omp parallel
    {
        std::unique_ptr<AnnSearcher> local_searcher_base = Clone();
        auto* local_searcher = dynamic_cast<DiskBasedQalshAnnSearcher*>(local_searcher_base.get());
        if (local_searcher == nullptr) {
            Log::error("Disk QALSH batch search clone has unexpected searcher type.");
        }

#pragma omp for schedule(static)
        for (long long query_index = 0; query_index < static_cast<long long>(query_count); ++query_index) {
            const auto local_query_index = static_cast<unsigned int>(query_index);
            const Coordinate* local_query_data =
                query_data + static_cast<size_t>(local_query_index) * num_dimensions;
            query_states[local_query_index].Reset(local_query_data, num_hash_tables, num_points,
                                                  bucket_width / 2.0F);
        }

        for (unsigned int table_id = 0; table_id < num_hash_tables; ++table_id) {
#pragma omp for schedule(static)
            for (long long query_index = 0; query_index < static_cast<long long>(query_count); ++query_index) {
                const auto local_query_index = static_cast<unsigned int>(query_index);
                DiskQalshBatchQueryState& state = query_states[local_query_index];
                const float query_key =
                    DotProductRawData(state.query_data, index_data_->dot_vectors[table_id].data(), num_dimensions);
                state.cursors[table_id] = local_searcher->InitCursor(table_id, query_key);
            }
        }

        while (true) {
#pragma omp single
            {
                all_queries_done = std::ranges::all_of(
                    query_states, [](const DiskQalshBatchQueryState& state) { return state.done; });
            }
            if (all_queries_done) {
                break;
            }

            for (unsigned int table_id = 0; table_id < num_hash_tables; ++table_id) {
#pragma omp for schedule(static)
                for (long long query_index = 0; query_index < static_cast<long long>(query_count); ++query_index) {
                    const auto local_query_index = static_cast<unsigned int>(query_index);
                    DiskQalshBatchQueryState& state = query_states[local_query_index];
                    if (state.done || state.table_done[table_id] != 0) {
                        continue;
                    }

                    const bool table_done = local_searcher->IncrementalSearch(
                        table_id, state.cursors[table_id], state.bound,
                        [&](unsigned int point_id, [[maybe_unused]] Projection projection) {
                            if (state.should_terminate) {
                                return;
                            }

                            QalshSearchState& point_state = state.search_states[point_id];
                            if (point_state.visited_stamp == kBatchSearchEpoch) {
                                return;
                            }

                            if (point_state.collision_stamp != kBatchSearchEpoch) {
                                point_state.collision_stamp = kBatchSearchEpoch;
                                point_state.collision_count = 0;
                            }

                            if (++point_state.collision_count >= collision_threshold) {
                                point_state.visited_stamp = kBatchSearchEpoch;
                                local_searcher->ReadCandidatePoint(point_id);
                                if (options_.profile) {
                                    ++local_searcher->profile_.candidates_evaluated;
                                }
                                const float distance = Calc::LpDistanceBoundedRaw(
                                    local_searcher->candidate_buffer_.data(), state.query_data,
                                    index_data_->metadata.num_dimensions, index_data_->norm_order,
                                    state.best_candidate.distance);
                                const AnnResult candidate{.distance = distance, .point_id = point_id};
                                if (IsBetterAnnResult(candidate, state.best_candidate)) {
                                    state.has_candidate = true;
                                    state.best_candidate = candidate;
                                }

                                if (state.T1Satisfied(approximation_ratio)) {
                                    state.should_terminate = true;
                                }
                            }
                        });

                    if (table_done) {
                        state.table_done[table_id] = 1;
                    }
                    if (state.should_terminate || state.T1Satisfied(approximation_ratio)) {
                        state.done = true;
                    }
                }
            }

#pragma omp single
            {
                for (DiskQalshBatchQueryState& state : query_states) {
                    if (state.done || !state.AllTablesDone()) {
                        continue;
                    }

                    if (state.T1Satisfied(approximation_ratio)) {
                        state.done = true;
                    } else if (state.CursorsExhausted()) {
                        if (state.has_candidate) {
                            state.done = true;
                        } else {
                            Log::error("Disk QALSH batch search exhausted all table cursors without finding a "
                                       "candidate.");
                        }
                    } else {
                        state.radius *= approximation_ratio;
                        state.bound = bucket_width * state.radius / 2.0F;
                        std::ranges::fill(state.table_done, 0);
                    }
                }
            }
        }

        if (options_.profile) {
            const DiskQalshProfileSnapshot snapshot = local_searcher->ProfileSnapshot();
#pragma omp critical(disk_qalsh_profile_merge)
            MergeProfile(snapshot);
        }
    }

    for (unsigned int query_index = 0; query_index < query_count; ++query_index) {
        const DiskQalshBatchQueryState& state = query_states[query_index];
        if (!state.has_candidate) {
            Log::error("Disk QALSH batch search did not find a candidate.");
        }
        results[query_index] = state.best_candidate;
    }

    if (options_.profile) {
        const auto batch_end = std::chrono::steady_clock::now();
        profile_.queries += query_count;
        profile_.total_search_time_ms +=
            std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
    }
#endif
}

void DiskBasedQalshAnnSearcher::ReadCandidatePoint(unsigned int point_id) {
    if (!index_data_) {
        Log::error("Disk QALSH searcher is not initialized.");
    }
    if (point_id >= index_data_->metadata.num_points) {
        Log::error("Disk QALSH candidate point id {} is out of range.", point_id);
    }
    if (candidate_buffer_.size() != index_data_->metadata.num_dimensions) {
        candidate_buffer_.assign(index_data_->metadata.num_dimensions, 0.0F);
    }
    if (options_.profile) {
        ++profile_.candidate_point_reads;
    }

    const std::uint64_t offset = static_cast<std::uint64_t>(point_id) *
                                 static_cast<std::uint64_t>(index_data_->metadata.num_dimensions) *
                                 static_cast<std::uint64_t>(sizeof(Coordinate));
    ReadExactAt(target_fd_, candidate_buffer_.data(),
                static_cast<size_t>(index_data_->metadata.num_dimensions) * sizeof(Coordinate), offset,
                "target point set");
}

AnnResult DiskBasedQalshAnnSearcher::Search(const Point& query_point) {
    if (!index_data_) {
        Log::error("Disk QALSH searcher is not initialized.");
    }
    if (query_point.size() != index_data_->metadata.num_dimensions) {
        Log::error("Query point dimension {} does not match disk QALSH target dimension {}.", query_point.size(),
                   index_data_->metadata.num_dimensions);
    }

    const auto search_start = std::chrono::steady_clock::now();
    if (options_.profile) {
        ++profile_.queries;
    }

    const unsigned int epoch = NextSearchEpoch();
    bool has_candidate = false;
    AnnResult best_candidate{.distance = std::numeric_limits<float>::infinity(), .point_id = 0};

    const unsigned int num_hash_tables = index_data_->qalsh_config.num_hash_tables;
    const unsigned int collision_threshold = index_data_->qalsh_config.collision_threshold;
    const float bucket_width = index_data_->qalsh_config.bucket_width;
    const float approximation_ratio = index_data_->qalsh_config.approximation_ratio;

    for (unsigned int table_id = 0; table_id < num_hash_tables; ++table_id) {
        search_cursors_[table_id] =
            InitCursor(table_id, DotProductRaw(query_point, index_data_->dot_vectors[table_id]));
    }

    float radius = 1.0F;
    float bound = bucket_width * radius / 2.0F;
    bool should_terminate = false;

    auto t1_satisfied = [&]() {
        return has_candidate && best_candidate.distance <= approximation_ratio * radius;
    };

    auto cursors_exhausted = [&]() {
        return std::ranges::all_of(search_cursors_, [](const InMemoryQalshTableCursor& cursor) {
            return !cursor.left.has_value() && !cursor.right.has_value();
        });
    };

    while (!should_terminate) {
        if (t1_satisfied()) {
            break;
        }

        std::queue<unsigned int> active_tables;
        for (unsigned int table_id = 0; table_id < num_hash_tables; ++table_id) {
            active_tables.push(table_id);
        }

        while (!active_tables.empty() && !should_terminate) {
            const unsigned int table_id = active_tables.front();
            active_tables.pop();

            const bool table_done = IncrementalSearch(
                table_id, search_cursors_[table_id], bound,
                [&](unsigned int point_id, [[maybe_unused]] Projection projection) {
                    if (should_terminate) {
                        return;
                    }

                    QalshSearchState& state = search_states_[point_id];
                    if (state.visited_stamp == epoch) {
                        return;
                    }

                    if (state.collision_stamp != epoch) {
                        state.collision_stamp = epoch;
                        state.collision_count = 0;
                    }

                    if (++state.collision_count >= collision_threshold) {
                        state.visited_stamp = epoch;
                        ReadCandidatePoint(point_id);
                        if (options_.profile) {
                            ++profile_.candidates_evaluated;
                        }
                        const float distance =
                            Calc::LpDistanceBoundedRaw(candidate_buffer_.data(), query_point.data(),
                                                       index_data_->metadata.num_dimensions, index_data_->norm_order,
                                                       best_candidate.distance);
                        const AnnResult candidate{.distance = distance, .point_id = point_id};
                        if (IsBetterAnnResult(candidate, best_candidate)) {
                            has_candidate = true;
                            best_candidate = candidate;
                        }

                        if (t1_satisfied()) {
                            should_terminate = true;
                        }
                    }
                });

            if (!table_done) {
                active_tables.push(table_id);
            }
        }

        if (t1_satisfied()) {
            break;
        }
        if (cursors_exhausted()) {
            if (has_candidate) {
                break;
            }
            Log::error("Disk QALSH search exhausted all table cursors without finding a candidate.");
        }

        radius *= approximation_ratio;
        bound = bucket_width * radius / 2.0F;
    }

    if (!has_candidate) {
        Log::error("Disk QALSH search did not find a candidate.");
    }
    if (options_.profile) {
        const auto search_end = std::chrono::steady_clock::now();
        profile_.total_search_time_ms +=
            std::chrono::duration<double, std::milli>(search_end - search_start).count();
    }
    return best_candidate;
}
