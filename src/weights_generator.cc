#include "weights_generator.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ann_searcher.h"
#include "config.h"
#include "disk_qalsh_index.h"
#include "global.h"
#include "log.h"
#include "point_set.h"
#include "utils.h"

#ifdef QALSH4C_USE_OPENMP
#include <omp.h>
#endif

namespace {

using CellKey = std::vector<std::uint64_t>;

constexpr unsigned int kBitsPerWord = 64;
constexpr Weight kMinimumWeightFloor = 1e-12F;
constexpr Weight kWeightFloorScale = 1e-12F;

struct CellKeyHash {
    std::size_t operator()(const CellKey& key) const {
        std::size_t seed = key.size();
        for (std::uint64_t word : key) {
            seed ^= std::hash<std::uint64_t>{}(word) + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        }
        return seed;
    }
};

bool IsLowerCell(const CellKey& key, unsigned int dimension) {
    const unsigned int word_index = dimension / kBitsPerWord;
    const unsigned int bit_index = dimension % kBitsPerWord;
    return ((key[word_index] >> bit_index) & 1ULL) != 0;
}

CellKey ComputeCellKey(const Point& point, const Point& midpoint) {
    CellKey key((midpoint.size() + kBitsPerWord - 1) / kBitsPerWord, 0);
    for (unsigned int dimension = 0; dimension < midpoint.size(); ++dimension) {
        if (point[dimension] <= midpoint[dimension]) {
            const unsigned int word_index = dimension / kBitsPerWord;
            const unsigned int bit_index = dimension % kBitsPerWord;
            key[word_index] |= (1ULL << bit_index);
        }
    }
    return key;
}

std::filesystem::path QalshWeightsPath(const PointSetMetadata& from_metadata, float norm_order) {
    std::filesystem::path parent_directory = from_metadata.file_path.parent_path();
    std::string stem = from_metadata.file_path.stem();
    return parent_directory / std::format("l{}_qalsh_weights_{}.bin", norm_order, stem);
}

std::filesystem::path QuadtreeWeightsPath(const PointSetMetadata& from_metadata, float norm_order,
                                          unsigned int max_levels) {
    const std::filesystem::path parent_directory = from_metadata.file_path.parent_path();
    const std::string stem = from_metadata.file_path.stem().string();
    return parent_directory / std::format("l{}_quadtree_l{}_weights_{}.bin", norm_order, max_levels, stem);
}

std::streamsize ExpectedWeightsBytes(const PointSetMetadata& from_metadata) {
    return static_cast<std::streamsize>(sizeof(Weight) * static_cast<size_t>(from_metadata.num_points));
}

bool HasValidQalshWeightsCache(const PointSetMetadata& from_metadata, float norm_order) {
    const std::filesystem::path weights_path = QalshWeightsPath(from_metadata, norm_order);
    const std::streamsize expected_bytes = ExpectedWeightsBytes(from_metadata);
    if (!std::filesystem::exists(weights_path)) {
        return false;
    }

    const auto actual_bytes = static_cast<std::streamsize>(std::filesystem::file_size(weights_path));
    if (actual_bytes != expected_bytes) {
        Log::error("QALSH weights cache file has unexpected size: expected {} bytes, got {} bytes from {}",
                   expected_bytes, actual_bytes, weights_path.string());
    }
    return true;
}

bool ReadQalshWeightsCacheIfAvailable(const PointSetMetadata& from_metadata, float norm_order,
                                      std::vector<Weight>& weights) {
    if (!HasValidQalshWeightsCache(from_metadata, norm_order)) {
        return false;
    }

    const std::filesystem::path weights_path = QalshWeightsPath(from_metadata, norm_order);
    const std::streamsize expected_bytes = ExpectedWeightsBytes(from_metadata);
    std::ifstream input(weights_path, std::ios::binary);
    if (!input.is_open()) {
        Log::error("Failed to open QALSH weights cache file: {}", weights_path.string());
    }
    input.read(reinterpret_cast<char*>(weights.data()), expected_bytes);
    if (input.gcount() != expected_bytes) {
        Log::error("Failed to read QALSH weights cache file: expected {} bytes, read {} bytes from {}",
                   expected_bytes, input.gcount(), weights_path.string());
    }
    return true;
}

void WriteQalshWeightsCache(const PointSetMetadata& from_metadata, float norm_order,
                            const std::vector<Weight>& weights) {
    const std::filesystem::path weights_path = QalshWeightsPath(from_metadata, norm_order);
    const std::streamsize expected_bytes = ExpectedWeightsBytes(from_metadata);
    std::ofstream output(weights_path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        Log::error("Failed to open QALSH weights cache file for writing: {}", weights_path.string());
    }
    output.write(reinterpret_cast<const char*>(weights.data()), expected_bytes);
}

bool HasValidQuadtreeWeightsCache(const std::filesystem::path& weights_path,
                                  const PointSetMetadata& from_metadata) {
    const std::streamsize expected_bytes = ExpectedWeightsBytes(from_metadata);
    if (!std::filesystem::exists(weights_path)) {
        return false;
    }

    const auto actual_bytes = static_cast<std::streamsize>(std::filesystem::file_size(weights_path));
    if (actual_bytes != expected_bytes) {
        Log::error("Quadtree weights cache file has unexpected size: expected {} bytes, got {} bytes from {}",
                   expected_bytes, actual_bytes, weights_path.string());
    }
    return true;
}

bool ReadQuadtreeWeightsCacheIfAvailable(const std::filesystem::path& weights_path,
                                         const PointSetMetadata& from_metadata,
                                         std::vector<Weight>& weights) {
    if (!HasValidQuadtreeWeightsCache(weights_path, from_metadata)) {
        return false;
    }

    const std::streamsize expected_bytes = ExpectedWeightsBytes(from_metadata);
    std::ifstream input(weights_path, std::ios::binary);
    if (!input.is_open()) {
        Log::error("Failed to open quadtree weights cache file: {}", weights_path.string());
    }
    input.read(reinterpret_cast<char*>(weights.data()), expected_bytes);
    if (input.gcount() != expected_bytes) {
        Log::error("Failed to read quadtree weights cache file: expected {} bytes, read {} bytes from {}",
                   expected_bytes, input.gcount(), weights_path.string());
    }
    return true;
}

void WriteQuadtreeWeightsCache(const std::filesystem::path& weights_path, const PointSetMetadata& from_metadata,
                               const std::vector<Weight>& weights) {
    const std::streamsize expected_bytes = ExpectedWeightsBytes(from_metadata);
    std::ofstream output(weights_path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        Log::error("Failed to open quadtree weights cache file for writing: {}", weights_path.string());
    }
    output.write(reinterpret_cast<const char*>(weights.data()), expected_bytes);
}

std::pair<Point, Point> ComputeBounds(const std::vector<Point>& points, unsigned int num_dimensions) {
    Point lower(num_dimensions, std::numeric_limits<Coordinate>::infinity());
    Point upper(num_dimensions, -std::numeric_limits<Coordinate>::infinity());

    for (const Point& point : points) {
        for (unsigned int dimension = 0; dimension < num_dimensions; ++dimension) {
            lower[dimension] = std::min(lower[dimension], point[dimension]);
            upper[dimension] = std::max(upper[dimension], point[dimension]);
        }
    }

    return {lower, upper};
}

std::pair<Point, Point> ChildBounds(const CellKey& key, const Point& lower, const Point& midpoint,
                                    const Point& upper) {
    Point child_lower(lower.size());
    Point child_upper(upper.size());

    for (unsigned int dimension = 0; dimension < lower.size(); ++dimension) {
        if (IsLowerCell(key, dimension)) {
            child_lower[dimension] = lower[dimension];
            child_upper[dimension] = midpoint[dimension];
        } else {
            child_lower[dimension] = midpoint[dimension];
            child_upper[dimension] = upper[dimension];
        }
    }

    return {child_lower, child_upper};
}

void AppendQuadtreeDfsOrder(const std::vector<Point>& points, const std::vector<unsigned int>& cluster,
                            const Point& lower, const Point& upper, unsigned int max_levels,
                            std::vector<unsigned int>& dfs_order) {
    if (max_levels == 1 || cluster.size() <= 1) {
        dfs_order.insert(dfs_order.end(), cluster.begin(), cluster.end());
        return;
    }

    Point midpoint(lower.size());
    for (unsigned int dimension = 0; dimension < lower.size(); ++dimension) {
        midpoint[dimension] = 0.5F * (lower[dimension] + upper[dimension]);
    }

    std::vector<CellKey> cell_order;
    std::vector<std::vector<unsigned int>> subclusters;

    if (midpoint.size() <= 3) {
        const unsigned int num_cells = 1U << static_cast<unsigned int>(midpoint.size());
        std::vector<std::vector<unsigned int>> buckets(num_cells);
        for (unsigned int point_id : cluster) {
            unsigned int cell_number = 0;
            for (unsigned int dimension = 0; dimension < midpoint.size(); ++dimension) {
                if (points[point_id][dimension] <= midpoint[dimension]) {
                    cell_number |= 1U << dimension;
                }
            }
            buckets[cell_number].push_back(point_id);
        }
        for (unsigned int cell_number = 0; cell_number < num_cells; ++cell_number) {
            if (buckets[cell_number].empty()) {
                continue;
            }
            cell_order.push_back(CellKey{cell_number});
            subclusters.push_back(std::move(buckets[cell_number]));
        }
    } else {
        std::unordered_map<CellKey, std::size_t, CellKeyHash> subcluster_by_key;
        for (unsigned int point_id : cluster) {
            CellKey key = ComputeCellKey(points[point_id], midpoint);
            auto [it, inserted] = subcluster_by_key.emplace(key, subclusters.size());
            if (inserted) {
                cell_order.push_back(key);
                subclusters.emplace_back();
            }
            subclusters[it->second].push_back(point_id);
        }
    }

    for (std::size_t cell_index = subclusters.size(); cell_index > 0; --cell_index) {
        const std::size_t current_index = cell_index - 1;
        auto [child_lower, child_upper] = ChildBounds(cell_order[current_index], lower, midpoint, upper);
        AppendQuadtreeDfsOrder(points, subclusters[current_index], child_lower, child_upper, max_levels - 1,
                               dfs_order);
    }
}

std::vector<Point> LoadCombinedPoints(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata) {
    if (from_metadata.num_dimensions != to_metadata.num_dimensions) {
        Log::error("Point set dimensions do not match: {} and {}.", from_metadata.num_dimensions,
                   to_metadata.num_dimensions);
    }
    if (to_metadata.num_points == 0) {
        Log::error("Quadtree sampling requires a non-empty target point set.");
    }

    std::vector<Point> from_points =
        PointSet::LoadPointsFromFile(from_metadata.file_path, from_metadata.num_points, from_metadata.num_dimensions);
    std::vector<Point> to_points =
        PointSet::LoadPointsFromFile(to_metadata.file_path, to_metadata.num_points, to_metadata.num_dimensions);

    std::vector<Point> combined_points;
    combined_points.reserve(from_points.size() + to_points.size());
    combined_points.insert(combined_points.end(), std::make_move_iterator(from_points.begin()),
                           std::make_move_iterator(from_points.end()));
    combined_points.insert(combined_points.end(), std::make_move_iterator(to_points.begin()),
                           std::make_move_iterator(to_points.end()));
    return combined_points;
}

void MakeWeightsPositive(std::vector<Weight>& weights) {
    Weight min_positive_weight = std::numeric_limits<Weight>::infinity();
    for (Weight weight : weights) {
        if (std::isfinite(weight) && weight > 0.0F) {
            min_positive_weight = std::min(min_positive_weight, weight);
        }
    }

    if (!std::isfinite(min_positive_weight)) {
        std::ranges::fill(weights, 1.0F);
        return;
    }

    const Weight weight_floor = std::max(kMinimumWeightFloor, min_positive_weight * kWeightFloorScale);
    for (Weight& weight : weights) {
        if (!std::isfinite(weight) || weight <= 0.0F) {
            weight = weight_floor;
        }
    }
}

void ReadContiguousPointBatch(std::ifstream& input, const PointSetMetadata& metadata, unsigned int batch_begin,
                              unsigned int point_count, std::vector<Coordinate>& buffer) {
    const size_t coordinate_count = static_cast<size_t>(point_count) * metadata.num_dimensions;
    buffer.resize(coordinate_count);
    const auto bytes_to_read =
        static_cast<std::streamsize>(coordinate_count * static_cast<size_t>(sizeof(Coordinate)));
    input.read(reinterpret_cast<char*>(buffer.data()), bytes_to_read);
    if (input.gcount() != bytes_to_read) {
        Log::error("Failed to read query batch starting at point {} from {}.", batch_begin,
                   metadata.file_path.string());
    }
}

void GenerateWeightsFromSearcherStreaming(AnnSearcher& ann_searcher, const PointSetMetadata& from_metadata,
                                          unsigned int query_batch_size, std::vector<Weight>& weights) {
    if (query_batch_size == 0) {
        Log::error("Disk query batch size must be positive.");
    }

    auto* root_disk_qalsh_searcher = dynamic_cast<DiskBasedQalshAnnSearcher*>(&ann_searcher);
    if (root_disk_qalsh_searcher != nullptr) {
        root_disk_qalsh_searcher->ResetProfile();
    }

    std::ifstream query_input(from_metadata.file_path, std::ios::binary);
    if (!query_input.is_open()) {
        Log::error("Could not open query point set file: {}", from_metadata.file_path.string());
    }

    std::vector<Coordinate> query_batch;
    query_batch.reserve(static_cast<size_t>(query_batch_size) * from_metadata.num_dimensions);

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel
    {
        std::unique_ptr<AnnSearcher> local_searcher = ann_searcher.Clone();
        Point query_point(from_metadata.num_dimensions);

        for (unsigned int batch_begin = 0; batch_begin < from_metadata.num_points; batch_begin += query_batch_size) {
            const unsigned int query_count = std::min(query_batch_size, from_metadata.num_points - batch_begin);

#pragma omp single
            ReadContiguousPointBatch(query_input, from_metadata, batch_begin, query_count, query_batch);

#pragma omp for schedule(dynamic)
            for (long long query_index = 0; query_index < static_cast<long long>(query_count); ++query_index) {
                const auto local_query_index = static_cast<unsigned int>(query_index);
                const Coordinate* query_data =
                    query_batch.data() + static_cast<size_t>(local_query_index) * from_metadata.num_dimensions;
                std::copy_n(query_data, from_metadata.num_dimensions, query_point.begin());
                weights[batch_begin + local_query_index] = local_searcher->Search(query_point).distance;
            }
        }

        if (root_disk_qalsh_searcher != nullptr) {
            auto* local_disk_qalsh_searcher = dynamic_cast<DiskBasedQalshAnnSearcher*>(local_searcher.get());
            if (local_disk_qalsh_searcher != nullptr) {
                const DiskQalshProfileSnapshot snapshot = local_disk_qalsh_searcher->ProfileSnapshot();
#pragma omp critical(disk_qalsh_profile_merge)
                root_disk_qalsh_searcher->MergeProfile(snapshot);
            }
        }
    }
#else
    for (unsigned int batch_begin = 0; batch_begin < from_metadata.num_points; batch_begin += query_batch_size) {
        const unsigned int query_count = std::min(query_batch_size, from_metadata.num_points - batch_begin);
        ReadContiguousPointBatch(query_input, from_metadata, batch_begin, query_count, query_batch);

        Point query_point(from_metadata.num_dimensions);
        for (unsigned int local_query_index = 0; local_query_index < query_count; ++local_query_index) {
            const Coordinate* query_data =
                query_batch.data() + static_cast<size_t>(local_query_index) * from_metadata.num_dimensions;
            std::copy_n(query_data, from_metadata.num_dimensions, query_point.begin());
            weights[batch_begin + local_query_index] = ann_searcher.Search(query_point).distance;
        }
    }
#endif
}

}  // namespace

// --------------------------------------------------
// UniformWeightsGenerator Implementation
// --------------------------------------------------
bool UniformWeightsGenerator::RequiresIndex() const {
    return false;
}

std::vector<Weight> UniformWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                      [[maybe_unused]] const PointSetMetadata& to_metadata,
                                                      [[maybe_unused]] float norm_order,
                                                      [[maybe_unused]] bool use_cache) {
    std::vector<Weight> weights(from_metadata.num_points, 1.0);
    return weights;
}

// --------------------------------------------------
// QuadtreeWeightsGenerator Implementation
// --------------------------------------------------
QuadtreeWeightsGenerator::QuadtreeWeightsGenerator(unsigned int max_levels) : max_levels_(max_levels) {
    if (max_levels_ < 1) {
        Log::error("Quadtree max-levels must be at least 1.");
    }
}

bool QuadtreeWeightsGenerator::RequiresIndex() const {
    return false;
}

std::vector<Weight> QuadtreeWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                       const PointSetMetadata& to_metadata, float norm_order,
                                                       bool use_cache) {
    if (std::abs(norm_order - 1.0F) >= Global::kEpsilon && std::abs(norm_order - 2.0F) >= Global::kEpsilon) {
        Log::error("Unsupported norm order for quadtree sampling: {}", norm_order);
    }

    std::vector<Weight> weights(from_metadata.num_points);
    const std::filesystem::path weights_path = QuadtreeWeightsPath(from_metadata, norm_order, max_levels_);

    if (use_cache) {
        Log::warn("The quadtree weights will be read from the cache file if it exists.");
        if (ReadQuadtreeWeightsCacheIfAvailable(weights_path, from_metadata, weights)) {
            return weights;
        }
        Log::warn("Cache file doesn't exist. Generating the new one...");
    }

    if (from_metadata.num_points == 0) {
        return weights;
    }

    Log::info("Generating weights using quadtree sampling...");
    const std::vector<Point> combined_points = LoadCombinedPoints(from_metadata, to_metadata);
    const unsigned int total_points = static_cast<unsigned int>(combined_points.size());
    const unsigned int target_offset = from_metadata.num_points;

    std::vector<unsigned int> root_cluster(total_points);
    std::iota(root_cluster.begin(), root_cluster.end(), 0);

    auto [lower, upper] = ComputeBounds(combined_points, from_metadata.num_dimensions);
    std::vector<unsigned int> dfs_order;
    dfs_order.reserve(total_points);
    AppendQuadtreeDfsOrder(combined_points, root_cluster, lower, upper, max_levels_, dfs_order);

    if (dfs_order.size() != total_points) {
        Log::error("Quadtree DFS produced {} points, expected {}.", dfs_order.size(), total_points);
    }

    std::vector<unsigned int> dfs_position(total_points);
    for (unsigned int position = 0; position < total_points; ++position) {
        dfs_position[dfs_order[position]] = position;
    }

    unsigned int first_target_position = total_points;
    unsigned int last_target_position = 0;
    for (unsigned int position = 0; position < total_points; ++position) {
        if (dfs_order[position] >= target_offset) {
            first_target_position = std::min(first_target_position, position);
            last_target_position = position;
        }
    }

    if (first_target_position == total_points) {
        Log::error("Quadtree sampling requires target points in DFS order.");
    }

    std::vector<unsigned int> left_target_by_position(total_points);
    unsigned int current_target_position = first_target_position;
    for (unsigned int position = 0; position < total_points; ++position) {
        if (dfs_order[position] >= target_offset) {
            current_target_position = position;
        }
        left_target_by_position[position] = dfs_order[current_target_position];
    }

    std::vector<unsigned int> right_target_by_position(total_points);
    current_target_position = last_target_position;
    for (unsigned int position = total_points; position > 0; --position) {
        const unsigned int current_position = position - 1;
        if (dfs_order[current_position] >= target_offset) {
            current_target_position = current_position;
        }
        right_target_by_position[current_position] = dfs_order[current_target_position];
    }

    for (unsigned int source_id = 0; source_id < from_metadata.num_points; ++source_id) {
        const unsigned int source_position = dfs_position[source_id];
        const unsigned int left_target_id = left_target_by_position[source_position];
        const unsigned int right_target_id = right_target_by_position[source_position];
        const Weight left_distance =
            Calc::LpDistance(combined_points[source_id], combined_points[left_target_id], norm_order);
        const Weight right_distance =
            Calc::LpDistance(combined_points[source_id], combined_points[right_target_id], norm_order);
        weights[source_id] = std::min(left_distance, right_distance);
    }

    MakeWeightsPositive(weights);

    if (use_cache) {
        WriteQuadtreeWeightsCache(weights_path, from_metadata, weights);
    }

    return weights;
}

// --------------------------------------------------
// QalshWeightsGenerator Implementation
// --------------------------------------------------
QalshWeightsGenerator::QalshWeightsGenerator(float approximation_ratio)
    : approximation_ratio_(approximation_ratio),
      ann_searcher_(std::make_unique<QalshAnnSearcher>(approximation_ratio_)) {}

QalshWeightsGenerator::QalshWeightsGenerator(std::optional<float> expected_approximation_ratio,
                                             unsigned int disk_query_batch_size,
                                             DiskQalshSearchOptions disk_qalsh_options)
    : approximation_ratio_(expected_approximation_ratio.value_or(Global::kDefaultApproximationRatio)),
      disk_based_(true),
      disk_query_batch_size_(disk_query_batch_size),
      disk_qalsh_options_(disk_qalsh_options),
      expected_approximation_ratio_(expected_approximation_ratio),
      ann_searcher_(std::make_unique<DiskBasedQalshAnnSearcher>(expected_approximation_ratio_,
                                                                 disk_qalsh_options_)) {}

bool QalshWeightsGenerator::RequiresIndex() const {
    return false;
}

QalshConfig QalshWeightsGenerator::BuildConfig(const PointSetMetadata& from_metadata,
                                               const PointSetMetadata& to_metadata, float norm_order,
                                               bool use_cache) const {
    if (disk_based_) {
        if (use_cache && HasValidQalshWeightsCache(from_metadata, norm_order)) {
            const DiskQalsh::Target target = DiskQalsh::ParseTargetFromPointSetPath(to_metadata.file_path);
            const std::filesystem::path index_directory =
                DiskQalsh::IndexDirectory(to_metadata.file_path.parent_path(), norm_order, target);
            if (std::filesystem::exists(index_directory)) {
                return DiskQalsh::ValidateIndex(to_metadata, norm_order, expected_approximation_ratio_).ToConfig();
            }

            QalshConfig config{.approximation_ratio = approximation_ratio_};
            config.Regularize(to_metadata.num_points, norm_order);
            return config;
        }

        return DiskQalsh::ValidateIndex(to_metadata, norm_order, expected_approximation_ratio_).ToConfig();
    }

    QalshConfig config{.approximation_ratio = approximation_ratio_};
    config.Regularize(to_metadata.num_points, norm_order);
    return config;
}

std::shared_ptr<const std::vector<Point>> QalshWeightsGenerator::TargetPoints() const {
    const auto* qalsh_searcher = dynamic_cast<const QalshAnnSearcher*>(ann_searcher_.get());
    if (qalsh_searcher == nullptr) {
        return {};
    }
    return qalsh_searcher->TargetPoints();
}

void QalshWeightsGenerator::PreflightGenerate(const PointSetMetadata& from_metadata,
                                              const PointSetMetadata& to_metadata, float norm_order,
                                              bool use_cache) const {
    if (!disk_based_) {
        return;
    }

    if (use_cache && HasValidQalshWeightsCache(from_metadata, norm_order)) {
        return;
    }

    const auto* disk_searcher = dynamic_cast<const DiskBasedQalshAnnSearcher*>(ann_searcher_.get());
    if (disk_searcher == nullptr) {
        Log::error("Disk QALSH weights generator is not configured with a disk QALSH searcher.");
    }
    disk_searcher->PreflightIndex(to_metadata, norm_order);
}

std::vector<Weight> QalshWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                    const PointSetMetadata& to_metadata, float norm_order,
                                                    bool use_cache) {
    std::vector<Weight> weights(from_metadata.num_points);

    if (use_cache) {
        Log::warn("The weights will be read from the cache file if it exists.");
        if (ReadQalshWeightsCacheIfAvailable(from_metadata, norm_order, weights)) {
            return weights;
        }
        Log::warn("Cache file doesn't exist. Generating the new one...");
    }

    // Generate weights based on QALSH algorithm.
    Log::info("Generating weights using QALSH...");

    ann_searcher_->Init(to_metadata, norm_order);

    if (disk_based_) {
        GenerateWeightsFromSearcherStreaming(*ann_searcher_, from_metadata, disk_query_batch_size_, weights);
        auto* disk_searcher = dynamic_cast<DiskBasedQalshAnnSearcher*>(ann_searcher_.get());
        if (disk_searcher != nullptr) {
            disk_searcher->PrintProfile();
        }
        if (use_cache) {
            WriteQalshWeightsCache(from_metadata, norm_order, weights);
        }
        return weights;
    }

#ifdef QALSH4C_USE_OPENMP
    std::vector<Point> query_points =
        PointSet::LoadPointsFromFile(from_metadata.file_path, from_metadata.num_points, from_metadata.num_dimensions);

#pragma omp parallel
    {
        std::unique_ptr<AnnSearcher> local_searcher = ann_searcher_->Clone();

#pragma omp for schedule(dynamic)
        for (long long query_index = 0; query_index < static_cast<long long>(from_metadata.num_points); ++query_index) {
            const auto point_id = static_cast<std::size_t>(query_index);
            weights[point_id] = local_searcher->Search(query_points[point_id]).distance;
        }
    }
#else
    std::vector<Point> query_points =
        PointSet::LoadPointsFromFile(from_metadata.file_path, from_metadata.num_points, from_metadata.num_dimensions);

    for (unsigned int i = 0; i < from_metadata.num_points; i++) {
        weights[i] = ann_searcher_->Search(query_points[i]).distance;
    }
#endif

    if (use_cache) {
        WriteQalshWeightsCache(from_metadata, norm_order, weights);
    }

    return weights;
}
