#include "weights_generator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ann_searcher.h"
#include "global.h"
#include "log.h"
#include "point_set.h"
#include "utils.h"

namespace {

using CellKey = std::vector<std::uint64_t>;

constexpr unsigned int kBitsPerWord = 64;
constexpr double kMinimumWeightFloor = 1e-12;
constexpr double kWeightFloorScale = 1e-12;

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

std::pair<Point, Point> ComputeBounds(const std::vector<Point>& points, unsigned int num_dimensions) {
    Point lower(num_dimensions, std::numeric_limits<double>::infinity());
    Point upper(num_dimensions, -std::numeric_limits<double>::infinity());

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
        midpoint[dimension] = 0.5 * (lower[dimension] + upper[dimension]);
    }

    std::unordered_map<CellKey, std::size_t, CellKeyHash> subcluster_by_key;
    std::vector<CellKey> cell_order;
    std::vector<std::vector<unsigned int>> subclusters;

    for (unsigned int point_id : cluster) {
        CellKey key = ComputeCellKey(points[point_id], midpoint);
        auto [it, inserted] = subcluster_by_key.emplace(key, subclusters.size());
        if (inserted) {
            cell_order.push_back(key);
            subclusters.emplace_back();
        }
        subclusters[it->second].push_back(point_id);
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
        Log::error("Point set dimensions do not match: {} and {}", from_metadata.num_dimensions,
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
    std::move(from_points.begin(), from_points.end(), std::back_inserter(combined_points));
    std::move(to_points.begin(), to_points.end(), std::back_inserter(combined_points));
    return combined_points;
}

void MakeWeightsPositive(std::vector<double>& weights) {
    double min_positive_weight = std::numeric_limits<double>::infinity();
    for (double weight : weights) {
        if (std::isfinite(weight) && weight > 0.0) {
            min_positive_weight = std::min(min_positive_weight, weight);
        }
    }

    if (!std::isfinite(min_positive_weight)) {
        std::ranges::fill(weights, 1.0);
        return;
    }

    const double weight_floor = std::max(kMinimumWeightFloor, min_positive_weight * kWeightFloorScale);
    for (double& weight : weights) {
        if (!std::isfinite(weight) || weight <= 0.0) {
            weight = weight_floor;
        }
    }
}

std::filesystem::path QuadtreeWeightsPath(const PointSetMetadata& from_metadata, double norm_order,
                                          unsigned int max_levels) {
    const std::filesystem::path parent_directory = from_metadata.file_path.parent_path();
    const std::string stem = from_metadata.file_path.stem().string();
    return parent_directory / std::format("l{}_quadtree_l{}_weights_{}.bin", norm_order, max_levels, stem);
}

bool TryReadWeightsCache(const std::filesystem::path& weights_path, std::vector<double>& weights) {
    const auto expected_size = static_cast<std::uintmax_t>(sizeof(double) * weights.size());
    if (!std::filesystem::exists(weights_path)) {
        return false;
    }

    if (std::filesystem::file_size(weights_path) != expected_size) {
        Log::warn("Cache file size mismatch for {}. Regenerating weights...", weights_path.string());
        return false;
    }

    std::ifstream ifs(weights_path, std::ios::binary);
    if (!ifs.is_open()) {
        Log::warn("Failed to open cache file {}. Regenerating weights...", weights_path.string());
        return false;
    }

    ifs.read(reinterpret_cast<char*>(weights.data()), static_cast<std::streamoff>(expected_size));
    if (ifs.gcount() != static_cast<std::streamsize>(expected_size)) {
        Log::warn("Failed to read complete cache file {}. Regenerating weights...", weights_path.string());
        return false;
    }

    return true;
}

void WriteWeightsCache(const std::filesystem::path& weights_path, const std::vector<double>& weights) {
    std::ofstream ofs(weights_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        Log::error("Failed to open file for writing: {}", weights_path.string());
    }
    ofs.write(reinterpret_cast<const char*>(weights.data()),
              static_cast<std::streamoff>(sizeof(double) * weights.size()));
}

}  // namespace

// --------------------------------------------------
// UniformWeightsGenerator Implementation
// --------------------------------------------------
bool UniformWeightsGenerator::RequiresIndex() const {
    return false;
}

std::vector<double> UniformWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                      [[maybe_unused]] const PointSetMetadata& to_metadata,
                                                      [[maybe_unused]] double norm_order,
                                                      [[maybe_unused]] bool use_cache) {
    std::vector<double> weights(from_metadata.num_points, 1.0);
    return weights;
}

// --------------------------------------------------
// QalshWeightsGenerator Implementation
// --------------------------------------------------
QalshWeightsGenerator::QalshWeightsGenerator() : ann_searcher_(std::make_unique<QalshAnnSearcher>()) {}

bool QalshWeightsGenerator::RequiresIndex() const {
    return true;
}

std::vector<double> QalshWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                    const PointSetMetadata& to_metadata, double norm_order,
                                                    bool use_cache) {
    std::vector<double> weights(from_metadata.num_points);
    std::filesystem::path parent_directory = from_metadata.file_path.parent_path();
    std::string stem = from_metadata.file_path.stem();
    std::filesystem::path weights_path = parent_directory / std::format("l{}_qalsh_weights_{}.bin", norm_order, stem);

    if (use_cache) {
        Log::warn("The weights will be read from the cache file if it exists.");
        if (std::filesystem::exists(weights_path)) {
            std::ifstream ifs(weights_path);
            ifs.read(reinterpret_cast<char*>(weights.data()),
                     static_cast<std::streamoff>(sizeof(double) * from_metadata.num_points));
            return weights;
        }
        Log::warn("Cache file doesn't exist. Generating the new one...");
    }

    // Generate weights based on QALSH algorithm.
    Log::info("Generating weights using QALSH...");
    ann_searcher_->Init(to_metadata, norm_order);

    PointSet base;
    base.Init(from_metadata);

    for (unsigned int i = 0; i < from_metadata.num_points; i++) {
        weights[i] = ann_searcher_->Search(base.ReadPoint(i)).distance;
    }

    if (use_cache) {
        std::ofstream ofs(weights_path);
        ofs.write(reinterpret_cast<const char*>(weights.data()),
                  static_cast<std::streamoff>(sizeof(double) * from_metadata.num_points));
    }

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

std::vector<double> QuadtreeWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                       const PointSetMetadata& to_metadata, double norm_order,
                                                       bool use_cache) {
    if (std::abs(norm_order - 1.0) >= Global::kEpsilon && std::abs(norm_order - 2.0) >= Global::kEpsilon) {
        Log::error("Unsupported norm order for quadtree sampling: {}", norm_order);
    }

    std::vector<double> weights(from_metadata.num_points);
    const std::filesystem::path weights_path = QuadtreeWeightsPath(from_metadata, norm_order, max_levels_);

    if (use_cache) {
        Log::warn("The quadtree weights will be read from the cache file if it exists.");
        if (TryReadWeightsCache(weights_path, weights)) {
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
        const double left_distance = Calc::LpDistance(combined_points[source_id], combined_points[left_target_id],
                                                      norm_order);
        const double right_distance = Calc::LpDistance(combined_points[source_id], combined_points[right_target_id],
                                                       norm_order);
        weights[source_id] = std::min(left_distance, right_distance);
    }

    MakeWeightsPositive(weights);

    if (use_cache) {
        WriteWeightsCache(weights_path, weights);
    }

    return weights;
}
