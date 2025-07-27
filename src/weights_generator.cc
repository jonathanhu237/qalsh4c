#include "weights_generator.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <optional>
#include <variant>
#include <vector>

#include "ann_searcher.h"
#include "point_set.h"
#include "quadtree.h"
#include "types.h"
#include "utils.h"

// --------------------------------------------------
// UniformWeightsGenerator Implementation
// --------------------------------------------------
std::vector<double> UniformWeightsGenerator::Generate(const std::filesystem::path& dataset_directory) {
    // Load dataset metadata
    DatasetMetadata metadata;
    metadata.Load(dataset_directory / "metadata.toml");

    std::vector<double> weights(metadata.query_num_points, 1.0);
    return weights;
}

// --------------------------------------------------
// QalshWeightsGenerator Implementation
// --------------------------------------------------
std::vector<double> QalshWeightsGenerator::Generate(const std::filesystem::path& dataset_directory) {
    // Load dataset metadata.
    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory / "metadata.toml");

    auto query_set_reader{PointSetReaderFactory::Create(dataset_directory / "query.bin", dataset_metadata.data_type,
                                                        dataset_metadata.query_num_points,
                                                        dataset_metadata.num_dimensions)};

    // Generate weights based on QALSH algorithm.
    std::vector<double> weights(dataset_metadata.query_num_points);
    QalshAnnSearcher qalsh_ann_searcher;
    qalsh_ann_searcher.Init(dataset_directory / "qalsh_index");
    for (unsigned int i = 0; i < dataset_metadata.query_num_points; i++) {
        PointVariant query_point = query_set_reader->GetPoint(i);
        AnnResult result = qalsh_ann_searcher.Search(query_point);
        weights[i] = result.distance;  // Use the distance as the weight
    }

    return weights;
}

// --------------------------------------------------
// QuadtreeWeightsGenerator Implementation
// --------------------------------------------------
std::vector<double> QuadtreeWeightsGenerator::Generate(const std::filesystem::path& dataset_directory) {
    // Load dataset metadata.
    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory / "metadata.toml");

    // Load Quadtree configuration.
    QuadtreeConfiguration config;
    config.Load(dataset_directory / "quadtree_index" / "config.toml");

    // Construct the Quadtree.
    Quadtree quadtree(dataset_directory, config.max_level, false);
    quadtree.Build();

    // Get the DFS order from the Quadtree.
    std::vector<unsigned int> dfs_order = quadtree.DepthFirstSearch();

    // --------------------------------------------------
    // Find left and right neighbors in the DFS order.
    // --------------------------------------------------
    unsigned int base_num_points = dataset_metadata.base_num_points;
    unsigned int query_num_points = dataset_metadata.query_num_points;
    unsigned int total_num_points = base_num_points + query_num_points;
    std::vector<unsigned int> left_neighbors(total_num_points);
    std::vector<unsigned int> right_neighbors(total_num_points);

    if (base_num_points == 0) {
        spdlog::error("Cannot find neighbors: The base set is empty.");
        // The program will terminate automatically here, thus we don't need to return anything.
    }

    // ----- Left pass -----
    std::optional<unsigned int> last_seen_base_idx;

    // Find the initial base point to handle any query points at the beginning.
    for (unsigned int point_idx : dfs_order) {
        if (!last_seen_base_idx.has_value() && point_idx < base_num_points) {
            last_seen_base_idx = point_idx;
            break;
        }
    }

    // Propagate the last-seen base index from left to right.
    for (unsigned int i = 0; i < total_num_points; i++) {
        if (dfs_order[i] < base_num_points) {
            last_seen_base_idx = dfs_order[i];
        }
        left_neighbors[i] = last_seen_base_idx.value();
    }

    // ----- Right pass -----
    last_seen_base_idx.reset();  // Reset for the right pass.
    // Find the initial base point from the end to handle query points at the end.
    for (int i = static_cast<int>(total_num_points) - 1; i >= 0; i--) {
        if (!last_seen_base_idx.has_value() && dfs_order[static_cast<unsigned int>(i)] < base_num_points) {
            last_seen_base_idx = dfs_order[static_cast<unsigned int>(i)];
            break;
        }
    }

    // Propagate the last-seen base index from right to left.
    for (int i = static_cast<int>(total_num_points) - 1; i >= 0; i--) {
        if (dfs_order[static_cast<unsigned int>(i)] < base_num_points) {
            last_seen_base_idx = dfs_order[static_cast<unsigned int>(i)];
        }
        right_neighbors[static_cast<unsigned int>(i)] = last_seen_base_idx.value();
    }

    // --------------------------------------------------
    // Generate the weights using the neighbors.
    // --------------------------------------------------
    // Create an inverse map for O(1) lookups of a point's position in the DFS order.
    std::vector<unsigned int> point_pos_in_dfs(total_num_points);
    for (unsigned int i = 0; i < total_num_points; i++) {
        point_pos_in_dfs[dfs_order[i]] = i;
    }

    // Initialize the base reader and the query reader
    auto base_reader = PointSetReaderFactory::Create(dataset_directory / "base.bin", dataset_metadata.data_type,
                                                     dataset_metadata.base_num_points, dataset_metadata.num_dimensions);
    auto query_reader =
        PointSetReaderFactory::Create(dataset_directory / "query.bin", dataset_metadata.data_type,
                                      dataset_metadata.query_num_points, dataset_metadata.num_dimensions);

    // Calculate the weights
    std::vector<double> weights(query_num_points);
    for (unsigned int i = 0; i < query_num_points; i++) {
        unsigned int query_idx_in_dfs = base_num_points + i;
        unsigned int dfs_pos = point_pos_in_dfs[query_idx_in_dfs];

        // Get its left and right neighbors.
        unsigned int left_neighbor = left_neighbors[dfs_pos];
        unsigned int right_neighbor = right_neighbors[dfs_pos];

        // Get the actual points.
        PointVariant query = query_reader->GetPoint(i);
        PointVariant left_point = base_reader->GetPoint(left_neighbor);
        PointVariant right_point = base_reader->GetPoint(right_neighbor);

        // Calculate the weights.
        std::visit(
            [&](const auto& concrete_query, const auto& concrete_left, const auto& concrete_right) {
                double dist_left = Utils::CalculateL1Distance(concrete_query, concrete_left);
                double dist_right = Utils::CalculateL1Distance(concrete_query, concrete_right);
                weights[i] = std::min(dist_left, dist_right);
            },
            query, left_point, right_point);
    }

    // Return
    return weights;
}