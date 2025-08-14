#include "command.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <memory>
#include <random>
#include <ratio>
#include <utility>

#include "b_plus_tree.h"
#include "estimator.h"
#include "global.h"
#include "utils.h"

// --------------------------------------------------
// IndexCommand Implementation
// --------------------------------------------------
IndexCommand::IndexCommand(double norm_order, double approximation_ratio, unsigned int page_size,
                           std::filesystem::path dataset_directory)
    : norm_order_(norm_order),
      approximation_ratio_(approximation_ratio),
      page_size_(page_size),
      dataset_directory_(std::move(dataset_directory)),
      gen_(Utils::CreateSeededGenerator()) {}

void IndexCommand::Execute() {
    // Read dataset metadata.
    DatasetMetadata dataset_metadata = Utils::LoadDatasetMetadata(dataset_directory_ / "metadata.json");

    // Begin to record the time and memory.
    auto start = std::chrono::high_resolution_clock::now();
    double memory_before = Utils::GetMemoryUsage();

    // Build index for point set B.
    BuildIndex(PointSetMetadata{.file_path = dataset_directory_ / "B.bin",
                                .num_points = dataset_metadata.num_points_b,
                                .num_dimensions = dataset_metadata.num_dimensions},
               dataset_directory_ / "index" / std::format("l{}", norm_order_) / "B");

    // Build index for point set A.
    BuildIndex(PointSetMetadata{.file_path = dataset_directory_ / "A.bin",
                                .num_points = dataset_metadata.num_points_a,
                                .num_dimensions = dataset_metadata.num_dimensions},
               dataset_directory_ / "index" / std::format("l{}", norm_order_) / "A");

    // End to record the time and memory.
    auto end = std::chrono::high_resolution_clock::now();
    double memory_after = Utils::GetMemoryUsage();

    // Output the result.
    std::cout << std::format(
        "Time Consumed: {:.3f} ms\n"
        "Memory Usage: {:.2f} MB\n",
        std::chrono::duration<double, std::milli>(end - start).count(), memory_after - memory_before);
}

void IndexCommand::BuildIndex(const PointSetMetadata& point_set_metadata,
                              const std::filesystem::path& index_directory) {
    // Regularize the QALSH configuration
    QalshConfig config{.approximation_ratio = approximation_ratio_, .page_size = page_size_};
    Utils::RegularizeQalshConfig(config, point_set_metadata.num_points, norm_order_);

    // Print the QalshConfig parameters.
    spdlog::info(
        "QALSH Configuration:\n"
        "\tApproximation Ratio: {}\n"
        "\tBucket Width: {}\n"
        "\tError Probability: {}\n"
        "\tNumber of Hash Tables: {}\n"
        "\tCollision Threshold: {}\n"
        "\tPage Size: {}",
        config.approximation_ratio, config.bucket_width, config.error_probability, config.num_hash_tables,
        config.collision_threshold, config.page_size);

    // Create the index directory if it does not exist.
    if (!std::filesystem::exists(index_directory)) {
        spdlog::info("Creating index directory: {}", index_directory.string());
        std::filesystem::create_directories(index_directory);
    }

    // Save the QALSH configuration.
    spdlog::info("Saving QALSH configuration...");
    Utils::SaveQalshConfig(config, index_directory / "config.json");

    // Create the B+ tree directory.
    std::filesystem::path b_plus_tree_directory = index_directory / "b_plus_trees";
    if (!std::filesystem::exists(b_plus_tree_directory)) {
        spdlog::info("Creating B+ tree directory: {}", b_plus_tree_directory.string());
        std::filesystem::create_directories(b_plus_tree_directory);
    }

    // Generate the dot vectors.
    spdlog::info("Generating dot vectors for {} hash tables...", config.num_hash_tables);
    std::vector<std::vector<double>> dot_vectors(config.num_hash_tables);
    std::cauchy_distribution<double> standard_cauchy_dist(0.0, 1.0);
    for (unsigned int i = 0; i < config.num_hash_tables; i++) {
        dot_vectors[i].reserve(point_set_metadata.num_dimensions);
        std::ranges::generate_n(std::back_inserter(dot_vectors[i]), point_set_metadata.num_dimensions,
                                [&]() { return standard_cauchy_dist(gen_); });
    }

    // Save the dot product vectors
    spdlog::info("Saving dot product vectors...");
    std::ofstream ofs(index_directory / "dot_vectors.bin", std::ios::binary);
    if (!ofs.is_open()) {
        spdlog::error(
            std::format("Failed to open file for writing: {}", (index_directory / "dot_vectors.bin").string()));
    }
    for (unsigned int i = 0; i < config.num_hash_tables; i++) {
        ofs.write(reinterpret_cast<const char*>(dot_vectors[i].data()),
                  static_cast<std::streamsize>(dot_vectors[i].size() * sizeof(Coordinate)));
    }

    // Open the point set file
    std::ifstream base_file(point_set_metadata.file_path, std::ios::binary);
    if (!base_file.is_open()) {
        spdlog::error("Failed to open base file: {}", point_set_metadata.file_path.string());
        return;
    }

    // Build the B+ trees for each hash table.
    spdlog::info("Building B+ trees for each hash table...");
    std::vector<std::vector<DotProductPointIdPair>> data(config.num_hash_tables);
    for (unsigned int i = 0; i < point_set_metadata.num_points; i++) {
        Point point = Utils::ReadPoint(base_file, point_set_metadata.num_dimensions, i);
        for (unsigned int j = 0; j < config.num_hash_tables; j++) {
            double dot_product = Utils::DotProduct(point, dot_vectors[j]);
            data[j].emplace_back(DotProductPointIdPair{.dot_product = dot_product, .point_id = i});
        }
    }
    for (unsigned int i = 0; i < config.num_hash_tables; i++) {
        // Sort the dot products.
        std::ranges::sort(data[i], {}, &DotProductPointIdPair::dot_product);

        // Bulk load the B+ tree.
        BPlusTreeBulkLoader bulk_loader(b_plus_tree_directory / std::format("{}.bin", i), config.page_size);
        bulk_loader.Build(data[i]);
    }
}

// --------------------------------------------------
// EstimateCommand Implementation
// --------------------------------------------------
EstimateCommand::EstimateCommand(std::unique_ptr<Estimator> estimator, double norm_order,
                                 std::filesystem::path dataset_directory, bool in_memory)
    : estimator_(std::move(estimator)),
      norm_order_(norm_order),
      dataset_directory_(std::move(dataset_directory)),
      in_memory_(in_memory) {}

void EstimateCommand::Execute() {
    // Load dataset metadata
    spdlog::info("Loading dataset metadata...");
    DatasetMetadata dataset_metadata = Utils::LoadDatasetMetadata(dataset_directory_ / "metadata.json");

    // Construct point set metadata
    PointSetMetadata point_set_metadata_a = {
        .file_path = dataset_directory_ / "A.bin",
        .num_points = dataset_metadata.num_points_a,
        .num_dimensions = dataset_metadata.num_dimensions,
    };
    PointSetMetadata point_set_metadata_b = {
        .file_path = dataset_directory_ / "B.bin",
        .num_points = dataset_metadata.num_points_b,
        .num_dimensions = dataset_metadata.num_dimensions,
    };

    // Calculate the distance from A to B
    auto start = std::chrono::high_resolution_clock::now();
    double memory_before = Utils::GetMemoryUsage();
    spdlog::info("Calculating the distance from A to B...");
    double distance_ab = estimator_->EstimateDistance(point_set_metadata_a, point_set_metadata_b, in_memory_);

    // Calculate the distance from B to A
    spdlog::info("Calculating the distance from B to A...");
    double distance_ba = estimator_->EstimateDistance(point_set_metadata_b, point_set_metadata_a, in_memory_);
    auto end = std::chrono::high_resolution_clock::now();
    double memory_after = Utils::GetMemoryUsage();

    // Output the result.
    double estimation = distance_ab + distance_ba;
    double ground_truth{0.0};

    // NOLINTBEGIN(readability-magic-numbers)
    if (std::abs(norm_order_ - 1.0) < Global::kEpsilon) {
        ground_truth = dataset_metadata.chamfer_distance_l1;
    } else if (std::abs(norm_order_ - 2.0) < Global::kEpsilon) {
        ground_truth = dataset_metadata.chamfer_distance_l2;
    } else {
        spdlog::error("Unsupported norm order: {}", norm_order_);
    }
    // NOLINTEND(readability-magic-numbers)

    double relative_error_percentage =
        std::fabs(estimation - ground_truth) / ground_truth * 100;  // NOLINT: readability-magic-numbers
    double elapsed_time = std::chrono::duration<double, std::milli>(end - start).count();
    double memory_usage = memory_after - memory_before;

    std::cout << std::format(
        "Time Consumed: {:.3f} ms\n"
        "Memory Usage: {:.2f} MB\n"
        "Relative Error: {:.2f}%\n",
        elapsed_time, memory_usage, relative_error_percentage);
}