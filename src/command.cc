#include "command.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <ratio>
#include <utility>

#include "b_plus_tree.h"
#include "estimator.h"
#include "external_sorter.h"
#include "global.h"
#include "log.h"
#include "point_set.h"
#include "utils.h"

// --------------------------------------------------
// IndexCommand Implementation
// --------------------------------------------------
IndexCommand::IndexCommand(double norm_order, double approximation_ratio, unsigned int page_size,
                           std::filesystem::path dataset_directory, bool fast)
    : norm_order_(norm_order),
      approximation_ratio_(approximation_ratio),
      page_size_(page_size),
      dataset_directory_(std::move(dataset_directory)),
      fast_(fast),
      gen_(Random::SeededGenerator()) {}

void IndexCommand::Execute() {
    // Read dataset metadata.
    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory_ / "metadata.json");

    // Begin to record the time and memory.
    auto start = std::chrono::high_resolution_clock::now();
    double memory_before = Memory::Usage();

    // Build index for point set A.
    BuildIndex(PointSetMetadata{.file_path = dataset_directory_ / "A.bin",
                                .num_points = dataset_metadata.num_points_a,
                                .num_dimensions = dataset_metadata.num_dimensions},
               dataset_directory_ / "index" / std::format("l{}", norm_order_) / "A");

    // Build index for point set B.
    BuildIndex(PointSetMetadata{.file_path = dataset_directory_ / "B.bin",
                                .num_points = dataset_metadata.num_points_b,
                                .num_dimensions = dataset_metadata.num_dimensions},
               dataset_directory_ / "index" / std::format("l{}", norm_order_) / "B");

    // End to record the time and memory.
    auto end = std::chrono::high_resolution_clock::now();
    double memory_after = Memory::Usage();

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
    config.Regularize(point_set_metadata.num_points, norm_order_);

    // Print the QalshConfig parameters.
    Log::info(
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
        Log::info("Creating index directory: {}", index_directory.string());
        std::filesystem::create_directories(index_directory);
    }

    // Save the QALSH configuration.
    Log::info("Saving QALSH configuration...");
    config.Save(index_directory / "config.json");

    // Create the B+ tree directory.
    std::filesystem::path b_plus_tree_directory = index_directory / "b_plus_trees";
    if (!std::filesystem::exists(b_plus_tree_directory)) {
        Log::info("Creating B+ tree directory: {}", b_plus_tree_directory.string());
        std::filesystem::create_directories(b_plus_tree_directory);
    }

    // Generate the dot vectors.
    Log::info("Generating dot vectors for {} hash tables...", config.num_hash_tables);
    std::vector<std::vector<double>> dot_vectors(config.num_hash_tables);
    std::function<double()> generator;

    if (std::abs(norm_order_ - 1.0) < Global::kEpsilon) {
        std::cauchy_distribution<double> dist(0.0, 1.0);
        generator = [dist, this]() mutable { return dist(gen_); };
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    else if (std::abs(norm_order_ - 2.0) < Global::kEpsilon) {
        std::normal_distribution<double> dist(0.0, 1.0);
        generator = [dist, this]() mutable { return dist(gen_); };
    } else {
        Log::error("Unsupported norm order: {}", norm_order_);
    }

    for (unsigned int i = 0; i < config.num_hash_tables; i++) {
        dot_vectors[i].reserve(point_set_metadata.num_dimensions);
        std::ranges::generate_n(std::back_inserter(dot_vectors[i]), point_set_metadata.num_dimensions, generator);
    }

    // Save the dot product vectors
    Log::info("Saving dot product vectors...");
    std::ofstream ofs(index_directory / "dot_vectors.bin", std::ios::binary);
    if (!ofs.is_open()) {
        Log::error("Failed to open file for writing: {}", (index_directory / "dot_vectors.bin").string());
    }
    for (unsigned int i = 0; i < config.num_hash_tables; i++) {
        ofs.write(reinterpret_cast<const char*>(dot_vectors[i].data()),
                  static_cast<std::streamsize>(dot_vectors[i].size() * sizeof(Coordinate)));
    }

    // Build the B+ trees for each hash table.
    Log::info("Building B+ trees for each hash table...");
    if (fast_) {
        PointSet base;
        base.Init(point_set_metadata);

        std::vector<std::vector<DotProductPointIdPair>> data(config.num_hash_tables);
        for (unsigned int i = 0; i < point_set_metadata.num_points; i++) {
            Point point = base.ReadPoint(i);
            for (unsigned int j = 0; j < config.num_hash_tables; j++) {
                double dot_product = Calc::DotProduct(point, dot_vectors[j]);
                data[j].emplace_back(DotProductPointIdPair{.dot_product = dot_product, .point_id = i});
            }
        }
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
        for (long long hash_table_index = 0; hash_table_index < static_cast<long long>(config.num_hash_tables);
             ++hash_table_index) {
            const unsigned int hash_table_id = static_cast<unsigned int>(hash_table_index);
            std::ranges::sort(data[hash_table_id], DotProductPointIdPairLess{});

            BPlusTreeBulkLoader bulk_loader(b_plus_tree_directory / std::format("{}.bin", hash_table_id),
                                            config.page_size);
            bulk_loader.Build(data[hash_table_id]);
        }
#else
        for (unsigned int i = 0; i < config.num_hash_tables; i++) {
            std::ranges::sort(data[i], DotProductPointIdPairLess{});

            BPlusTreeBulkLoader bulk_loader(b_plus_tree_directory / std::format("{}.bin", i), config.page_size);
            bulk_loader.Build(data[i]);
        }
#endif
        return;
    }

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
    for (long long hash_table_index = 0; hash_table_index < static_cast<long long>(config.num_hash_tables);
         ++hash_table_index) {
        const unsigned int hash_table_id = static_cast<unsigned int>(hash_table_index);
        PointSet base;
        base.Init(point_set_metadata);

        ExternalSorter sorter(b_plus_tree_directory, std::format("ht{}", hash_table_id));
        for (unsigned int point_id = 0; point_id < point_set_metadata.num_points; point_id++) {
            Point point = base.ReadPoint(point_id);
            double dot_product = Calc::DotProduct(point, dot_vectors[hash_table_id]);
            sorter.Add(DotProductPointIdPair{.dot_product = dot_product, .point_id = point_id});
        }

        sorter.Finish();

        BPlusTreeBulkLoader bulk_loader(b_plus_tree_directory / std::format("{}.bin", hash_table_id), config.page_size);
        bulk_loader.BuildFromStream(point_set_metadata.num_points, sorter.MakeMergeReader());
    }
#else
    for (unsigned int hash_table_id = 0; hash_table_id < config.num_hash_tables; hash_table_id++) {
        PointSet base;
        base.Init(point_set_metadata);

        ExternalSorter sorter(b_plus_tree_directory, std::format("ht{}", hash_table_id));
        for (unsigned int point_id = 0; point_id < point_set_metadata.num_points; point_id++) {
            Point point = base.ReadPoint(point_id);
            double dot_product = Calc::DotProduct(point, dot_vectors[hash_table_id]);
            sorter.Add(DotProductPointIdPair{.dot_product = dot_product, .point_id = point_id});
        }

        sorter.Finish();

        BPlusTreeBulkLoader bulk_loader(b_plus_tree_directory / std::format("{}.bin", hash_table_id), config.page_size);
        bulk_loader.BuildFromStream(point_set_metadata.num_points, sorter.MakeMergeReader());
    }
#endif
}

// --------------------------------------------------
// EstimateCommand Implementation
// --------------------------------------------------
EstimateCommand::EstimateCommand(std::unique_ptr<Estimator> estimator, double norm_order,
                                 std::filesystem::path dataset_directory)
    : estimator_(std::move(estimator)), norm_order_(norm_order), dataset_directory_(std::move(dataset_directory)) {}

void EstimateCommand::Execute() {
    // Load dataset metadata
    Log::info("Loading dataset metadata...");
    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory_ / "metadata.json");

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
    double memory_before = Memory::Usage();
    Log::info("Calculating the distance from A to B...");
    double distance_ab = estimator_->EstimateDistance(point_set_metadata_a, point_set_metadata_b, norm_order_);

    // Calculate the distance from B to A
    Log::info("Calculating the distance from B to A...");
    double distance_ba = estimator_->EstimateDistance(point_set_metadata_b, point_set_metadata_a, norm_order_);
    auto end = std::chrono::high_resolution_clock::now();
    double memory_after = Memory::Usage();

    // Output the result.
    double estimation = distance_ab + distance_ba;
    double ground_truth{0.0};

    // NOLINTBEGIN(readability-magic-numbers)
    if (std::abs(norm_order_ - 1.0) < Global::kEpsilon) {
        ground_truth = dataset_metadata.chamfer_distance_l1;
    } else if (std::abs(norm_order_ - 2.0) < Global::kEpsilon) {
        ground_truth = dataset_metadata.chamfer_distance_l2;
    } else {
        Log::error("Unsupported norm order: {}", norm_order_);
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
