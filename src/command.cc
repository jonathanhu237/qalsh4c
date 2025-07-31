#include "command.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <memory>
#include <random>
#include <ratio>
#include <utility>

#include "b_plus_tree.h"
#include "dataset_metadata.h"
#include "estimator.h"
#include "point_set.h"
#include "utils.h"

// --------------------------------------------------
// GenerateDatasetCommand Implementation
// --------------------------------------------------
GenerateDatasetCommand::GenerateDatasetCommand(DatasetMetadata dataset_metadata, double left_boundary,
                                               double right_boundary, std::filesystem::path output_directory,
                                               bool in_memory)
    : dataset_metadata_(dataset_metadata),
      left_boundary_(left_boundary),
      right_boundary_(right_boundary),
      output_directory_(std::move(output_directory)),
      in_memory_(in_memory),
      gen_(std::random_device{}()) {}

void GenerateDatasetCommand::Execute() {
    // Create the directory if it does not exist.
    if (!std::filesystem::exists(output_directory_)) {
        spdlog::info("Creating dataset directory: {}", output_directory_.string());
        std::filesystem::create_directory(output_directory_);
    }

    // Generate the point sets.
    spdlog::info("Generating point set A...");
    GeneratePointSet(output_directory_ / "A.bin", dataset_metadata_.num_points_a);

    spdlog::info("Generating point set B...");
    GeneratePointSet(output_directory_ / "B.bin", dataset_metadata_.num_points_b);

    // We must save the metadata first since it will be used by the estimator (update the chamfer distance later)
    dataset_metadata_.Save(output_directory_ / "metadata.json");

    // Calculate the Chamfer distance between the base and query sets
    spdlog::info("Calculating Chamfer distance between base and query sets...");
    AnnEstimator ann_estimator(std::make_unique<LinearScanAnnSearcher>());
    dataset_metadata_.chamfer_distance = ann_estimator.Estimate(output_directory_);

    // Save the updated metadata
    dataset_metadata_.Save(output_directory_ / "metadata.json");
}

void GenerateDatasetCommand::GeneratePointSet(const std::string& point_set_file_path, unsigned int num_points) {
    std::unique_ptr<PointSetWriter> pointset;
    if (in_memory_) {
        pointset = std::make_unique<InMemoryPointSetWriter>(point_set_file_path);
    } else {
        pointset = std::make_unique<DiskPointSetWriter>(point_set_file_path);
    }

    std::uniform_real_distribution<double> dist(left_boundary_, right_boundary_);
    for (unsigned int i = 0; i < num_points; i++) {
        Point point(dataset_metadata_.num_dimensions);
        std::ranges::generate(point, [&]() { return dist(gen_); });
        pointset->AddPoint(point);
    }

    pointset->Flush();
}

// --------------------------------------------------
// IndexCommand Implementation
// --------------------------------------------------
IndexCommand::IndexCommand(double approximation_ratio, unsigned int page_size, std::filesystem::path dataset_directory)
    : dataset_directory_(std::move(dataset_directory)), gen_(std::random_device{}()) {
    qalsh_config_.approximation_ratio = approximation_ratio;
    qalsh_config_.page_size = page_size;
}

void IndexCommand::Execute() {
    // Read dataset metadata.
    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory_ / "metadata.json");

    // Build index for point set A.
    BuildIndex(PointSetMetadata{.file_path = dataset_directory_ / "A.bin",
                                .num_points = dataset_metadata.num_points_a,
                                .num_dimensions = dataset_metadata.num_dimensions},
               dataset_directory_ / "index" / "A");

    // Build index for point set B.
    BuildIndex(PointSetMetadata{.file_path = dataset_directory_ / "B.bin",
                                .num_points = dataset_metadata.num_points_b,
                                .num_dimensions = dataset_metadata.num_dimensions},
               dataset_directory_ / "index" / "B");
}

void IndexCommand::BuildIndex(const PointSetMetadata& point_set_metadata,
                              const std::filesystem::path& index_directory) {
    // Regularize the QALSH configuration
    qalsh_config_.Regularize(point_set_metadata.num_points);

    // Initialize the point set reader.
    auto point_set = std::make_unique<DiskPointSetReader>(point_set_metadata);

    // Create the index directory if it does not exist.
    if (!std::filesystem::exists(index_directory)) {
        spdlog::info("Creating index directory: {}", index_directory.string());
        std::filesystem::create_directories(index_directory);
    }

    // Save the QALSH configuration.
    spdlog::info("Saving QALSH configuration...");
    qalsh_config_.Save(index_directory / "config.json");

    // Create the B+ tree directory.
    std::filesystem::path b_plus_tree_directory = index_directory / "b_plus_trees";
    if (!std::filesystem::exists(b_plus_tree_directory)) {
        spdlog::info("Creating B+ tree directory: {}", b_plus_tree_directory.string());
        std::filesystem::create_directories(b_plus_tree_directory);
    }

    // Generate the dot vectors.
    spdlog::info("Generating dot vectors for {} hash tables...", qalsh_config_.num_hash_tables);
    std::cauchy_distribution<double> standard_cauchy_dist(0.0, 1.0);
    std::vector<std::vector<double>> dot_vectors(qalsh_config_.num_hash_tables,
                                                 std::vector<double>(point_set->get_num_dimensions()));
    for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
        std::ranges::generate(dot_vectors[i], [&]() { return standard_cauchy_dist(gen_); });
    }

    // Index the point set.
    spdlog::info("Indexing the pooint set...");
    for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
        spdlog::debug("Indexing hash table {}/{}", i + 1, qalsh_config_.num_hash_tables);
        std::vector<DotProductPointIdPair> dot_products_with_id(point_set->get_num_points());
        for (unsigned int j = 0; j < point_set->get_num_points(); j++) {
            // Calculate the dot product for each point in the base set
            Point point = point_set->GetPoint(j);
            double dot_product = Utils::DotProduct(point, dot_vectors[i]);
            dot_products_with_id[j] = {
                .dot_product = dot_product,
                .point_id = j,
            };
        }

        // Sort the dot products
        std::ranges::sort(dot_products_with_id,
                          [](const auto& a, const auto& b) { return a.dot_product < b.dot_product; });

        // Bulk load the B+ tree with sorted dot products
        BPlusTreeBulkLoader bulk_loader(b_plus_tree_directory / std::format("{}.bin", i), qalsh_config_.page_size);
        bulk_loader.Build(dot_products_with_id);
    }

    // Save the dot product vectors
    spdlog::info("Saving dot product vectors...");
    std::ofstream ofs(index_directory / "dot_vectors.bin", std::ios::binary);
    if (!ofs.is_open()) {
        spdlog::error(
            std::format("Failed to open file for writing: {}", (index_directory / "dot_vectors.bin").string()));
    }
    for (const auto& vec : dot_vectors) {
        ofs.write(reinterpret_cast<const char*>(vec.data()), static_cast<std::streamsize>(vec.size() * sizeof(double)));
    }
}

// --------------------------------------------------
// EstimateCommand Implementation
// --------------------------------------------------
EstimateCommand::EstimateCommand(std::unique_ptr<Estimator> estimator, std::filesystem::path dataset_directory,
                                 bool in_memory)
    : estimator_(std::move(estimator)), dataset_directory_(std::move(dataset_directory)), in_memory_(in_memory) {}

void EstimateCommand::Execute() {
    estimator_->set_in_memory(in_memory_);

    auto start = std::chrono::high_resolution_clock::now();
    double estimation = estimator_->Estimate(dataset_directory_);
    auto end = std::chrono::high_resolution_clock::now();

    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory_ / "metadata.json");

    // Output the result.
    std::cout << std::format(
        "The result is as follows:\n"
        "    Time Consumed: {:.2f} ms\n"
        "    Estimated Chamfer distance: {}\n"
        "    Relative error: {:.2f}%\n",
        std::chrono::duration<double, std::milli>(end - start).count(), estimation,
        std::fabs(estimation - dataset_metadata.chamfer_distance) / dataset_metadata.chamfer_distance *
            100);  // NOLINT: readability-magic-numbers
}