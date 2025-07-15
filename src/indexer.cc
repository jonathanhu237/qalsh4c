#include "indexer.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <format>
#include <ios>
#include <memory>
#include <vector>

#include "b_plus_tree.h"
#include "point_set.h"
#include "types.h"
#include "utils.h"

// ---------------------------------------------
// QalshIndexer Implementation
// ---------------------------------------------

QalshIndexer::QalshIndexer(std::filesystem::path dataset_directory, QalshConfiguration qalsh_config, bool in_memory)
    : dataset_directory_(std::move(dataset_directory)),
      qalsh_config_(qalsh_config),
      in_memory_(in_memory),
      gen_(std::random_device{}()) {
    // Read dataset metadata
    dataset_metadata_.Load(dataset_directory_ / "metadata.toml");

    // Regularize the QALSH configuration
    qalsh_config_.Regularize(dataset_metadata_.base_num_points);

    // Initialize the base reader
    base_reader_ =
        PointSetReaderFactory::Create(in_memory_, dataset_metadata_.data_type, dataset_directory_ / "base.bin",
                                      dataset_metadata_.base_num_points, dataset_metadata_.num_dimensions);
}

void QalshIndexer::BuildIndex() {
    PrintConfiguration();

    // Create the index directory if it does not exist
    std::filesystem::path index_directory = dataset_directory_ / "qalsh_index";
    if (!std::filesystem::exists(index_directory)) {
        spdlog::info("Creating index directory: {}", index_directory.string());
        std::filesystem::create_directories(index_directory);
    }

    // Generate the dot vectors
    spdlog::info("Generating dot vectors for {} hash tables...", qalsh_config_.num_hash_tables);
    std::cauchy_distribution<double> standard_cauchy_dist(0.0, 1.0);
    std::vector<std::vector<double>> dot_vectors(qalsh_config_.num_hash_tables,
                                                 std::vector<double>(dataset_metadata_.num_dimensions));
    for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
        std::ranges::generate(dot_vectors[i], [&]() { return standard_cauchy_dist(gen_); });
    }

    // Index the base set
    spdlog::info("Indexing base set...");
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
        spdlog::debug("Indexing hash table {}/{}", i + 1, qalsh_config_.num_hash_tables);
        std::vector<std::pair<double, unsigned int>> dot_products_with_id(dataset_metadata_.base_num_points);
        for (unsigned int j = 0; j < dataset_metadata_.base_num_points; j++) {
            // Calculate the dot product for each point in the base set
            PointVariant base_point = base_reader_->GetPoint(j);
            double dot_product = 0.0;
            std::visit(
                [&](const auto& concrete_point) { dot_product = Utils::DotProduct(concrete_point, dot_vectors[i]); },
                base_point);
            dot_products_with_id[j] = {dot_product, j};
        }
        // Sort the dot products
        std::ranges::sort(dot_products_with_id);

        // Bulk load the B+ tree with sorted dot products
        BPlusTreeBulkLoader bulk_loader(index_directory / std::format("base_idx_{}.bin", i), qalsh_config_.page_size);
        bulk_loader.Build(dot_products_with_id);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    spdlog::info("Indexing completed in {:.2f} ms.", elapsed.count());

    // Save the dot product vectors
    spdlog::info("Saving dot product vectors...");
    std::ofstream ofs(index_directory / "dot_vectors.bin", std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error(
            std::format("Failed to open file for writing: {}", (index_directory / "dot_vectors.bin").string()));
    }
    for (const auto& vec : dot_vectors) {
        ofs.write(reinterpret_cast<const char*>(vec.data()), static_cast<std::streamsize>(vec.size() * sizeof(double)));
    }

    // Save the QALSH configuration
    spdlog::info("Saving QALSH configuration...");
    qalsh_config_.Save(index_directory / "config.toml");
}

void QalshIndexer::PrintConfiguration() const {
    spdlog::debug(R"(The configuration is as follows:
---------- QALSH Indexer Configuration ----------
Dataset Directory: {}
Number of points in base set: {}
Number of points in query set: {}
Number of Dimensions: {}
Data Type: {}
Approximation Ratio: {}
Bucket Width: {}
Beta: {}
Error Probability: {}
Number of Hash Tables: {}
Collision Threshold: {}
Page Size: {}
In Memory: {}
-----------------------------------------------------)",
                  dataset_directory_.string(), dataset_metadata_.base_num_points, dataset_metadata_.query_num_points,
                  dataset_metadata_.num_dimensions, dataset_metadata_.data_type, qalsh_config_.approximation_ratio,
                  qalsh_config_.bucket_width, qalsh_config_.beta, qalsh_config_.error_probability,
                  qalsh_config_.num_hash_tables, qalsh_config_.collision_threshold, qalsh_config_.page_size,
                  in_memory_);
}
