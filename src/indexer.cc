#include "indexer.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <utility>
#include <vector>

#include "b_plus_tree.h"
#include "constants.h"
#include "pager.h"
#include "utils.h"

namespace qalsh_chamfer {

// ------ IndexerBuilder Implementation ------

IndexerBuilder::IndexerBuilder()
    : num_points_(0),
      num_dimensions_(0),
      approximation_ratio_(0.0),
      bucket_width_(0.0),
      beta_(0.0),
      error_probability_(0.0),
      num_hash_tables_(0),
      page_size_(0),
      verbose_(false) {}

auto IndexerBuilder::set_dataset_name(const std::string& dataset_name) -> IndexerBuilder& {
    dataset_name_ = dataset_name;
    return *this;
}

auto IndexerBuilder::set_parent_directory(const fs::path& parent_directory) -> IndexerBuilder& {
    parent_directory_ = parent_directory;
    return *this;
}

auto IndexerBuilder::set_num_points(unsigned int num_points) -> IndexerBuilder& {
    num_points_ = num_points;
    return *this;
}

auto IndexerBuilder::set_num_dimensions(unsigned int num_dimensions) -> IndexerBuilder& {
    num_dimensions_ = num_dimensions;
    return *this;
}

auto IndexerBuilder::set_approximation_ratio(double approximation_ratio) -> IndexerBuilder& {
    approximation_ratio_ = approximation_ratio;
    return *this;
}

auto IndexerBuilder::set_bucket_width(double bucket_width) -> IndexerBuilder& {
    if (bucket_width <= kEpsilon) {
        bucket_width_ = 2.0 * std::sqrt(approximation_ratio_);
    } else {
        bucket_width_ = bucket_width;
    }
    return *this;
}

auto IndexerBuilder::set_beta(double beta) -> IndexerBuilder& {
    if (beta <= kEpsilon) {
        beta_ = 100.0 / static_cast<double>(num_points_);
    } else {
        beta_ = beta;
    }
    return *this;
}

auto IndexerBuilder::set_error_probability(double error_probability) -> IndexerBuilder& {
    error_probability_ = error_probability;
    return *this;
}

auto IndexerBuilder::set_num_hash_tables(unsigned int num_hash_tables) -> IndexerBuilder& {
    if (num_hash_tables == 0) {
        // Calculate the numerator
        double term1 = std::sqrt(std::log(2.0 / beta_));
        double term2 = std::sqrt(std::log(1.0 / error_probability_));
        double numerator = std::pow(term1 + term2, 2.0);

        // Calculate the denominator
        double p1 = 2.0 / std::numbers::pi_v<double> * atan(bucket_width_ / 2.0);
        double p2 = 2.0 / std::numbers::pi_v<double> * atan(bucket_width_ / (2.0 * approximation_ratio_));
        double denominator = 2.0 * std::pow(p1 - p2, 2.0);

        // Calculate the number of hash tables
        num_hash_tables_ = static_cast<unsigned int>(std::ceil(numerator / denominator));
    } else {
        num_hash_tables_ = num_hash_tables;
    }
    return *this;
}

auto IndexerBuilder::set_page_size(unsigned int page_size) -> IndexerBuilder& {
    if (page_size == 0) {
        page_size_ = static_cast<unsigned int>(ceil((static_cast<double>(BPlusTree::GetHeaderBasicInfoSize()) +
                                                     static_cast<double>(num_dimensions_) * sizeof(double)) /
                                                    kBasePageSize)) *
                     kBasePageSize;
    } else {
        page_size_ = page_size;
    }
    return *this;
}

auto IndexerBuilder::set_verbose(bool verbose) -> IndexerBuilder& {
    verbose_ = verbose;
    return *this;
}

auto IndexerBuilder::Build() const -> std::unique_ptr<Indexer> {
    return std::unique_ptr<Indexer>(new Indexer(dataset_name_, parent_directory_, num_points_, num_dimensions_,
                                                approximation_ratio_, bucket_width_, beta_, error_probability_,
                                                num_hash_tables_, page_size_, verbose_));
}

// ------ Indexer Implementation ------

Indexer::Indexer(std::string dataset_name, fs::path parent_directory, unsigned int num_points,
                 unsigned int num_dimensions, double approximation_ratio, double bucket_width, double beta,
                 double error_probability, unsigned int num_hash_tables, unsigned int page_size, bool verbose)
    : dataset_name_(std::move(dataset_name)),
      parent_directory_(std::move(parent_directory)),
      num_points_(num_points),
      num_dimensions_(num_dimensions),
      approximation_ratio_(approximation_ratio),
      bucket_width_(bucket_width),
      beta_(beta),
      error_probability_(error_probability),
      num_hash_tables_(num_hash_tables),
      page_size_(page_size),
      verbose_(verbose) {}

auto Indexer::PrintConfiguration() const -> void {
    std::cout << "------------- Indexer Configuration -------------\n";
    std::cout << std::format("Dataset Name: {}\n", dataset_name_);
    std::cout << std::format("Parent Directory: {}\n", parent_directory_.string());
    std::cout << std::format("Number of Points: {}\n", num_points_);
    std::cout << std::format("Number of Dimensions: {}\n", num_dimensions_);
    std::cout << std::format("Approximation Ratio: {}\n", approximation_ratio_);
    std::cout << std::format("Bucket Width: {}\n", bucket_width_);
    std::cout << std::format("Beta: {}\n", beta_);
    std::cout << std::format("Error Probability: {}\n", error_probability_);
    std::cout << std::format("Number of Hash Tables: {}\n", num_hash_tables_);
    std::cout << std::format("Page Size: {}\n", page_size_);
    std::cout << "-------------------------------------------------\n";
}

auto Indexer::Execute() const -> void {
    std::mt19937 rng(std::random_device{}());
    std::cauchy_distribution<double> standard_cauchy_dist(0.0, 1.0);
    fs::path index_directory = parent_directory_ / dataset_name_ / "index";

    // Create the index directory if it does not exist
    if (!fs::exists(index_directory)) {
        fs::create_directories(index_directory);
    }

    auto indexing_helper = [this, &standard_cauchy_dist, &rng](const std::string& set_name) -> void {
        // Read the set from the file
        fs::path set_file_path = parent_directory_ / dataset_name_ / std::format("{}_{}.bin", dataset_name_, set_name);
        std::vector<std::vector<double>> set =
            Utils::ReadSetFromFile(set_file_path, num_points_, num_dimensions_, set_name, verbose_);

        for (unsigned int i = 0; i < num_hash_tables_; i++) {
            if (verbose_) {
                std::cout << std::format("Indexing set {} ... ({}/{})\r", set_name, i + 1, num_hash_tables_)
                          << std::flush;
            }

            // Generate the dot vector
            std::vector<double> dot_vector(num_dimensions_);
            std::ranges::generate(dot_vector, [&]() { return standard_cauchy_dist(rng); });

            // Build the index
            fs::path index_file_path = parent_directory_ / dataset_name_ / "index" /
                                       std::format("{}_{}_idx_{}.bin", dataset_name_, set_name, i);

            BuildIndexForSet(dot_vector, set, index_file_path);
        }

        std::cout << "\n";
    };

    indexing_helper("A");
    indexing_helper("B");
}

auto Indexer::BuildIndexForSet(std::vector<double>& dot_vector, std::vector<std::vector<double>>& set,
                               const fs::path& index_file_path) const -> void {
    // Calculate the dot products and sort them
    std::vector<std::pair<double, unsigned int>> dot_products_with_id;
    dot_products_with_id.reserve(num_points_);

    for (unsigned int j = 0; j < num_points_; j++) {
        dot_products_with_id.emplace_back(Utils::DotProduct(set[j], dot_vector), j);
    }
    std::ranges::sort(dot_products_with_id);

    // Bulk load the B+ tree with sorted dot products
    Pager pager(index_file_path, page_size_);
    BPlusTree b_plus_tree(std::move(pager), dot_vector);

    b_plus_tree.BulkLoad(dot_products_with_id);
}

}  // namespace qalsh_chamfer