#include "crude_nn.h"

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "b_plus_tree.h"
#include "utils.h"

namespace qalsh_chamfer {

// ---------- CrudeNnBuilder Implementation ----------

CrudeNnBuilder::CrudeNnBuilder()
    : num_points_(0),
      num_dimensions_(0),
      approximation_ratio_(0.0),
      bucket_width_(0.0),
      beta_(0.0),
      error_probability_(0.0),
      num_hash_tables_(0),
      collision_threshold_(0),
      page_size_(0),
      verbose_(false) {}

auto CrudeNnBuilder::set_dataset_name(const std::string& dataset_name) -> CrudeNnBuilder& {
    dataset_name_ = dataset_name;
    return *this;
}

auto CrudeNnBuilder::set_parent_directory(const fs::path& parent_directory) -> CrudeNnBuilder& {
    parent_directory_ = parent_directory;
    return *this;
}

auto CrudeNnBuilder::set_num_points(unsigned int num_points) -> CrudeNnBuilder& {
    num_points_ = num_points;
    return *this;
}

auto CrudeNnBuilder::set_num_dimensions(unsigned int num_dimensions) -> CrudeNnBuilder& {
    num_dimensions_ = num_dimensions;
    return *this;
}

auto CrudeNnBuilder::set_verbose(bool debug) -> CrudeNnBuilder& {
    verbose_ = debug;
    return *this;
}

auto CrudeNnBuilder::ReadParamFromBinaryFile() -> CrudeNnBuilder& {
    // Read the parameters from the file
    fs::path param_file_path = parent_directory_ / dataset_name_ / "index" / "index_params.bin";
    std::ifstream ifs(param_file_path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open parameter file, you may need to run the indexer first.");
    }

    ifs.read(reinterpret_cast<char*>(&approximation_ratio_), sizeof(approximation_ratio_));
    ifs.read(reinterpret_cast<char*>(&bucket_width_), sizeof(bucket_width_));
    ifs.read(reinterpret_cast<char*>(&beta_), sizeof(beta_));
    ifs.read(reinterpret_cast<char*>(&error_probability_), sizeof(error_probability_));
    ifs.read(reinterpret_cast<char*>(&num_hash_tables_), sizeof(num_hash_tables_));
    ifs.read(reinterpret_cast<char*>(&collision_threshold_), sizeof(collision_threshold_));
    ifs.read(reinterpret_cast<char*>(&page_size_), sizeof(page_size_));

    dot_vectors_.resize(num_hash_tables_);
    for (unsigned int i = 0; i < num_hash_tables_; i++) {
        dot_vectors_[i].resize(num_dimensions_);
        ifs.read(reinterpret_cast<char*>(dot_vectors_[i].data()),
                 static_cast<std::streamsize>(sizeof(double) * num_dimensions_));
    }

    return *this;
}

auto CrudeNnBuilder::Build() const -> std::unique_ptr<CrudeNn> {
    return std::unique_ptr<CrudeNn>(new CrudeNn(
        dataset_name_, parent_directory_, num_points_, num_dimensions_, approximation_ratio_, bucket_width_, beta_,
        error_probability_, num_hash_tables_, collision_threshold_, page_size_, dot_vectors_, verbose_));
}

// ---------- CrudeNn Implementation ----------

CrudeNn::CrudeNn(std::string dataset_name, fs::path parent_directory, unsigned int num_points,
                 unsigned int num_dimensions, double approximation_ratio, double bucket_width, double beta,
                 double error_probability, unsigned int num_hash_tables, unsigned int collision_threshold,
                 unsigned int page_size, std::vector<std::vector<double>> dot_vectors, bool verbose)
    : dataset_name_(std::move(dataset_name)),
      parent_directory_(std::move(parent_directory)),
      num_points_(num_points),
      num_dimensions_(num_dimensions),
      approximation_ratio_(approximation_ratio),
      bucket_width_(bucket_width),
      beta_(beta),
      error_probability_(error_probability),
      num_hash_tables_(num_hash_tables),
      collision_threshold_(collision_threshold),
      page_size_(page_size),
      dot_vectors_(std::move(dot_vectors)),
      verbose_(verbose) {}

auto CrudeNn::PrintConfiguration() const -> void {
    std::cout << std::format("---------- Crude Nearest Neighbor Configuration ----------\n");
    std::cout << std::format("Dataset Name: {}\n", dataset_name_);
    std::cout << std::format("Parent Directory: {}\n", parent_directory_.string());
    std::cout << std::format("Number of Points: {}\n", num_points_);
    std::cout << std::format("Number of Dimensions: {}\n", num_dimensions_);
    std::cout << std::format("Approximation Ratio: {}\n", approximation_ratio_);
    std::cout << std::format("Bucket Width: {}\n", bucket_width_);
    std::cout << std::format("Beta: {}\n", beta_);
    std::cout << std::format("Error Probability: {}\n", error_probability_);
    std::cout << std::format("Number of Hash Tables: {}\n", num_hash_tables_);
    std::cout << std::format("Collision Threshold: {}\n", collision_threshold_);
    std::cout << std::format("Page Size: {}\n", page_size_);
    std::cout << std::format("-----------------------------------------------------\n");
}

auto CrudeNn::Execute() const -> void {
    // Read the sets from the files
    fs::path setA_file_path = parent_directory_ / dataset_name_ / "A.bin";
    std::vector<std::vector<double>> setA =
        Utils::ReadSetFromFile(setA_file_path, num_points_, num_dimensions_, "A", verbose_);

    fs::path setB_file_path = parent_directory_ / dataset_name_ / "B.bin";
    std::vector<std::vector<double>> setB =
        Utils::ReadSetFromFile(setB_file_path, num_points_, num_dimensions_, "B", verbose_);

    // Generate the D arrays for both sets
    std::vector<double> d_array_A = GenerateDArrayForSet(setA, setB, "A", "B");
    std::vector<double> d_array_B = GenerateDArrayForSet(setB, setA, "B", "A");

    // Write the D arrays to files
    fs::path d_array_A_file_path = parent_directory_ / dataset_name_ / "index" / "D_A.bin";
    Utils::WriteArrayToFile(d_array_A_file_path, d_array_A);

    fs::path d_array_B_file_path = parent_directory_ / dataset_name_ / "index" / "D_B.bin";
    Utils::WriteArrayToFile(d_array_B_file_path, d_array_B);
}

auto CrudeNn::GenerateDArrayForSet(const std::vector<std::vector<double>>& set_from,
                                   const std::vector<std::vector<double>>& set_to, const std::string& set_from_name,
                                   const std::string& set_to_name) const -> std::vector<double> {
    std::vector<double> d_array(num_points_);

    for (unsigned int i = 0; i < num_points_; i++) {
        if (verbose_) {
            std::cout << std::format("Processing point {}/{} in set {}...\r", i + 1, num_points_, set_from_name)
                      << std::flush;
        }

        d_array[i] = CAnnSearch(set_from[i], set_to, set_to_name).first;
    }

    if (verbose_) {
        std::cout << "\n";
    }

    return d_array;
}

auto CrudeNn::CAnnSearch(const std::vector<double>& query, const std::vector<std::vector<double>>& dataset,
                         const std::string& set_name) const -> Candidate {
    std::priority_queue<Candidate, std::vector<Candidate>> candidates;
    std::vector<bool> visited(num_points_, false);
    std::unordered_map<unsigned int, unsigned int> collision_count;
    std::vector<BPlusTreeSearcher> b_plus_tree_searchers;
    std::vector<double> keys(num_hash_tables_);
    double search_radius = 1.0;

    // Initialize the keys
    for (unsigned int i = 0; i < num_hash_tables_; i++) {
        keys[i] = Utils::DotProduct(query, dot_vectors_[i]);
    }

    // Initialize B+ trees
    b_plus_tree_searchers.reserve(num_hash_tables_);
    for (unsigned int j = 0; j < num_hash_tables_; j++) {
        const fs::path index_file_path =
            parent_directory_ / dataset_name_ / "index" / std::format("{}_idx_{}.bin", set_name, j);
        b_plus_tree_searchers.emplace_back(index_file_path, page_size_, keys[j]);
    }

    // c-ANN search
    while (candidates.size() < static_cast<size_t>(std::ceil(beta_ * num_points_))) {
        for (unsigned int j = 0; j < num_hash_tables_; j++) {
            std::vector<unsigned int> point_ids =
                b_plus_tree_searchers[j].IncrementalSearch(bucket_width_ * search_radius / 2.0);

            for (auto point_id : point_ids) {
                if (visited.at(point_id)) {
                    continue;
                }

                collision_count[point_id]++;
                if (collision_count.at(point_id) >= collision_threshold_) {
                    const std::vector<double>& point = dataset[point_id];
                    candidates.emplace(Utils::CalculateManhattan(point, query), point_id);

                    visited.at(point_id) = true;
                }
            }
        }

        if (!candidates.empty() && candidates.top().first <= approximation_ratio_ * search_radius) {
            break;
        }

        search_radius *= approximation_ratio_;
    }

    if (!candidates.empty()) {
        return candidates.top();
    }

    // Defense programming
    return {std::numeric_limits<double>::max(), std::numeric_limits<unsigned int>::max()};
}

};  // namespace qalsh_chamfer