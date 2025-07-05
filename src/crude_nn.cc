#include "crude_nn.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

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
    fs::path param_file_path = parent_directory_ / dataset_name_ / "index_params.bin";
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

    return *this;
}

auto CrudeNnBuilder::Build() const -> std::unique_ptr<CrudeNn> {
    return std::unique_ptr<CrudeNn>(new CrudeNn(dataset_name_, parent_directory_, num_points_, num_dimensions_,
                                                approximation_ratio_, bucket_width_, beta_, error_probability_,
                                                num_hash_tables_, collision_threshold_, page_size_, verbose_));
}

// ---------- CrudeNn Implementation ----------

CrudeNn::CrudeNn(std::string dataset_name, fs::path parent_directory, unsigned int num_points,
                 unsigned int num_dimensions, double approximation_ratio, double bucket_width, double beta,
                 double error_probability, unsigned int num_hash_tables, unsigned int collision_threshold,
                 unsigned int page_size, bool verbose)
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

auto CrudeNn::Execute() const -> void {}

};  // namespace qalsh_chamfer