#include "crude_nn.h"

#include <filesystem>
#include <format>
#include <iostream>
#include <memory>

namespace qalsh_chamfer {

// ---------- CrudeNnBuilder Implementation ----------

CrudeNnBuilder::CrudeNnBuilder() : num_points_(0), num_dimensions_(0), verbose_(false) {}

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

auto CrudeNnBuilder::Build() const -> std::unique_ptr<CrudeNn> {
    return std::unique_ptr<CrudeNn>(
        new CrudeNn(dataset_name_, parent_directory_, num_points_, num_dimensions_, verbose_));
}

// ---------- CrudeNn Implementation ----------

CrudeNn::CrudeNn(std::string dataset_name, fs::path parent_directory, unsigned int num_points,
                 unsigned int num_dimensions, bool verbose)
    : dataset_name_(std::move(dataset_name)),
      parent_directory_(std::move(parent_directory)),
      num_points_(num_points),
      num_dimensions_(num_dimensions),
      verbose_(verbose) {}

auto CrudeNn::PrintConfiguration() const -> void {
    std::cout << std::format("---------- Dataset Generator Configuration ----------\n");
    std::cout << std::format("Dataset Name: {}\n", dataset_name_);
    std::cout << std::format("Parent Directory: {}\n", parent_directory_.string());
    std::cout << std::format("Number of Points: {}\n", num_points_);
    std::cout << std::format("Number of Dimensions: {}\n", num_dimensions_);
    std::cout << std::format("-----------------------------------------------------\n");
}

auto CrudeNn::Execute() const -> void {
    // Execute the nearest neighbor search
}

};  // namespace qalsh_chamfer