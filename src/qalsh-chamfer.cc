#include "qalsh-chamfer.h"

#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

namespace qalsh_chamfer {

// ---------- QalshChamferBuilder Implementation ----------

QalshChamferBuilder::QalshChamferBuilder() = default;

auto QalshChamferBuilder::set_dataset_name(const std::string& dataset_name) -> QalshChamferBuilder& {
    dataset_name_ = dataset_name;
    return *this;
}

auto QalshChamferBuilder::set_parent_directory(const std::string& parent_directory) -> QalshChamferBuilder& {
    parent_directory_ = parent_directory;
    return *this;
}

auto QalshChamferBuilder::set_num_points(unsigned int num_points) -> QalshChamferBuilder& {
    num_points_ = num_points;
    return *this;
}

auto QalshChamferBuilder::set_num_dimensions(unsigned int num_dimensions) -> QalshChamferBuilder& {
    num_dimensions_ = num_dimensions;
    return *this;
}

auto QalshChamferBuilder::set_build_index(bool build_index) -> QalshChamferBuilder& {
    build_index_ = build_index;
    return *this;
}

auto QalshChamferBuilder::set_verbose(bool verbose) -> QalshChamferBuilder& {
    verbose_ = verbose;
    return *this;
}

auto QalshChamferBuilder::Build() const -> std::unique_ptr<QalshChamfer> {
    return std::unique_ptr<QalshChamfer>(
        new QalshChamfer(dataset_name_, parent_directory_, num_points_, num_dimensions_, build_index_, verbose_));
}

// ---------- QalshChamfer Implementation ----------

QalshChamfer::QalshChamfer(std::string dataset_name, std::string parent_directory, unsigned int num_points,
                           unsigned int num_dimensions, bool build_index, bool verbose)
    : dataset_name_(std::move(dataset_name)),
      parent_directory_(std::move(parent_directory)),
      num_points_(num_points),
      num_dimensions_(num_dimensions),
      build_index_(build_index),
      verbose_(verbose) {}

auto QalshChamfer::PrintConfiguration() const -> void {
    std::cout << std::format("---------- Dataset Generator Configuration ----------\n");
    std::cout << std::format("Dataset Name: {}\n", dataset_name_);
    std::cout << std::format("Parent Directory: {}\n", parent_directory_);
    std::cout << std::format("Number of Points: {}\n", num_points_);
    std::cout << std::format("Number of Dimensions: {}\n", num_dimensions_);
    std::cout << std::format("Build Index: {}\n", build_index_ ? "true" : "false");
    std::cout << std::format("-----------------------------------------------------\n");
}

auto QalshChamfer::Execute() const -> void {}

}  // namespace qalsh_chamfer