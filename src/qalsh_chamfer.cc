#include "qalsh_chamfer.h"

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

auto QalshChamferBuilder::set_verbose(bool verbose) -> QalshChamferBuilder& {
    verbose_ = verbose;
    return *this;
}

auto QalshChamferBuilder::Build() const -> std::unique_ptr<QalshChamfer> {
    return std::unique_ptr<QalshChamfer>(
        new QalshChamfer(dataset_name_, parent_directory_, num_points_, num_dimensions_, verbose_));
}

// ---------- QalshChamfer Implementation ----------

QalshChamfer::QalshChamfer(std::string dataset_name, std::string parent_directory, unsigned int num_points,
                           unsigned int num_dimensions, bool verbose)
    : dataset_name_(std::move(dataset_name)),
      parent_directory_(std::move(parent_directory)),
      num_points_(num_points),
      num_dimensions_(num_dimensions),
      verbose_(verbose) {}

auto QalshChamfer::PrintConfiguration() const -> void {
    std::cout << std::format("------------ QALSH Chamfer Configuration ------------\n");
    std::cout << std::format("Dataset Name: {}\n", dataset_name_);
    std::cout << std::format("Parent Directory: {}\n", parent_directory_);
    std::cout << std::format("Number of Points: {}\n", num_points_);
    std::cout << std::format("Number of Dimensions: {}\n", num_dimensions_);
    std::cout << std::format("-----------------------------------------------------\n");
}

auto QalshChamfer::Execute() const -> void {
    // TODO: implement the execution logic
}

}  // namespace qalsh_chamfer