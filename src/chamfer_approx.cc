#include "chamfer_approx.h"

#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <string>

namespace qalsh_chamfer {

// ---------- ChamferApproxBuilder Implementation ----------

ChamferApproxBuilder::ChamferApproxBuilder() : num_points_(0), num_dimensions_(0), verbose_(false) {}

auto ChamferApproxBuilder::set_dataset_name(const std::string& dataset_name) -> ChamferApproxBuilder& {
    this->dataset_name_ = dataset_name;
    return *this;
}

auto ChamferApproxBuilder::set_parent_directory(const fs::path& parent_directory) -> ChamferApproxBuilder& {
    this->parent_directory_ = parent_directory;
    return *this;
}

auto ChamferApproxBuilder::set_num_points(unsigned int num_points) -> ChamferApproxBuilder& {
    this->num_points_ = num_points;
    return *this;
}

auto ChamferApproxBuilder::set_num_dimensions(unsigned int num_dimensions) -> ChamferApproxBuilder& {
    this->num_dimensions_ = num_dimensions;
    return *this;
}

auto ChamferApproxBuilder::set_verbose(bool verbose) -> ChamferApproxBuilder& {
    this->verbose_ = verbose;
    return *this;
}

auto ChamferApproxBuilder::Build() const -> std::unique_ptr<ChamferApprox> {
    return std::unique_ptr<ChamferApprox>(
        new ChamferApprox(dataset_name_, parent_directory_, num_points_, num_dimensions_, verbose_));
}

// ---------- ChamferApprox Implementation ----------

ChamferApprox::ChamferApprox(std::string dataset_name, fs::path parent_directory, unsigned int num_points,
                             unsigned int num_dimensions, bool verbose)
    : dataset_name_(std::move(dataset_name)),
      parent_directory_(std::move(parent_directory)),
      num_points_(num_points),
      num_dimensions_(num_dimensions),
      verbose_(verbose) {}

auto ChamferApprox::PrintConfiguration() const -> void {
    std::cout << std::format("---------- Chamfer Approximation Configuration ----------\n");
    std::cout << std::format("Dataset Name: {}\n", dataset_name_);
    std::cout << std::format("Parent Directory: {}\n", parent_directory_.string());
    std::cout << std::format("Number of Points: {}\n", num_points_);
    std::cout << std::format("Number of Dimensions: {}\n", num_dimensions_);
    std::cout << std::format("-----------------------------------------------------\n");
}

auto ChamferApprox::Execute() const -> void {}

};  // namespace qalsh_chamfer