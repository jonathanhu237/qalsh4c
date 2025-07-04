#include "dataset_generator.h"

#include <Eigen/Dense>
#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "utils.h"

namespace qalsh_chamfer {

namespace fs = std::filesystem;

// ---------- DatasetGeneratorBuilder Implementation ----------

DatasetGeneratorBuilder::DatasetGeneratorBuilder()
    : num_points_(0), num_dimensions_(0), left_boundary_(0), right_boundary_(0), verbose_(false) {}

auto DatasetGeneratorBuilder::set_dataset_name(const std::string& dataset_name) -> DatasetGeneratorBuilder& {
    dataset_name_ = dataset_name;
    return *this;
}

auto DatasetGeneratorBuilder::set_parent_directory(const fs::path& parent_directory) -> DatasetGeneratorBuilder& {
    parent_directory_ = parent_directory;
    return *this;
}

auto DatasetGeneratorBuilder::set_num_points_(unsigned int num_points) -> DatasetGeneratorBuilder& {
    num_points_ = num_points;
    return *this;
}

auto DatasetGeneratorBuilder::set_num_dimensions(unsigned int num_dimensions) -> DatasetGeneratorBuilder& {
    num_dimensions_ = num_dimensions;
    return *this;
}

auto DatasetGeneratorBuilder::set_left_boundary(int left_boundary) -> DatasetGeneratorBuilder& {
    left_boundary_ = left_boundary;
    return *this;
}

auto DatasetGeneratorBuilder::set_right_boundary(int right_boundary) -> DatasetGeneratorBuilder& {
    right_boundary_ = right_boundary;
    return *this;
}

auto DatasetGeneratorBuilder::set_verbose(bool debug) -> DatasetGeneratorBuilder& {
    verbose_ = debug;
    return *this;
}

auto DatasetGeneratorBuilder::Build() const -> std::unique_ptr<DatasetGenerator> {
    return std::unique_ptr<DatasetGenerator>(new DatasetGenerator(
        dataset_name_, parent_directory_, num_points_, num_dimensions_, left_boundary_, right_boundary_, verbose_));
}

// ---------- DatasetGenerator Implementation ----------

DatasetGenerator::DatasetGenerator(std::string dataset_name, fs::path parent_directory, unsigned int num_points,
                                   unsigned int num_dimensions, int left_boundary, int right_boundary, bool debug)
    : dataset_name_(std::move(dataset_name)),
      parent_directory_(std::move(parent_directory)),
      num_points_(num_points),
      num_dimensions_(num_dimensions),
      left_boundary_(left_boundary),
      right_boundary_(right_boundary),
      verbose_(debug) {}

auto DatasetGenerator::PrintConfiguration() const -> void {
    std::cout << std::format("---------- Dataset Generator Configuration ----------\n");
    std::cout << std::format("Dataset Name: {}\n", dataset_name_);
    std::cout << std::format("Parent Directory: {}\n", parent_directory_.string());
    std::cout << std::format("Number of Points: {}\n", num_points_);
    std::cout << std::format("Number of Dimensions: {}\n", num_dimensions_);
    std::cout << std::format("Left Boundary: {}\n", left_boundary_);
    std::cout << std::format("Right Boundary: {}\n", right_boundary_);
    std::cout << std::format("-----------------------------------------------------\n");
}

auto DatasetGenerator::Execute() const -> void {
    std::uniform_real_distribution<double> dist(left_boundary_, right_boundary_);
    std::mt19937 gen(std::random_device{}());

    std::vector<std::vector<double>> set_a = GenerateSet(dist, gen, "A");
    std::vector<std::vector<double>> set_b = GenerateSet(dist, gen, "B");

    fs::path dataset_directory = parent_directory_ / dataset_name_;
    if (!fs::exists(dataset_directory)) {
        fs::create_directories(dataset_directory);
    }

    Utils::WriteSetToFile(dataset_directory / std::format("{}_A.bin", dataset_name_), set_a, "A", verbose_);
    Utils::WriteSetToFile(dataset_directory / std::format("{}_B.bin", dataset_name_), set_b, "B", verbose_);

    // Calculate the chamfer distance from set A to set B and vice versa

    double chamfer_a_to_b = Utils::CalculateChamfer(set_a, set_b, "A", "B", verbose_);
    double chamfer_b_to_a = Utils::CalculateChamfer(set_b, set_a, "B", "A", verbose_);
    double total_chamfer_distance = chamfer_a_to_b + chamfer_b_to_a;

    fs::path gt_file_path = dataset_directory / std::format("{}_gt.bin", dataset_name_);
    std::ofstream gt_ofs(gt_file_path, std::ios::binary | std::ios::trunc);

    if (!gt_ofs.is_open()) {
        throw std::runtime_error(std::format("Error: Could not open file for writing: {}", gt_file_path.string()));
    }

    gt_ofs.write(reinterpret_cast<const char*>(&total_chamfer_distance), sizeof(double));

    if (verbose_) {
        std::cout << std::format("Chamfer distance (A->B): {}\n", chamfer_a_to_b);
        std::cout << std::format("Chamfer distance (B->A): {}\n", chamfer_b_to_a);
        std::cout << std::format("Total Chamfer distance: {}\n", total_chamfer_distance);
    }
}

auto DatasetGenerator::GenerateSet(std::uniform_real_distribution<double>& dist, std::mt19937& gen,
                                   const std::string& set_name) const -> std::vector<std::vector<double>> {
    std::vector<std::vector<double>> set(num_points_, std::vector<double>(num_dimensions_));

    for (unsigned int i = 0; i < num_points_; i++) {
        if (verbose_) {
            std::cout << std::format("Generating points for set {}... ({}/{})\r", set_name, i + 1, num_points_)
                      << std::flush;
        }

        std::ranges::generate(set[i], [&]() { return dist(gen); });
    }

    if (verbose_) {
        std::cout << "\n";
    }
    return set;
}

};  // namespace qalsh_chamfer