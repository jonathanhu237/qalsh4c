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

namespace qalsh_chamfer {

// ---------- DatasetGeneratorBuilder Implementation ----------

auto DatasetGeneratorBuilder::set_dataset_name(const std::string& dataset_name) -> DatasetGeneratorBuilder& {
    dataset_name_ = dataset_name;
    return *this;
}

auto DatasetGeneratorBuilder::set_parent_directory(const std::string& parent_directory) -> DatasetGeneratorBuilder& {
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

DatasetGenerator::DatasetGenerator(std::string dataset_name, std::string parent_directory, unsigned int num_points,
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
    std::cout << std::format("Parent Directory: {}\n", parent_directory_);
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

    std::string dataset_directory = std::format("{}/{}", parent_directory_, dataset_name_);
    if (!std::filesystem::exists(dataset_directory)) {
        std::filesystem::create_directories(dataset_directory);
    }

    WriteSetToFile(dataset_directory, "A", set_a);
    WriteSetToFile(dataset_directory, "B", set_b);

    // Calculate the chamfer distance from set A to set B and vice versa

    Eigen::MatrixXd eigen_set_a = ToEigenMatrix(set_a);
    Eigen::MatrixXd eigen_set_b = ToEigenMatrix(set_b);

    double chamfer_a_to_b = CalculateChamfer(eigen_set_a, eigen_set_b, "A", "B");
    double chamfer_b_to_a = CalculateChamfer(eigen_set_b, eigen_set_a, "B", "A");
    double total_chamfer_distance = chamfer_a_to_b + chamfer_b_to_a;

    std::string gt_file_path = std::format("{}/{}_gt.bin", dataset_directory, dataset_name_);
    std::ofstream gt_ofs(gt_file_path, std::ios::binary | std::ios::trunc);

    if (!gt_ofs.is_open()) {
        throw std::runtime_error(std::format("Error: Could not open file for writing: {}", gt_file_path));
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

auto DatasetGenerator::WriteSetToFile(const std::string& dataset_directory, const std::string& set_name,
                                      const std::vector<std::vector<double>>& set) const -> void {
    std::string file_path = std::format("{}/{}_{}.bin", dataset_directory, dataset_name_, set_name);
    std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);

    if (!ofs.is_open()) {
        throw std::runtime_error(std::format("Error: Could not open file for writing: {}", file_path));
    }

    for (const auto& point : set) {
        if (verbose_) {
            std::cout << std::format("Writing set {} to file ... ({}/{})\r", set_name, &point - set.data() + 1,
                                     set.size())
                      << std::flush;
        }

        ofs.write(reinterpret_cast<const char*>(point.data()),
                  static_cast<std::streamsize>(sizeof(double) * num_dimensions_));
    }
}

auto DatasetGenerator::ToEigenMatrix(const std::vector<std::vector<double>>& set) const
    -> Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix(num_points_, num_dimensions_);
    for (unsigned int i = 0; i < num_points_; ++i) {
        matrix.row(i) = Eigen::Map<const Eigen::VectorXd>(set[i].data(), num_dimensions_);
    }
    return matrix;
}

auto DatasetGenerator::CalculateChamfer(const Eigen::MatrixXd& from_matrix, const Eigen::MatrixXd& to_matrix,
                                        const std::string& from_name, const std::string& to_name) const -> double {
    double chamfer_distance = 0.0;
    for (unsigned int i = 0; i < num_points_; ++i) {
        if (verbose_) {
            std::cout << std::format("Calculating Chamfer distance ({} to {})... ({}/{})\r", from_name, to_name, i + 1,
                                     num_points_)
                      << std::flush;
        }
        chamfer_distance += (to_matrix.rowwise() - from_matrix.row(i)).rowwise().lpNorm<1>().minCoeff();
    }
    if (verbose_) {
        std::cout << "\n";
    }
    return chamfer_distance;
}

};  // namespace qalsh_chamfer