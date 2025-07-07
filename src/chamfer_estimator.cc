#include "chamfer_estimator.h"

#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.h"

namespace qalsh_chamfer {

// ---------- ChamferEstimatorBuilder Implementation ----------

ChamferEstimatorBuilder::ChamferEstimatorBuilder()
    : num_points_(0), num_dimensions_(0), num_samples_(0), verbose_(false) {}

auto ChamferEstimatorBuilder::set_dataset_name(const std::string& dataset_name) -> ChamferEstimatorBuilder& {
    dataset_name_ = dataset_name;
    return *this;
}

auto ChamferEstimatorBuilder::set_parent_directory(const fs::path& parent_directory) -> ChamferEstimatorBuilder& {
    parent_directory_ = parent_directory;
    return *this;
}

auto ChamferEstimatorBuilder::set_num_points(unsigned int num_points) -> ChamferEstimatorBuilder& {
    num_points_ = num_points;
    return *this;
}

auto ChamferEstimatorBuilder::set_num_dimensions(unsigned int num_dimensions) -> ChamferEstimatorBuilder& {
    num_dimensions_ = num_dimensions;
    return *this;
}

auto ChamferEstimatorBuilder::set_num_samples(unsigned int num_samples) -> ChamferEstimatorBuilder& {
    if (num_samples == 0) {
        // Default to log(N) samples if num_samples is not set
        num_samples_ = static_cast<unsigned int>(std::log(num_points_));
    } else {
        num_samples_ = num_samples;
    }

    return *this;
}

auto ChamferEstimatorBuilder::set_verbose(bool verbose) -> ChamferEstimatorBuilder& {
    verbose_ = verbose;
    return *this;
}

auto ChamferEstimatorBuilder::Build() const -> std::unique_ptr<ChamferEstimator> {
    return std::unique_ptr<ChamferEstimator>(
        new ChamferEstimator(dataset_name_, parent_directory_, num_points_, num_dimensions_, num_samples_, verbose_));
}

// ---------- ChamferEstimator Implementation ----------

ChamferEstimator::ChamferEstimator(std::string dataset_name, fs::path parent_directory, unsigned int num_points,
                                   unsigned int num_dimensions, unsigned int num_samples, bool verbose)
    : dataset_name_(std::move(dataset_name)),
      parent_directory_(std::move(parent_directory)),
      num_points_(num_points),
      num_dimensions_(num_dimensions),
      num_samples_(num_samples),
      verbose_(verbose) {}

auto ChamferEstimator::PrintConfiguration() const -> void {
    std::cout << std::format("---------- Chamfer Approximation Configuration ----------\n");
    std::cout << std::format("Dataset Name: {}\n", dataset_name_);
    std::cout << std::format("Parent Directory: {}\n", parent_directory_.string());
    std::cout << std::format("Number of Points: {}\n", num_points_);
    std::cout << std::format("Number of Dimensions: {}\n", num_dimensions_);
    std::cout << std::format("Number of Samples: {}\n", num_samples_);
    std::cout << std::format("-----------------------------------------------------\n");
}

auto ChamferEstimator::Execute() const -> void {
    // Read the sets and D arrays from the files
    fs::path set_A_file_path = parent_directory_ / dataset_name_ / "A.bin";
    std::vector<std::vector<double>> set_A =
        Utils::ReadSetFromFile(set_A_file_path, num_points_, num_dimensions_, "A", verbose_);
    fs::path D_A_file_path = parent_directory_ / dataset_name_ / "index" / "D_A.bin";
    std::vector<double> D_A = Utils::ReadArrayFromFile(D_A_file_path, num_points_);

    fs::path set_B_file_path = parent_directory_ / dataset_name_ / "B.bin";
    std::vector<std::vector<double>> set_B =
        Utils::ReadSetFromFile(set_B_file_path, num_points_, num_dimensions_, "B", verbose_);
    fs::path D_B_file_path = parent_directory_ / dataset_name_ / "index" / "D_B.bin";
    std::vector<double> D_B = Utils::ReadArrayFromFile(D_B_file_path, num_points_);

    // Approximate the Chamfer distance
    double chamfer_A_to_B = ApproximateChamferDistance(set_A, set_B, D_A, "A", "B");
    double chamfer_B_to_A = ApproximateChamferDistance(set_B, set_A, D_B, "B", "A");
    double total_approximation = chamfer_A_to_B + chamfer_B_to_A;

    // Read the ground truth
    fs::path ground_truth_file_path = parent_directory_ / dataset_name_ / "gt.bin";
    std::ifstream ifs(ground_truth_file_path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open ground truth file.");
    }

    double ground_truth = 0.0;
    ifs.read(reinterpret_cast<char*>(&ground_truth), sizeof(ground_truth));

    // Calculate the relative error
    double relative_error = std::fabs(total_approximation - ground_truth) / ground_truth;
    std::cout << std::format("Relative Error: {:.3f}%\n", relative_error * 100.0);
}

auto ChamferEstimator::ApproximateChamferDistance(const std::vector<std::vector<double>>& from_set,
                                                  const std::vector<std::vector<double>>& to_set,
                                                  const std::vector<double>& from_set_D_array,
                                                  const std::string& from_set_name,
                                                  const std::string& to_set_name) const -> double {
    double approximation = 0.0;
    double sum = std::accumulate(from_set_D_array.begin(), from_set_D_array.end(), 0.0);

    for (unsigned int i = 0; i < num_samples_; i++) {
        if (verbose_) {
            std::cout << std::format("Approximate Chamfer Distance from {} to {}...({}/{})\r", from_set_name,
                                     to_set_name, i + 1, num_samples_)
                      << std::flush;
        }

        unsigned int point_id = Utils::SampleFromWeights(from_set_D_array);
        const auto& point = from_set.at(point_id);
        const double min_distance = Utils::MinDistance(point, to_set);
        approximation += sum * min_distance / from_set_D_array.at(point_id);
    }

    if (verbose_) {
        std::cout << "\n";
    }

    return approximation / num_samples_;
}

};  // namespace qalsh_chamfer