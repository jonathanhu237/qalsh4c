#include "command.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <ratio>
#include <utility>

#include "estimator.h"
#include "global.h"
#include "log.h"
#include "point_set.h"
#include "utils.h"

namespace {

PointSetMetadata MakePointSetMetadata(const std::filesystem::path& dataset_directory,
                                      const DatasetMetadata& dataset_metadata, const std::string& point_set_name) {
    const bool is_a = point_set_name == "A";
    return PointSetMetadata{
        .file_path = dataset_directory / std::format("{}.bin", point_set_name),
        .num_points = is_a ? dataset_metadata.num_points_a : dataset_metadata.num_points_b,
        .num_dimensions = dataset_metadata.num_dimensions,
    };
}

double RelativeErrorPercentage(double estimation, double ground_truth) {
    if (std::abs(ground_truth) < Global::kEpsilon) {
        return std::abs(estimation) < Global::kEpsilon ? 0.0 : std::numeric_limits<double>::infinity();
    }

    return std::fabs(estimation - ground_truth) / ground_truth * 100;  // NOLINT: readability-magic-numbers
}

}  // namespace

// --------------------------------------------------
// EstimateCommand Implementation
// --------------------------------------------------
EstimateCommand::EstimateCommand(std::unique_ptr<Estimator> estimator, double norm_order,
                                 std::filesystem::path dataset_directory, ChamferDirection direction)
    : estimator_(std::move(estimator)),
      norm_order_(norm_order),
      dataset_directory_(std::move(dataset_directory)),
      direction_(direction) {}

void EstimateCommand::Execute() {
    // Load dataset metadata
    Log::info("Loading dataset metadata...");
    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory_ / "metadata.json");

    // Construct point set metadata
    PointSetMetadata point_set_metadata_a = MakePointSetMetadata(dataset_directory_, dataset_metadata, "A");
    PointSetMetadata point_set_metadata_b = MakePointSetMetadata(dataset_directory_, dataset_metadata, "B");

    auto start = std::chrono::high_resolution_clock::now();
    double memory_before = Memory::Usage();
    double estimation{0.0};

    if (direction_ == ChamferDirection::kAb || direction_ == ChamferDirection::kBoth) {
        Log::info("Calculating the distance from A to B...");
        estimation += estimator_->EstimateDistance(point_set_metadata_a, point_set_metadata_b, norm_order_);
    }

    if (direction_ == ChamferDirection::kBa || direction_ == ChamferDirection::kBoth) {
        Log::info("Calculating the distance from B to A...");
        estimation += estimator_->EstimateDistance(point_set_metadata_b, point_set_metadata_a, norm_order_);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double memory_after = Memory::Usage();

    // Output the result.
    std::optional<double> ground_truth = dataset_metadata.GroundTruth(norm_order_, direction_);
    double elapsed_time = std::chrono::duration<double, std::milli>(end - start).count();
    double memory_usage = memory_after - memory_before;

    if (ground_truth.has_value()) {
        double relative_error_percentage = RelativeErrorPercentage(estimation, *ground_truth);
        std::cout << std::format(
            "Direction: {}\n"
            "Estimated Distance: {:.6f}\n"
            "Ground Truth: {:.6f}\n"
            "Time Consumed: {:.3f} ms\n"
            "Memory Usage: {:.2f} MB\n"
            "Relative Error: {:.2f}%\n",
            ChamferDirectionLabel(direction_), estimation, *ground_truth, elapsed_time, memory_usage,
            relative_error_percentage);
        return;
    }

    std::cout << std::format(
        "Direction: {}\n"
        "Estimated Distance: {:.6f}\n"
        "Ground Truth: unavailable\n"
        "Time Consumed: {:.3f} ms\n"
        "Memory Usage: {:.2f} MB\n"
        "Relative Error: unavailable\n",
        ChamferDirectionLabel(direction_), estimation, elapsed_time, memory_usage);
}

// --------------------------------------------------
// Dataset Commands Implementation
// --------------------------------------------------
DatasetGenerateGmmCommand::DatasetGenerateGmmCommand(DatasetPrep::GmmOptions options)
    : options_(std::move(options)) {}

void DatasetGenerateGmmCommand::Execute() {
    DatasetPrep::GenerateGmmDataset(options_);
    std::cout << std::format("Dataset Directory: {}\n", options_.output.output_directory.string());
}

DatasetConvertCommand::DatasetConvertCommand(DatasetPrep::ConvertOptions options) : options_(std::move(options)) {}

void DatasetConvertCommand::Execute() {
    DatasetPrep::ConvertDataset(options_);
    std::cout << std::format("Dataset Directory: {}\n", options_.output.output_directory.string());
}

DatasetGroundTruthCommand::DatasetGroundTruthCommand(DatasetPrep::GroundTruthOptions options)
    : options_(std::move(options)) {}

void DatasetGroundTruthCommand::Execute() {
    DatasetPrep::ComputeDatasetGroundTruth(options_);
    std::cout << std::format("Dataset Directory: {}\n", options_.dataset_directory.string());
}
