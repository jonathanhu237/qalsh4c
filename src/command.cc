#include "command.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <memory>
#include <random>
#include <ratio>

#include "dataset_metadata.h"
#include "estimator.h"
#include "point_set.h"

// --------------------------------------------------
// GenerateDatasetCommand Implementation
// --------------------------------------------------
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
GenerateDatasetCommand::GenerateDatasetCommand(DatasetMetadata dataset_metadata, double left_boundary,
                                               double right_boundary, std::filesystem::path output_directory,
                                               bool in_memory)
    : dataset_metadata_(dataset_metadata),
      left_boundary_(left_boundary),
      right_boundary_(right_boundary),
      output_directory_(std::move(output_directory)),
      in_memory_(in_memory),
      gen_(std::random_device{}()) {}

void GenerateDatasetCommand::Execute() {
    // Create the directory if it does not exist.
    if (!std::filesystem::exists(output_directory_)) {
        spdlog::info("Creating dataset directory: {}", output_directory_.string());
        std::filesystem::create_directory(output_directory_);
    }

    // Generate the point sets.
    spdlog::info("Generating point set A...");
    GeneratePointSet(output_directory_ / "A.bin", dataset_metadata_.num_points_a);

    spdlog::info("Generating point set B...");
    GeneratePointSet(output_directory_ / "B.bin", dataset_metadata_.num_points_b);

    // We must save the metadata first since it will be used by the estimator (update the chamfer distance later)
    dataset_metadata_.Save(output_directory_ / "metadata.toml");

    // Calculate the Chamfer distance between the base and query sets
    spdlog::info("Calculating Chamfer distance between base and query sets...");
    AnnEstimator ann_estimator(std::make_unique<LinearScanAnnSearcher>());
    dataset_metadata_.chamfer_distance = ann_estimator.Estimate(output_directory_);

    // Save the updated metadata
    dataset_metadata_.Save(output_directory_ / "metadata.toml");
}

void GenerateDatasetCommand::GeneratePointSet(const std::string &point_set_file_path, unsigned int num_points) {
    std::unique_ptr<PointSetWriter> pointset;
    if (in_memory_) {
        pointset = std::make_unique<InMemoryPointSetWriter>(point_set_file_path);
    } else {
        pointset = std::make_unique<DiskPointSetWriter>(point_set_file_path);
    }

    std::uniform_real_distribution<double> dist(left_boundary_, right_boundary_);
    for (unsigned int i = 0; i < num_points; i++) {
        Point point(dataset_metadata_.num_dimensions);
        std::ranges::generate(point, [&]() { return dist(gen_); });
        pointset->AddPoint(point);
    }

    pointset->Flush();
}

// --------------------------------------------------
// EstimateCommand Implementation
// --------------------------------------------------
EstimateCommand::EstimateCommand(std::unique_ptr<Estimator> estimator, std::filesystem::path dataset_directory,
                                 bool in_memory)
    : estimator_(std::move(estimator)), dataset_directory_(std::move(dataset_directory)), in_memory_(in_memory) {}

void EstimateCommand::Execute() {
    estimator_->set_in_memory(in_memory_);

    auto start = std::chrono::high_resolution_clock::now();
    double estimation = estimator_->Estimate(dataset_directory_);
    auto end = std::chrono::high_resolution_clock::now();

    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory_ / "metadata.json");

    // Output the result.
    std::cout << std::format(
        "The result is as follows:\n"
        "    Time Consumed: {:.2f} ms\n"
        "    Estimated Chamfer distance: {}\n"
        "    Relative error: {:.2f}%\n",
        std::chrono::duration<double, std::milli>(end - start).count(), estimation,
        std::fabs(estimation - dataset_metadata.chamfer_distance) / dataset_metadata.chamfer_distance *
            100);  // NOLINT: readability-magic-numbers
}