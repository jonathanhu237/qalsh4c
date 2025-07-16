#include "command.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <format>
#include <ratio>

#include "types.h"

// ---------------------------------------------
// GenerateDatasetCommand Implementation
// ---------------------------------------------

GenerateDatasetCommand::GenerateDatasetCommand(std::unique_ptr<DatasetGenerator> dataset_generator,
                                               std::filesystem::path dataset_directory)
    : dataset_generator_(std::move(dataset_generator)), dataset_directory_(std::move(dataset_directory)) {}

void GenerateDatasetCommand::Execute() {
    if (dataset_generator_ == nullptr) {
        spdlog::error("Dataset generator is not set.");
    }
    dataset_generator_->Generate(dataset_directory_);
}

// ---------------------------------------------
// IndexCommand Implementation
// ---------------------------------------------

IndexCommand::IndexCommand(std::unique_ptr<Indexer> indexer, std::filesystem::path dataset_directory)
    : indexer_(std::move(indexer)), dataset_directory_(std::move(dataset_directory)) {}

void IndexCommand::Execute() {
    if (indexer_ == nullptr) {
        spdlog::error("Indexer is not set.");
    }

    indexer_->BuildIndex(dataset_directory_);
};

// ---------------------------------------------
// EstimateCommand Implementation
// ---------------------------------------------

EstimateCommand::EstimateCommand(std::unique_ptr<Estimator> estimator, std::filesystem::path dataset_directory)
    : estimator_(std::move(estimator)), dataset_directory_(std::move(dataset_directory)) {}

void EstimateCommand::Execute() {
    if (estimator_ == nullptr) {
        spdlog::error("Estimator is not set.");
    }

    auto start = std::chrono::high_resolution_clock::now();
    double estimate = estimator_->Estimate(dataset_directory_);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double estimated_time = elapsed.count();

    // Read the ground truth value from the metadata file
    DatasetMetadata metadata;
    metadata.Load(dataset_directory_ / "metadata.toml");

    // Calculate the relative error
    double relative_error = std::abs((estimate - metadata.chamfer_distance) / metadata.chamfer_distance);

    // Log the results
    spdlog::info(
        std::format("The result is as follows:\n"
                    "    Time Consumed: {:.2f} ms\n"
                    "    Estimated Chamfer distance: {}\n"
                    "    Relative error: {:.2f}%",
                    estimated_time, estimate, relative_error * 100.0));
}