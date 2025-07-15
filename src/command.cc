#include "command.h"

#include <spdlog/spdlog.h>

// ---------------------------------------------
// GenerateDatasetCommand Implementation
// ---------------------------------------------

GenerateDatasetCommand::GenerateDatasetCommand(std::unique_ptr<DatasetGenerator> dataset_generator,
                                               std::filesystem::path dataset_directory)
    : dataset_generator_(std::move(dataset_generator)), dataset_directory_(std::move(dataset_directory)) {}

void GenerateDatasetCommand::Execute() {
    if (dataset_generator_ == nullptr) {
        spdlog::critical("Dataset generator is not set.");
    }
    dataset_generator_->Generate(dataset_directory_);
}

// ---------------------------------------------
// IndexCommand Implementation
// ---------------------------------------------

IndexCommand::IndexCommand(std::unique_ptr<Indexer> indexer) : indexer_(std::move(indexer)) {}

void IndexCommand::Execute() {
    if (indexer_ == nullptr) {
        spdlog::critical("Indexer is not set.");
    }

    indexer_->BuildIndex();
};

// ---------------------------------------------
// EstimateCommand Implementation
// ---------------------------------------------

EstimateCommand::EstimateCommand(std::unique_ptr<Estimator> estimator) : estimator_(std::move(estimator)) {}

void EstimateCommand::Execute() {
    if (estimator_ == nullptr) {
        spdlog::critical("Estimator is not set.");
    }

    double estimate = estimator_->Estimate();
    spdlog::info("Estimated Chamfer distance: {}", estimate);
}