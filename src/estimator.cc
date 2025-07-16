#include "estimator.h"

#include <spdlog/spdlog.h>

#include <filesystem>

#include "point_set.h"

LinearScanEstimator::LinearScanEstimator(bool in_memory) : in_memory_(in_memory) {}

double LinearScanEstimator::Estimate(const std::filesystem::path& dataset_directory) {
    // Load dataset metadata
    dataset_directory_ = dataset_directory;
    dataset_metadata_.Load(dataset_directory_ / "metadata.toml");

    // Initialize the base and query set readers
    auto base_set_reader{PointSetReaderFactory::Create(dataset_directory_ / "base.bin", dataset_metadata_.data_type,
                                                       dataset_metadata_.base_num_points,
                                                       dataset_metadata_.num_dimensions, in_memory_)};
    auto query_set_reader{PointSetReaderFactory::Create(dataset_directory_ / "query.bin", dataset_metadata_.data_type,
                                                        dataset_metadata_.query_num_points,
                                                        dataset_metadata_.num_dimensions, in_memory_)};

    // Print the configuration
    PrintConfiguration();

    spdlog::info("Calculating Chamfer distance using Linear Scan Estimator...");
    double chamfer_distance{0.0};

    for (unsigned int i = 0; i < dataset_metadata_.query_num_points; i++) {
        PointVariant query = query_set_reader->GetPoint(i);
        chamfer_distance += base_set_reader->CalculateDistance(query);
    }

    return chamfer_distance;
}

void LinearScanEstimator::PrintConfiguration() const {
    spdlog::debug(
        "The configuration is as follows:\n"
        "    Dataset Directory: {}\n"
        "    Data Type: {}\n"
        "    Number of Points in Base Set: {}\n"
        "    Number of Points in Query Set: {}\n"
        "    Number of Dimensions: {}\n"
        "    In Memory: {}",
        dataset_directory_.string(), dataset_metadata_.data_type, dataset_metadata_.base_num_points,
        dataset_metadata_.query_num_points, dataset_metadata_.num_dimensions, in_memory_);
}