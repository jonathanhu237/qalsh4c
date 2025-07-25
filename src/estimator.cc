#include "estimator.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <memory>

#include "ann_searcher.h"
#include "point_set.h"
#include "types.h"

// ---------------------------------------------
// AnnEstimator Implementation
// ---------------------------------------------

AnnEstimator::AnnEstimator(std::string searcher_type) : searcher_type_(std::move(searcher_type)) {
    spdlog::debug("Creating AnnEstimator with searcher type: {}", searcher_type_);
}

double AnnEstimator::Estimate(const std::filesystem::path& dataset_directory) {
    // Load dataset metadata
    spdlog::debug("Loading dataset metadata");
    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory / "metadata.toml");

    // Initialize the base and query set readers
    spdlog::debug("Creating base and query set readers");
    auto base_set_reader{PointSetReaderFactory::Create(dataset_directory / "base.bin", dataset_metadata.data_type,
                                                       dataset_metadata.base_num_points,
                                                       dataset_metadata.num_dimensions)};
    auto query_set_reader{PointSetReaderFactory::Create(dataset_directory / "query.bin", dataset_metadata.data_type,
                                                        dataset_metadata.query_num_points,
                                                        dataset_metadata.num_dimensions)};

    // Initialize the ANN searcher
    spdlog::debug("Creating ANN searcher of type: {}", searcher_type_);
    std::unique_ptr<AnnSearcher> ann_searcher;
    if (searcher_type_ == "linear_scan") {
        ann_searcher = std::make_unique<LinearScanAnnSearcher>(base_set_reader.get());
    } else if (searcher_type_ == "qalsh") {
        ann_searcher = std::make_unique<QalshAnnSearcher>(base_set_reader.get(), dataset_directory / "qalsh_index");
    } else {
        spdlog::error("Unsupported searcher type: {}", searcher_type_);
        return 0.0;
    }

    if (!ann_searcher) {
        spdlog::error("Failed to create ANN searcher of type: {}", searcher_type_);
        return 0.0;
    }

    spdlog::debug("Calculating Chamfer distance using ANN searcher");
    double chamfer_distance{0.0};
    for (unsigned int i = 0; i < dataset_metadata.query_num_points; i++) {
        PointVariant query = query_set_reader->GetPoint(i);
        chamfer_distance += ann_searcher->Search(query).distance;
    }

    return chamfer_distance;
}
