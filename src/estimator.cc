#include "estimator.h"

#include <spdlog/spdlog.h>

#include <filesystem>

#include "ann_searcher.h"
#include "point_set.h"
#include "types.h"

// ---------------------------------------------
// LinearScanEstimator Implementation
// ---------------------------------------------

AnnEstimator::AnnEstimator(std::string searcher_type) : searcher_type_(std::move(searcher_type)) {}

double AnnEstimator::Estimate(const std::filesystem::path& dataset_directory) {
    // Load dataset metadata
    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory / "metadata.toml");

    // Initialize the base and query set readers
    auto base_set_reader{PointSetReaderFactory::Create(dataset_directory / "base.bin", dataset_metadata.data_type,
                                                       dataset_metadata.base_num_points,
                                                       dataset_metadata.num_dimensions)};
    auto query_set_reader{PointSetReaderFactory::Create(dataset_directory / "query.bin", dataset_metadata.data_type,
                                                        dataset_metadata.query_num_points,
                                                        dataset_metadata.num_dimensions)};

    // Initialize the ANN searcher
    auto ann_searcher = AnnSearcherFactory::Create(base_set_reader.get(), searcher_type_);
    double chamfer_distance{0.0};

    for (unsigned int i = 0; i < dataset_metadata.query_num_points; i++) {
        PointVariant query = query_set_reader->GetPoint(i);
        chamfer_distance += ann_searcher->Search(query).distance;
    }

    return chamfer_distance;
}
