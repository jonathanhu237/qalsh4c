#include "estimator.h"

#include <memory>

#include "dataset_metadata.h"
#include "point_set.h"

// --------------------------------------------------
// AnnEstimator Implementation
// --------------------------------------------------
AnnEstimator::AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher) : ann_searcher_(std::move(ann_searcher)) {}

void AnnEstimator::set_in_memory(bool in_memory) { in_memory_ = in_memory; }

EstimateResult AnnEstimator::Estimate(const std::filesystem::path& dataset_directory) {
    // Load dataset metadata
    spdlog::info("Loading dataset metadata...");
    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory / "metadata.json");

    // Check the ANN searcher
    if (!ann_searcher_) {
        spdlog::error("The ANN searcher is not set.");
    }

    // Construct point set metadata
    PointSetMetadata point_set_metadata_a = {
        .file_path = dataset_directory / "A.bin",
        .num_points = dataset_metadata.num_points_a,
        .num_dimensions = dataset_metadata.num_dimensions,
    };
    PointSetMetadata point_set_metadata_b = {
        .file_path = dataset_directory / "B.bin",
        .num_points = dataset_metadata.num_points_b,
        .num_dimensions = dataset_metadata.num_dimensions,
    };

    // Calculate the distance from A to B
    spdlog::info("Calculating the distance from A to B...");
    double distance_ab = CalculateDistance(point_set_metadata_a, point_set_metadata_b);

    // Calculate the distance from B to A
    spdlog::info("Calculating the distance from B to A...");
    double distance_ba = CalculateDistance(point_set_metadata_b, point_set_metadata_a);

    // Retrun the result
    EstimateResult res;
    res.chamfer_distance = distance_ab + distance_ba;
    res.relative_error =
        std::fabs(res.chamfer_distance - dataset_metadata.chamfer_distance) / dataset_metadata.chamfer_distance;
    return res;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double AnnEstimator::CalculateDistance(const PointSetMetadata& from, const PointSetMetadata& to) {
    ann_searcher_->Reset();
    ann_searcher_->Init(to, in_memory_);
    double distance = 0;

    std::unique_ptr<PointSet> query_set;
    if (in_memory_) {
        query_set = std::make_unique<InMemoryPointSet>(from);
    } else {
        query_set = std::make_unique<DiskPointSet>(from);
    }

    for (unsigned int point_id = 0; point_id < query_set->get_num_points(); point_id++) {
        Point query_point = query_set->GetPoint(point_id);
        distance += ann_searcher_->Search(query_point).distance;
    }

    return distance;
}