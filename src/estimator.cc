#include "estimator.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <memory>
#include <numeric>
#include <vector>

#include "ann_searcher.h"
#include "point_set.h"
#include "types.h"
#include "utils.h"
#include "weights_generator.h"

// ---------------------------------------------
// AnnEstimator Implementation
// ---------------------------------------------
AnnEstimator::AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher) : ann_searcher_(std::move(ann_searcher)) {}

double AnnEstimator::Estimate(const std::filesystem::path& dataset_directory) {
    // Load dataset metadata
    spdlog::info("Loading dataset metadata");
    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory / "metadata.toml");

    // Initialize the base and query set readers
    spdlog::info("Creating base and query set readers");
    auto base_set_reader{PointSetReaderFactory::Create(dataset_directory / "base.bin", dataset_metadata.data_type,
                                                       dataset_metadata.base_num_points,
                                                       dataset_metadata.num_dimensions)};
    auto query_set_reader{PointSetReaderFactory::Create(dataset_directory / "query.bin", dataset_metadata.data_type,
                                                        dataset_metadata.query_num_points,
                                                        dataset_metadata.num_dimensions)};

    if (!ann_searcher_) {
        spdlog::error("The ANN searcher is not set.");
        return 0.0;
    }

    spdlog::info("Calculating Chamfer distance using ANN searcher");
    ann_searcher_->Init(dataset_directory);
    double chamfer_distance{0.0};
    for (unsigned int i = 0; i < dataset_metadata.query_num_points; i++) {
        PointVariant query = query_set_reader->GetPoint(i);
        AnnResult result = ann_searcher_->Search(query);
        chamfer_distance += result.distance;
    }

    return chamfer_distance;
}

// ---------------------------------------------
// SamplingEstimator Implementation
// ---------------------------------------------
SamplingEstimator::SamplingEstimator(std::unique_ptr<WeightsGenerator> weights_generator, unsigned int num_samples)
    : weights_generator_(std::move(weights_generator)), num_samples_(num_samples) {}

double SamplingEstimator::Estimate(const std::filesystem::path& dataset_directory) {
    // Generate weights for the dataset.
    if (!weights_generator_) {
        spdlog::error("Weights generator is not set.");
        return 0.0;
    }
    std::vector<double> weights = weights_generator_->Generate(dataset_directory);

    // Load dataset metadata.
    spdlog::info("Loading dataset metadata");
    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory / "metadata.toml");

    // Check the size of weights.
    if (weights.size() != dataset_metadata.query_num_points) {
        spdlog::error("Weights size does not match the number of query points in the dataset");
        return 0.0;
    }

    // Sample the points using the generated weights.
    double approximation = 0.0;
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    auto query_set_reader{PointSetReaderFactory::Create(dataset_directory / "query.bin", dataset_metadata.data_type,
                                                        dataset_metadata.query_num_points,
                                                        dataset_metadata.num_dimensions)};

    if (num_samples_ == 0) {
        num_samples_ = static_cast<unsigned int>(std::log(dataset_metadata.query_num_points));
        spdlog::info("Number of samples set to log(n): {}", num_samples_);
    }

    LinearScanAnnSearcher linear_searcher;
    linear_searcher.Init(dataset_directory);
    for (unsigned int i = 0; i < num_samples_; i++) {
        unsigned int point_id = Utils::SampleFromWeights(weights);
        PointVariant query = query_set_reader->GetPoint(point_id);
        AnnResult result = linear_searcher.Search(query);
        approximation += sum * result.distance / weights[point_id];
    }

    return approximation / num_samples_;
}