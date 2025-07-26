#include "estimator.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <memory>
#include <numeric>
#include <vector>

#include "ann_searcher.h"
#include "global.h"
#include "point_set.h"
#include "types.h"
#include "utils.h"
#include "weights_generator.h"

// ---------------------------------------------
// AnnEstimator Implementation
// ---------------------------------------------

AnnEstimator::AnnEstimator(std::string searcher_type) : searcher_type_(std::move(searcher_type)) {
    spdlog::info("Creating AnnEstimator with searcher type: {}", searcher_type_);
}

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

    // Initialize the ANN searcher
    spdlog::info("Creating ANN searcher of type: {}", searcher_type_);
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

    spdlog::info("Calculating Chamfer distance using ANN searcher");
    double chamfer_distance{0.0};
    for (unsigned int i = 0; i < dataset_metadata.query_num_points; i++) {
        PointVariant query = query_set_reader->GetPoint(i);
        AnnResult result = ann_searcher->Search(query);
        chamfer_distance += result.distance;
    }

    if (searcher_type_ == "linear_scan" && Global::measure_time) {
        spdlog::info("Linear scan search took {:.2f} ms", Global::linear_scan_search_time_ms);
    }

    return chamfer_distance;
}

// ---------------------------------------------
// SamplingEstimator Implementation
// ---------------------------------------------
SamplingEstimator::SamplingEstimator(std::string searcher_type, unsigned int num_samples)
    : searcher_type_(std::move(searcher_type)), num_samples_(num_samples) {
    spdlog::info("Creating SamplingEstimator with searcher type: {}", searcher_type_);
}

double SamplingEstimator::Estimate(const std::filesystem::path& dataset_directory) {
    // Initialize the weights generator based on the searcher type.
    std::unique_ptr<WeightsGenerator> weights_generator;

    if (searcher_type_ == "uniform") {
        weights_generator = std::make_unique<UniformWeightsGenerator>();
    } else if (searcher_type_ == "qalsh") {
        weights_generator = std::make_unique<QalshWeightsGenerator>();
    } else {
        spdlog::error("Unsupported searcher type: {}", searcher_type_);
        return 0.0;
    }

    if (!weights_generator) {
        spdlog::error("Failed to create WeightsGenerator of type: {}", searcher_type_);
        return 0.0;
    }

    // Generate weights for the dataset.
    std::vector<double> weights = weights_generator->Generate(dataset_directory);

    // Load dataset metadata
    spdlog::info("Loading dataset metadata");
    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory / "metadata.toml");

    // Check the size of weights
    if (weights.size() != dataset_metadata.query_num_points) {
        spdlog::error("Weights size does not match the number of query points in the dataset");
        return 0.0;
    }

    // Sample the points using the generated weights.
    double approximation = 0.0;
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    auto base_set_reader{PointSetReaderFactory::Create(dataset_directory / "base.bin", dataset_metadata.data_type,
                                                       dataset_metadata.base_num_points,
                                                       dataset_metadata.num_dimensions)};
    auto query_set_reader{PointSetReaderFactory::Create(dataset_directory / "query.bin", dataset_metadata.data_type,
                                                        dataset_metadata.query_num_points,
                                                        dataset_metadata.num_dimensions)};
    LinearScanAnnSearcher linear_searcher(base_set_reader.get());

    if (num_samples_ == 0) {
        num_samples_ = static_cast<unsigned int>(std::log(dataset_metadata.query_num_points));
        spdlog::info("Number of samples set to log(n): {}", num_samples_);
    }

    for (unsigned int i = 0; i < num_samples_; i++) {
        unsigned int point_id = Utils::SampleFromWeights(weights);
        PointVariant query = query_set_reader->GetPoint(point_id);
        AnnResult result = linear_searcher.Search(query);
        approximation += sum * result.distance / weights[point_id];
    }

    return approximation / num_samples_;
}