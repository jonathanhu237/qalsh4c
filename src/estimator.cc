#include "estimator.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "ann_searcher.h"
#include "utils.h"

// --------------------------------------------------
// AnnEstimator Implementation
// --------------------------------------------------
AnnEstimator::AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher) : ann_searcher_(std::move(ann_searcher)) {}

double AnnEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, bool in_memory) {
    // Check the ANN searcher
    if (!ann_searcher_) {
        spdlog::error("The ANN searcher is not set.");
    }

    std::vector<double> distances(from.num_points, 0.0);
    ann_searcher_->Init(to);

    if (in_memory) {
        std::vector<Point> query_set = Utils::LoadPointsFromFile(from.file_path, from.num_points, from.num_dimensions);
        for (unsigned int point_id = 0; point_id < query_set.size(); point_id++) {
            distances[point_id] = ann_searcher_->Search(query_set[point_id]).distance;
        }
    } else {
        std::ifstream query_file(from.file_path, std::ios::binary);
        if (!query_file.is_open()) {
            spdlog::error("Failed to open query file: {}", from.file_path.string());
            return 0.0;
        }
        for (unsigned int point_id = 0; point_id < from.num_points; point_id++) {
            distances[point_id] =
                ann_searcher_->Search(Utils::ReadPoint(query_file, from.num_dimensions, point_id)).distance;
        }
    }

    return std::accumulate(distances.begin(), distances.end(), 0.0);
}

// --------------------------------------------------
// SamplingEstimator Implementation
// --------------------------------------------------
SamplingEstimator::SamplingEstimator(std::unique_ptr<WeightsGenerator> weights_generator, unsigned int num_samples,
                                     bool use_cache)
    : weights_generator_(std::move(weights_generator)), num_samples_(num_samples), use_cache_(use_cache) {}

double SamplingEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, bool in_memory) {
    // Check the ANN searcher
    if (!weights_generator_) {
        spdlog::error("The ANN searcher is not set.");
    }

    // Generate weights.
    spdlog::info("Generating weights...");
    std::vector<double> weights = weights_generator_->Generate(from, to, use_cache_);

    // Check the size of weights.
    if (weights.size() != from.num_points) {
        spdlog::error("Weights size does not match the number of query points in the dataset");
    }

    unsigned int num_samples = num_samples_;
    if (num_samples == 0) {
        // If num_samples is 0, set it to log(n)
        num_samples = static_cast<unsigned int>(std::log(from.num_points));
        spdlog::info("Number of samples set to log(n): {}", num_samples);
    }

    // Sample the points using the generated weights.
    double approximation = 0.0;
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);

    if (in_memory) {
        std::vector<Point> query_set = Utils::LoadPointsFromFile(from.file_path, from.num_points, from.num_dimensions);
        InMemoryLinearScanAnnSearcher ann_searcher;
        ann_searcher.Init(to);
        for (unsigned int i = 0; i < num_samples; i++) {
            unsigned int point_id = Utils::SampleFromWeights(weights);
            AnnResult result = ann_searcher.Search(query_set[point_id]);
            approximation += sum * result.distance / weights[point_id];
        }
    } else {
        std::ifstream query_file(from.file_path, std::ios::binary);
        if (!query_file.is_open()) {
            spdlog::error("Failed to open query file: {}", from.file_path.string());
            return 0.0;
        }
        DiskLinearScanAnnSearcher ann_searcher;
        ann_searcher.Init(to);
        for (unsigned int i = 0; i < num_samples; i++) {
            unsigned int point_id = Utils::SampleFromWeights(weights);
            AnnResult result = ann_searcher.Search(Utils::ReadPoint(query_file, from.num_dimensions, point_id));
            approximation += sum * result.distance / weights[point_id];
        }
    }

    return approximation / num_samples;
}