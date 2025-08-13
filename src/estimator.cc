#include "estimator.h"

#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "ann_searcher.h"
#include "types.h"
#include "utils.h"
#include "weights_generator.h"

// --------------------------------------------------
// AnnEstimator Implementation
// --------------------------------------------------
AnnEstimator::AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher) : ann_searcher_(std::move(ann_searcher)) {}

double AnnEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, bool in_memory) {
    // Check the ANN searcher
    if (!ann_searcher_) {
        spdlog::error("The ANN searcher is not set.");
    }

    std::vector<double> distances;
    ann_searcher_->Init(to);

    if (in_memory) {
        std::vector<Point> query_set = Utils::LoadPointsFromFile(from.file_path, from.num_points, from.num_dimensions);
        for (const auto& point : query_set) {
            distances.emplace_back(ann_searcher_->Search(point).distance);
        }
    } else {
        std::ifstream query_file(from.file_path, std::ios::binary);
        if (!query_file.is_open()) {
            spdlog::error("Failed to open query file: {}", from.file_path.string());
            return 0.0;
        }
        for (unsigned int point_id = 0; point_id < from.num_points; point_id++) {
            distances.emplace_back(
                ann_searcher_->Search(Utils::ReadPoint(query_file, from.num_dimensions, point_id)).distance);
        }
    }

    return std::accumulate(distances.begin(), distances.end(), 0.0);
}

// --------------------------------------------------
// SamplingEstimator Implementation
// --------------------------------------------------
SamplingEstimator::SamplingEstimator(std::unique_ptr<WeightsGenerator> weights_generator, unsigned int num_samples,
                                     double approximation_ratio, double error_probability, bool use_cache)
    : weights_generator_(std::move(weights_generator)),
      num_samples_(num_samples),
      approximation_ratio_(approximation_ratio),
      error_probability_(error_probability),
      use_cache_(use_cache) {}

double SamplingEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, bool in_memory) {
    // Check the ANN searcher
    if (!weights_generator_) {
        spdlog::error("The ANN searcher is not set.");
    }

    // Update the num_samples_
    unsigned int updated_num_samples = num_samples_;
    if (updated_num_samples == 0) {
        if ([[maybe_unused]] auto* disk_qalsh = dynamic_cast<DiskQalshWeightsGenerator*>(weights_generator_.get())) {
            std::filesystem::path qalsh_config_path =
                to.file_path.parent_path() / "index" / to.file_path.stem() / "config.json";
            QalshConfig config = Utils::LoadQalshConfig(qalsh_config_path);
            updated_num_samples =
                static_cast<unsigned int>(std::ceil(1 / (error_probability_ * (config.approximation_ratio - 1))));
        } else {
            updated_num_samples =
                static_cast<unsigned int>(std::ceil(1 / (error_probability_ * (approximation_ratio_ - 1))));
        }
    }

    // Generate weights.
    spdlog::info("Generating weights...");
    std::vector<double> weights = weights_generator_->Generate(from, to, use_cache_);

    // Check the size of weights.
    if (weights.size() != from.num_points) {
        spdlog::error("Weights size does not match the number of query points in the dataset");
    }

    // Sample the points using the generated weights.
    double estimation = 0.0;
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    spdlog::info("Total sum of weights: {}", sum);

    std::unique_ptr<AnnSearcher> ann_searcher;

    auto processing_loop = [&](auto&& get_point_by_id) {
        spdlog::info("Sampling {} points from the weights...", updated_num_samples);
        for (unsigned int cnt = 1; cnt <= updated_num_samples; cnt++) {
            unsigned int point_id = Utils::SampleFromWeights(weights);
            spdlog::info("Sampled point ID: {}", point_id);
            estimation += (sum * ann_searcher->Search(get_point_by_id(point_id)).distance / weights[point_id]);
        }
    };

    if (in_memory) {
        ann_searcher = std::make_unique<InMemoryLinearScanAnnSearcher>();
        ann_searcher->Init(to);
        std::vector<Point> query_set = Utils::LoadPointsFromFile(from.file_path, from.num_points, from.num_dimensions);
        processing_loop([&](unsigned int id) { return query_set[id]; });
    } else {
        ann_searcher = std::make_unique<DiskLinearScanAnnSearcher>();
        ann_searcher->Init(to);
        std::ifstream query_file(from.file_path, std::ios::binary);
        if (!query_file.is_open()) {
            spdlog::error("Failed to open query file: {}", from.file_path.string());
            return 0.0;
        }
        processing_loop([&](unsigned int id) { return Utils::ReadPoint(query_file, from.num_dimensions, id); });
    }

    return estimation / updated_num_samples;
}