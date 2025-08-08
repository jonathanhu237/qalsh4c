#include "estimator.h"

#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "ann_searcher.h"
#include "global.h"
#include "types.h"
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
                                     double approximation_ratio, double error_probability, bool use_cache)
    : weights_generator_(std::move(weights_generator)), num_samples_(num_samples), use_cache_(use_cache) {
    if (num_samples_ == 0) {
        num_samples_ = static_cast<unsigned int>(std::ceil(1 / (error_probability * (approximation_ratio - 1))));
    }
}

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

    // Sample the points using the generated weights.
    double estimation = 0.0;
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);

    std::unique_ptr<AnnSearcher> ann_searcher;

    auto processing_loop = [&](auto&& get_point_by_id) {
        for (unsigned int cnt = 0; cnt < num_samples_; cnt++) {
            unsigned int point_id = Utils::SampleFromWeights(weights);
            double prev_estimation = estimation;
            estimation = ((prev_estimation * (cnt - 1)) +
                          (sum * ann_searcher->Search(get_point_by_id(point_id)).distance / weights[point_id])) /
                         cnt;
            double estimation_delta = prev_estimation <= Global::kEpsilon
                                          ? std::numeric_limits<double>::max()
                                          : std::abs(estimation - prev_estimation) / prev_estimation;

            if (cnt == 1) {
                spdlog::debug("Number of samples: {}", cnt);
            }
            if (cnt > 1) {
                spdlog::debug("Number of samples: {}, Estimation Delta: {:.4f}, ", cnt, estimation_delta);
            }
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

    return estimation;
}