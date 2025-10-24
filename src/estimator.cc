#include "estimator.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "ann_searcher.h"
#include "config.h"
#include "global.h"
#include "point_set.h"
#include "utils.h"
#include "weights_generator.h"

// --------------------------------------------------
// AnnEstimator Implementation
// --------------------------------------------------
AnnEstimator::AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher) : ann_searcher_(std::move(ann_searcher)) {}

double AnnEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, double norm_order) {
    // Check the ANN searcher
    if (!ann_searcher_) {
        spdlog::error("The ANN searcher is not set.");
    }

    ann_searcher_->Init(to, norm_order);

    PointSet queries;
    queries.Init(from);

    std::vector<double> distances;
    distances.reserve(from.num_points);
    for (unsigned int point_id = 0; point_id < from.num_points; point_id++) {
        distances.emplace_back(ann_searcher_->Search(queries.ReadPoint(point_id)).distance);
    }

    return std::accumulate(distances.begin(), distances.end(), 0.0);
}

// --------------------------------------------------
// SamplingEstimator Implementation
// --------------------------------------------------
SamplingEstimator::SamplingEstimator(std::unique_ptr<WeightsGenerator> weights_generator, unsigned int num_samples,
                                     double epsilon, bool use_cache)
    : weights_generator_(std::move(weights_generator)),
      num_samples_(num_samples),
      epsilon_(epsilon),
      use_cache_(use_cache) {}

double SamplingEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to,
                                           double norm_order) {
    // Check the ANN searcher
    if (!weights_generator_) {
        spdlog::error("The ANN searcher is not set.");
    }

    // Update the num_samples_
    double updated_num_samples = num_samples_;
    if (updated_num_samples == 0) {
        if (dynamic_cast<QalshWeightsGenerator*>(weights_generator_.get()) != nullptr) {
            const double alpha = Global::kQalsh4cAlpha;
            const double rho = Global::kQalsh4cRho;
            const double epsilon = epsilon_;

            std::filesystem::path index_directory =
                to.file_path.parent_path() / "index" / std::format("l{}", norm_order) / to.file_path.stem();
            QalshConfig qalsh_config;
            qalsh_config.Load(index_directory / "config.json");

            const unsigned int m = qalsh_config.num_hash_tables;
            const unsigned int l = qalsh_config.collision_threshold;
            const double c = qalsh_config.approximation_ratio;

            double p1 = 0.0;
            if (std::abs(norm_order - 1.0) < Global::kEpsilon) {
                p1 = Calc::L1Probability(qalsh_config.bucket_width / 2.0);
            } else if (std::abs(norm_order - 2.0) < Global::kEpsilon) {
                p1 = Calc::L2Probability(qalsh_config.bucket_width / 2.0);
            } else {
                spdlog::error("Unsupported norm order: {}", norm_order);
                return 0.0;
            }

            double q = (static_cast<double>(l) - 1.0) / static_cast<double>(m);
            double p = p1;
            double kl_divergence = (q * std::log(q / p)) + ((1.0 - q) * std::log((1.0 - q) / (1.0 - p)));
            double delta0 = std::exp(-static_cast<double>(m) * kl_divergence);

            double kappa = (c * c) * (1.0 - delta0) / (1.0 - (c * delta0));
            double kappa_tilde = alpha * kappa;

            double M_tilde = std::max(kappa_tilde - 1.0, 1.0);

            double numerator = (2.0 * (kappa_tilde - 1.0)) + ((2.0 / 3.0) * M_tilde * epsilon);
            double denominator = epsilon * epsilon;

            double inner = 1.0 - (rho / (1.0 - (1.0 / alpha)));
            double log_term = std::log(2.0 / inner);

            double min_samples = (numerator / denominator) * log_term;
            updated_num_samples = static_cast<unsigned int>(std::ceil(min_samples));

            spdlog::info("SamplingEstimator automatically selected {} samples.", updated_num_samples);
        } else if ((dynamic_cast<UniformWeightsGenerator*>(weights_generator_.get()) != nullptr)) {
            spdlog::error("You must specify num_samples for UniformWeightsGenerator.");
            return 0.0;
        } else {
            spdlog::error("Unknown WeightsGenerator type.");
            return 0.0;
        }
    }

    // Generate weights.
    spdlog::info("Generating weights...");
    std::vector<double> weights = weights_generator_->Generate(from, to, norm_order, use_cache_);

    // Check the size of weights.
    if (weights.size() != from.num_points) {
        spdlog::error("Weights size does not match the number of query points in the dataset");
    }

    // Sample the points using the generated weights.
    double estimation = 0.0;
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    spdlog::info("Total sum of weights: {}", sum);

    std::unique_ptr<AnnSearcher> ann_searcher;

    ann_searcher = std::make_unique<LinearScanAnnSearcher>();
    ann_searcher->Init(to, norm_order);

    PointSet queries;
    queries.Init(from);
    for (unsigned int cnt = 1; cnt <= updated_num_samples; cnt++) {
        unsigned int point_id = Sample::FromWeights(weights);
        estimation += (sum * ann_searcher->Search(queries.ReadPoint(point_id)).distance / weights[point_id]);
    }

    return estimation / updated_num_samples;
}
