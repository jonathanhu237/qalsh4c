#include "estimator.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "ann_searcher.h"
#include "config.h"
#include "global.h"
#include "log.h"
#include "point_set.h"
#include "utils.h"
#include "weights_generator.h"

// --------------------------------------------------
// AnnEstimator Implementation
// --------------------------------------------------
AnnEstimator::AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher) : ann_searcher_(std::move(ann_searcher)) {}

bool AnnEstimator::RequiresIndex() const {
    return ann_searcher_ && ann_searcher_->RequiresIndex();
}

double AnnEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, double norm_order) {
    // Check the ANN searcher
    if (!ann_searcher_) {
        Log::error("The ANN searcher is not set.");
    }

#ifdef QALSH4C_USE_OPENMP
    std::vector<Point> query_points = PointSet::LoadPointsFromFile(from.file_path, from.num_points, from.num_dimensions);
    std::vector<double> distances(from.num_points, 0.0);
    ann_searcher_->Init(to, norm_order);

#pragma omp parallel
    {
        std::unique_ptr<AnnSearcher> local_searcher = ann_searcher_->Clone();
        local_searcher->Init(to, norm_order);

#pragma omp for schedule(dynamic)
        for (long long query_index = 0; query_index < static_cast<long long>(from.num_points); ++query_index) {
            const unsigned int point_id = static_cast<unsigned int>(query_index);
            distances[point_id] = local_searcher->Search(query_points[point_id]).distance;
        }
    }
#else
    ann_searcher_->Init(to, norm_order);

    PointSet queries;
    queries.Init(from);

    std::vector<double> distances;
    distances.reserve(from.num_points);
    for (unsigned int point_id = 0; point_id < from.num_points; point_id++) {
        distances.emplace_back(ann_searcher_->Search(queries.ReadPoint(point_id)).distance);
    }
#endif

    return std::accumulate(distances.begin(), distances.end(), 0.0);
}

// --------------------------------------------------
// SamplingEstimator Implementation
// --------------------------------------------------
SamplingEstimator::SamplingEstimator(std::unique_ptr<WeightsGenerator> weights_generator,
                                     unsigned int num_samples_default, unsigned int num_samples_ab,
                                     unsigned int num_samples_ba, double epsilon)
    : weights_generator_(std::move(weights_generator)),
      num_samples_default_(num_samples_default),
      num_samples_ab_(num_samples_ab),
      num_samples_ba_(num_samples_ba),
      epsilon_(epsilon) {}

bool SamplingEstimator::RequiresIndex() const {
    return weights_generator_ && weights_generator_->RequiresIndex();
}

double SamplingEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to,
                                           double norm_order) {
    // Check the ANN searcher
    if (!weights_generator_) {
        Log::error("The ANN searcher is not set.");
    }

    // Determine the direction and select the appropriate sample count
    std::string from_name = from.file_path.stem().string();
    std::string to_name = to.file_path.stem().string();
    bool is_ab_direction = (from_name == "A" && to_name == "B");
    bool is_ba_direction = (from_name == "B" && to_name == "A");

    if (!is_ab_direction && !is_ba_direction) {
        Log::warn("Cannot determine direction from file names: {} -> {}, falling back to point count comparison",
                  from_name, to_name);
        is_ab_direction = (from.num_points > to.num_points);
    }

    // Select base sample count based on direction-specific parameters
    unsigned int base_num_samples = 0;
    if (is_ab_direction && num_samples_ab_ != 0) {
        base_num_samples = num_samples_ab_;
        Log::info("Using specified A->B sample count: {}", base_num_samples);
    } else if (is_ba_direction && num_samples_ba_ != 0) {
        base_num_samples = num_samples_ba_;
        Log::info("Using specified B->A sample count: {}", base_num_samples);
    } else {
        base_num_samples = num_samples_default_;
        if (base_num_samples != 0) {
            Log::info("Using default sample count: {}", base_num_samples);
        }
    }

    // Generate weights.
    Log::info("Generating weights...");
    std::vector<double> weights = weights_generator_->Generate(from, to, norm_order);

    // Update the num_samples.
    double updated_num_samples = base_num_samples;
    if (updated_num_samples == 0) {
        if (dynamic_cast<QalshWeightsGenerator*>(weights_generator_.get()) != nullptr) {
            std::optional<QalshConfig> qalsh_config = weights_generator_->GetQalshConfig();
            if (!qalsh_config.has_value()) {
                Log::error("QALSH config is unavailable for automatic sample count selection.");
                return 0.0;
            }

            const double alpha = Global::kQalsh4cAlpha;
            const double rho = Global::kQalsh4cRho;
            const double epsilon = epsilon_;

            const unsigned int m = qalsh_config->num_hash_tables;
            const unsigned int l = qalsh_config->collision_threshold;
            const double c = qalsh_config->approximation_ratio;

            double p1 = 0.0;
            if (std::abs(norm_order - 1.0) < Global::kEpsilon) {
                p1 = Calc::L1Probability(qalsh_config->bucket_width / 2.0);
            } else if (std::abs(norm_order - 2.0) < Global::kEpsilon) {
                p1 = Calc::L2Probability(qalsh_config->bucket_width / 2.0);
            } else {
                Log::error("Unsupported norm order: {}", norm_order);
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

            Log::info("SamplingEstimator automatically selected {} samples.", updated_num_samples);
        } else if ((dynamic_cast<UniformWeightsGenerator*>(weights_generator_.get()) != nullptr)) {
            Log::error("You must specify num_samples for UniformWeightsGenerator.");
            return 0.0;
        } else {
            Log::error("Unknown WeightsGenerator type.");
            return 0.0;
        }
    }

    // Check the size of weights.
    if (weights.size() != from.num_points) {
        Log::error("Weights size does not match the number of query points in the dataset");
    }

    // Sample the points using the generated weights.
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    Log::info("Total sum of weights: {}", sum);

    const unsigned int num_samples = static_cast<unsigned int>(updated_num_samples);
#ifdef QALSH4C_USE_OPENMP
    std::vector<unsigned int> sampled_ids;
    sampled_ids.reserve(num_samples);
    for (unsigned int sample_index = 0; sample_index < num_samples; ++sample_index) {
        sampled_ids.push_back(Sample::FromWeights(weights));
    }

    std::vector<Point> query_points = PointSet::LoadPointsFromFile(from.file_path, from.num_points, from.num_dimensions);
    double estimation = 0.0;

#pragma omp parallel reduction(+ : estimation)
    {
        LinearScanAnnSearcher ann_searcher;
        ann_searcher.Init(to, norm_order);

#pragma omp for schedule(dynamic)
        for (long long sample_index = 0; sample_index < static_cast<long long>(sampled_ids.size()); ++sample_index) {
            const unsigned int point_id = sampled_ids[static_cast<size_t>(sample_index)];
            estimation += (sum * ann_searcher.Search(query_points[point_id]).distance / weights[point_id]);
        }
    }
#else
    double estimation = 0.0;
    LinearScanAnnSearcher ann_searcher;
    ann_searcher.Init(to, norm_order);

    PointSet queries;
    queries.Init(from);
    for (unsigned int cnt = 1; cnt <= num_samples; cnt++) {
        unsigned int point_id = Sample::FromWeights(weights);
        estimation += (sum * ann_searcher.Search(queries.ReadPoint(point_id)).distance / weights[point_id]);
    }
#endif

    return estimation / updated_num_samples;
}
