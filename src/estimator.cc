#include "estimator.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ann_searcher.h"
#include "config.h"
#include "global.h"
#include "log.h"
#include "point_set.h"
#include "utils.h"
#include "weights_generator.h"

namespace {

double BinaryKlDivergence(double x, double y) {
    if (x < 0.0 || x > 1.0 || y <= 0.0 || y >= 1.0) {
        Log::error("Invalid KL divergence parameters: x={}, y={}", x, y);
    }

    const double left = x <= 0.0 ? 0.0 : x * std::log(x / y);
    const double right = x >= 1.0 ? 0.0 : (1.0 - x) * std::log((1.0 - x) / (1.0 - y));
    return left + right;
}

}  // namespace

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
                                     unsigned int num_samples_ba, double epsilon, double delta, bool use_cache)
    : weights_generator_(std::move(weights_generator)),
      num_samples_default_(num_samples_default),
      num_samples_ab_(num_samples_ab),
      num_samples_ba_(num_samples_ba),
      epsilon_(epsilon),
      delta_(delta),
      use_cache_(use_cache) {}

bool SamplingEstimator::RequiresIndex() const {
    return weights_generator_ && weights_generator_->RequiresIndex();
}

double SamplingEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to,
                                           double norm_order) {
    // Check the ANN searcher
    if (!weights_generator_) {
        Log::error("The ANN searcher is not set.");
    }
    const bool weights_are_upper_bounds = dynamic_cast<QalshWeightsGenerator*>(weights_generator_.get()) != nullptr;

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

    // Update the num_samples
    double updated_num_samples = base_num_samples;
    if (updated_num_samples == 0) {
        if (weights_are_upper_bounds) {
            const double epsilon = epsilon_;
            const double delta = delta_;
            if (epsilon <= 0.0) {
                Log::error("Epsilon must be positive for automatic QALSH sampling.");
            }
            if (delta <= 0.0 || delta >= 1.0) {
                Log::error("Delta must be in (0, 1) for automatic QALSH sampling.");
            }

            std::filesystem::path index_directory =
                to.file_path.parent_path() / "index" / std::format("l{}", norm_order) / to.file_path.stem();
            QalshConfig qalsh_config;
            qalsh_config.Load(index_directory / "config.json");

            const unsigned int m = qalsh_config.num_hash_tables;
            const unsigned int l = qalsh_config.collision_threshold;
            const double c = qalsh_config.approximation_ratio;
            if (m == 0) {
                Log::error("QALSH index has zero hash tables.");
            }

            double p1 = 0.0;
            if (std::abs(norm_order - 1.0) < Global::kEpsilon) {
                p1 = Calc::L1Probability(qalsh_config.bucket_width / 2.0);
            } else if (std::abs(norm_order - 2.0) < Global::kEpsilon) {
                p1 = Calc::L2Probability(qalsh_config.bucket_width / 2.0);
            } else {
                Log::error("Unsupported norm order: {}", norm_order);
                return 0.0;
            }

            const double theta = (static_cast<double>(l) - 1.0) / static_cast<double>(m);
            if (theta < 0.0 || theta >= p1) {
                Log::error("Invalid QALSH threshold ratio for automatic sampling: theta={}, p1={}", theta, p1);
            }

            const double kl_divergence = BinaryKlDivergence(theta, p1);
            const double delta0 = std::exp(-static_cast<double>(m) * kl_divergence);
            if (c * delta0 >= 1.0) {
                Log::error("Automatic QALSH sampling requires c * delta0 < 1, got c={}, delta0={}", c, delta0);
            }

            const double kappa_exp = (c * c) * (1.0 - delta0) / (1.0 - (c * delta0));
            const double min_samples = (kappa_exp - 1.0) / (delta * epsilon * epsilon);
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

    // Generate weights.
    Log::info("Generating weights...");
    std::vector<double> weights = weights_generator_->Generate(from, to, norm_order, use_cache_);

    // Check the size of weights.
    if (weights.size() != from.num_points) {
        Log::error("Weights size does not match the number of query points in the dataset");
    }

    // Sample the points using the generated weights.
    WeightedSampler sampler(weights);
    const double sum = sampler.Total();
    Log::info("Total sum of weights: {}", sum);

    const unsigned int num_samples = static_cast<unsigned int>(updated_num_samples);
    std::unordered_map<unsigned int, unsigned int> sample_count_map;
    sample_count_map.reserve(num_samples);
    for (unsigned int sample_index = 0; sample_index < num_samples; ++sample_index) {
        ++sample_count_map[sampler.Sample()];
    }

    std::vector<std::pair<unsigned int, unsigned int>> sample_counts;
    sample_counts.reserve(sample_count_map.size());
    for (const auto& [point_id, count] : sample_count_map) {
        sample_counts.emplace_back(point_id, count);
    }
    std::ranges::sort(sample_counts);

#ifdef QALSH4C_USE_OPENMP
    std::vector<Point> query_points = PointSet::LoadPointsFromFile(from.file_path, from.num_points, from.num_dimensions);
    double estimation = 0.0;

#pragma omp parallel reduction(+ : estimation)
    {
        LinearScanAnnSearcher ann_searcher;
        ann_searcher.Init(to, norm_order);

#pragma omp for schedule(dynamic)
        for (long long sample_index = 0; sample_index < static_cast<long long>(sample_counts.size()); ++sample_index) {
            const auto& [point_id, count] = sample_counts[static_cast<size_t>(sample_index)];
            const double initial_upper_bound =
                weights_are_upper_bounds ? weights[point_id] : std::numeric_limits<double>::infinity();
            const double nearest_distance =
                ann_searcher.SearchWithUpperBound(query_points[point_id], initial_upper_bound).distance;
            estimation +=
                (static_cast<double>(count) * sum * nearest_distance / weights[point_id]);
        }
    }
#else
    double estimation = 0.0;
    LinearScanAnnSearcher ann_searcher;
    ann_searcher.Init(to, norm_order);

    PointSet queries;
    queries.Init(from);
    for (const auto& [point_id, count] : sample_counts) {
        const double initial_upper_bound =
            weights_are_upper_bounds ? weights[point_id] : std::numeric_limits<double>::infinity();
        const double nearest_distance =
            ann_searcher.SearchWithUpperBound(queries.ReadPoint(point_id), initial_upper_bound).distance;
        estimation += (static_cast<double>(count) * sum * nearest_distance / weights[point_id]);
    }
#endif

    return estimation / updated_num_samples;
}
