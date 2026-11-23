#pragma once

#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

#include "ann_searcher.h"
#include "weights_generator.h"

class Estimator {
   public:
    virtual ~Estimator() = default;
    virtual float EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, float norm_order) = 0;
};

// The exact Chamfer distance: one nearest-neighbor query per source point (the exact scan of the paper). With a
// save path, it also writes d(a, B) of every source point as float64, the input of SamplingOptions::exact_distances.
class AnnEstimator : public Estimator {
   public:
    explicit AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher, std::filesystem::path save_distances = {});
    float EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, float norm_order) override;

   private:
    std::unique_ptr<AnnSearcher> ann_searcher_;
    std::filesystem::path save_distances_;
};

struct SamplingOptions {
    unsigned int num_samples_default{0};  // T for both directions; 0 lets QALSH weights choose T from epsilon, delta
    unsigned int num_samples_ab{0};       // T for A -> B (overrides num_samples_default)
    unsigned int num_samples_ba{0};       // T for B -> A (overrides num_samples_default)
    float epsilon{Global::kQalsh4cEpsilon};
    float delta{Global::kQalsh4cDelta};
    bool use_cache{false};
    // Draw this many independent samples from the same weights; every estimate is logged.
    unsigned int repeats{1};
    // For experiments that repeat the estimator: d(a, B) of every source point (float64), looked up for the sampled
    // points instead of running the exact queries. The estimates are the same.
    std::filesystem::path exact_distances;
};

// The importance-sampling estimator of Bakshi et al.: weights D_a from a weights generator, T source points drawn with
// probability D_a / D, and eta = (1/T) sum_l D d(x_l, B) / D_{x_l}.
class SamplingEstimator : public Estimator {
   public:
    SamplingEstimator(std::unique_ptr<WeightsGenerator> weights_generator, SamplingOptions options);
    float EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, float norm_order) override;

   private:
    unsigned int SampleCount(const PointSetMetadata& from, const PointSetMetadata& to, float norm_order) const;
    // d(x, B) of the distinct sampled points, by the exact scan.
    std::vector<double> ExactDistancesOfSamples(const PointSetMetadata& from, const PointSetMetadata& to,
                                                float norm_order,
                                                const std::vector<std::pair<unsigned int, unsigned int>>& sample_counts,
                                                const std::vector<Weight>& weights,
                                                bool weights_are_upper_bounds) const;

    std::unique_ptr<WeightsGenerator> weights_generator_;
    SamplingOptions options_;
};
