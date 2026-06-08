#pragma once

#include <memory>

#include "ann_searcher.h"
#include "weights_generator.h"

class Estimator {
   public:
    virtual ~Estimator() = default;
    [[nodiscard]] virtual bool RequiresIndex() const = 0;
    virtual double EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, double norm_order) = 0;
};

class AnnEstimator : public Estimator {
   public:
    AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher);
    [[nodiscard]] bool RequiresIndex() const override;
    double EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, double norm_order) override;

   private:
    std::unique_ptr<AnnSearcher> ann_searcher_;
};

class SamplingEstimator : public Estimator {
   public:
    SamplingEstimator(std::unique_ptr<WeightsGenerator> weights_generator, unsigned int num_samples_default,
                      unsigned int num_samples_ab, unsigned int num_samples_ba, double epsilon, double delta,
                      bool use_cache, unsigned int exact_batch_size);
    [[nodiscard]] bool RequiresIndex() const override;
    double EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, double norm_order) override;

   private:
    std::unique_ptr<WeightsGenerator> weights_generator_;
    unsigned int num_samples_default_;  // Default sample count for both directions
    unsigned int num_samples_ab_;       // Sample count for A->B direction
    unsigned int num_samples_ba_;       // Sample count for B->A direction
    double epsilon_;
    double delta_;
    bool use_cache_;
    unsigned int exact_batch_size_;
};
