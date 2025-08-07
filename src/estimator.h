#ifndef ESTIMATOR_H_
#define ESTIMATOR_H_

#include <memory>

#include "ann_searcher.h"
#include "weights_generator.h"

class Estimator {
   public:
    virtual ~Estimator() = default;
    virtual double EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, bool in_memory) = 0;
};

class AnnEstimator : public Estimator {
   public:
    AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher);
    double EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, bool in_memory) override;

   private:
    std::unique_ptr<AnnSearcher> ann_searcher_;
};

class SamplingEstimator : public Estimator {
   public:
    SamplingEstimator(std::unique_ptr<WeightsGenerator> weights_generator, unsigned int num_samples,
                      double delta_tolerance, bool use_cache);
    double EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, bool in_memory) override;

   private:
    std::unique_ptr<WeightsGenerator> weights_generator_;
    unsigned int num_samples_;
    double delta_threshold_;
    bool use_cache_;
};

#endif