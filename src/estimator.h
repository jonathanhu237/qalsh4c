#ifndef ESTIMATOR_H_
#define ESTIMATOR_H_

#include <filesystem>
#include <memory>

#include "ann_searcher.h"
#include "weights_generator.h"

class Estimator {
   public:
    virtual ~Estimator() = default;
    virtual double Estimate(const std::filesystem::path& dataset_directory) = 0;
};

class AnnEstimator : public Estimator {
   public:
    AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher);
    double Estimate(const std::filesystem::path& dataset_directory) override;

   private:
    std::unique_ptr<AnnSearcher> ann_searcher_;
};

class SamplingEstimator : public Estimator {
   public:
    SamplingEstimator(std::unique_ptr<WeightsGenerator> weights_generator, unsigned int num_samples,
                      bool use_cache = false);
    double Estimate(const std::filesystem::path& dataset_directory) override;

   private:
    std::unique_ptr<WeightsGenerator> weights_generator_;
    unsigned int num_samples_;
    bool use_cache_;
};

#endif