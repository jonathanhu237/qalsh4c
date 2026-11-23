#pragma once

#include <memory>

#include "ann_searcher.h"
#include "weights_generator.h"

class Estimator {
   public:
    virtual ~Estimator() = default;
    [[nodiscard]] virtual bool RequiresIndex() const = 0;
    virtual void PreflightBothDirections(const PointSetMetadata& a_metadata, const PointSetMetadata& b_metadata,
                                         float norm_order);
    virtual float EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, float norm_order) = 0;
};

class AnnEstimator : public Estimator {
   public:
    AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher, bool disk_based = false,
                 unsigned int disk_block_size = Global::kDefaultDiskBlockSize,
                 unsigned int disk_query_batch_size = Global::kDefaultDiskQueryBatchSize);
    [[nodiscard]] bool RequiresIndex() const override;
    void PreflightBothDirections(const PointSetMetadata& a_metadata, const PointSetMetadata& b_metadata,
                                 float norm_order) override;
    float EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, float norm_order) override;

   private:
    std::unique_ptr<AnnSearcher> ann_searcher_;
    bool disk_based_{false};
    unsigned int disk_block_size_{Global::kDefaultDiskBlockSize};
    unsigned int disk_query_batch_size_{Global::kDefaultDiskQueryBatchSize};
};

class SamplingEstimator : public Estimator {
   public:
    SamplingEstimator(std::unique_ptr<WeightsGenerator> weights_generator, unsigned int num_samples_default,
                      unsigned int num_samples_ab, unsigned int num_samples_ba, float epsilon, float delta,
                      bool use_cache, unsigned int exact_batch_size, bool disk_based = false,
                      unsigned int disk_block_size = Global::kDefaultDiskBlockSize,
                      unsigned int disk_query_batch_size = Global::kDefaultDiskQueryBatchSize);
    [[nodiscard]] bool RequiresIndex() const override;
    void PreflightBothDirections(const PointSetMetadata& a_metadata, const PointSetMetadata& b_metadata,
                                 float norm_order) override;
    float EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, float norm_order) override;

   private:
    std::unique_ptr<WeightsGenerator> weights_generator_;
    unsigned int num_samples_default_;  // Default sample count for both directions
    unsigned int num_samples_ab_;       // Sample count for A->B direction
    unsigned int num_samples_ba_;       // Sample count for B->A direction
    float epsilon_;
    float delta_;
    bool use_cache_;
    unsigned int exact_batch_size_;
    bool disk_based_{false};
    unsigned int disk_block_size_{Global::kDefaultDiskBlockSize};
    unsigned int disk_query_batch_size_{Global::kDefaultDiskQueryBatchSize};
};
