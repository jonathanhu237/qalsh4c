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

// How PoolSamplingEstimator chooses its pool size K and sample count T.
enum class PoolPlanMode {
    kFixed,      // K and T as given, or the QALSH4C sample count with kAutoPoolSizePerSample draws per sample
    kModel,      // A pilot estimates the variance coefficients and costs; K and T target an RMS relative error
    kCertified,  // A pilot certifies upper bounds on the coefficients; K and T meet an (epsilon, delta) guarantee
};

// Two-stage QALSH4C. Every source point gets a cheap upper bound from a uniform target subsample; a pool is
// drawn in proportion to these bounds, Modified QALSH runs only on pooled points, and exact scans run on
// points resampled from the pool in proportion to D / U.
class PoolSamplingEstimator : public Estimator {
   public:
    // A pool size of 0 selects this many first-stage draws per second-stage sample, the ratio chosen on the
    // MNIST development workloads for both norms.
    static constexpr unsigned int kAutoPoolSizePerSample = 5;
    // Pilot draws in proportion to U_a for the model and certified plans.
    static constexpr unsigned int kDefaultPilotSize = 256;

    PoolSamplingEstimator(unsigned int pool_size, unsigned int num_samples, unsigned int subsample_size,
                          float approximation_ratio, float epsilon, float delta, unsigned int exact_batch_size,
                          bool disk_based = false, unsigned int disk_block_size = Global::kDefaultDiskBlockSize,
                          unsigned int disk_query_batch_size = Global::kDefaultDiskQueryBatchSize,
                          std::optional<float> expected_approximation_ratio = std::nullopt,
                          DiskQalshSearchOptions disk_qalsh_options = {}, bool direct = false, bool full = false,
                          bool fuse = true, PoolPlanMode plan_mode = PoolPlanMode::kFixed, float target_error = 0.0F,
                          unsigned int pilot_size = kDefaultPilotSize, bool uniform = false);
    [[nodiscard]] bool RequiresIndex() const override;
    void PreflightBothDirections(const PointSetMetadata& a_metadata, const PointSetMetadata& b_metadata,
                                 float norm_order) override;
    float EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, float norm_order) override;

   private:
    // External-memory variant: streams A once, probes the persistent QALSH index only for pooled points, and scans
    // B sequentially in query batches for the exact stage.
    float EstimateDiskBased(const PointSetMetadata& from, const PointSetMetadata& to, float norm_order);
    unsigned int ResolveNumSamples(const QalshConfig& qalsh_config, float norm_order) const;
    unsigned int ResolvePoolSize(unsigned int num_samples) const;

    unsigned int pool_size_;       // Number of first-stage draws (K); 0 selects kAutoPoolSizePerSample * T
    unsigned int num_samples_;     // Number of second-stage draws (T); 0 selects the QALSH4C sample count
    unsigned int subsample_size_;  // Size of the uniform target subsample used for upper bounds (s)
    float approximation_ratio_;
    float epsilon_;
    float delta_;
    unsigned int exact_batch_size_;
    bool disk_based_{false};
    unsigned int disk_block_size_{Global::kDefaultDiskBlockSize};
    unsigned int disk_query_batch_size_{Global::kDefaultDiskQueryBatchSize};
    std::optional<float> expected_approximation_ratio_;
    DiskQalshSearchOptions disk_qalsh_options_{};
    bool direct_{false};  // Baseline: skip QALSH and scan points drawn in proportion to U_a directly
    bool full_{false};    // Ablation: query every source point instead of a pool (in memory only)
    bool fuse_{true};     // Clip QALSH distances with the subsample upper bounds, D_a = min(QALSH(a), U_a)
    PoolPlanMode plan_mode_{PoolPlanMode::kFixed};
    float target_error_{0.0F};  // RMS relative error (model) or epsilon with delta_ (certified)
    unsigned int pilot_size_{kDefaultPilotSize};
    bool uniform_{false};  // Baseline: uniform draws without bounds or QALSH, scanned from scratch
};
