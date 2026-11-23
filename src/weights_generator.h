#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "ann_searcher.h"
#include "config.h"
#include "global.h"
#include "point_set.h"

// --------------------------------------------------
// WeightsGenerator Definition
// --------------------------------------------------
class WeightsGenerator {
   public:
    virtual ~WeightsGenerator() = default;
    [[nodiscard]] virtual bool RequiresIndex() const = 0;
    virtual std::vector<Weight> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                         float norm_order, bool use_cache) = 0;
};

// --------------------------------------------------
// UniformWeightsGenerator Definition
// --------------------------------------------------
class UniformWeightsGenerator : public WeightsGenerator {
   public:
    UniformWeightsGenerator() = default;
    [[nodiscard]] bool RequiresIndex() const override;
    std::vector<Weight> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 float norm_order, bool use_cache) override;
};

// --------------------------------------------------
// QalshWeightsGenerator Definition
// --------------------------------------------------
// Weights read from a file of float32 values, one per source point: the crude distances of an earlier run or of
// another CrudeNN (for example HNSW), for experiments that repeat the estimator on fixed weights.
class PrecomputedWeightsGenerator : public WeightsGenerator {
   public:
    explicit PrecomputedWeightsGenerator(std::filesystem::path path);
    [[nodiscard]] bool RequiresIndex() const override;
    std::vector<Weight> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 float norm_order, bool use_cache) override;

   private:
    std::filesystem::path path_;
};

class QalshWeightsGenerator : public WeightsGenerator {
   public:
    explicit QalshWeightsGenerator(float approximation_ratio = Global::kDefaultApproximationRatio);
    [[nodiscard]] bool RequiresIndex() const override;
    [[nodiscard]] QalshConfig BuildConfig(const PointSetMetadata& from_metadata,
                                          const PointSetMetadata& to_metadata, float norm_order,
                                          bool use_cache) const;
    [[nodiscard]] std::shared_ptr<const std::vector<Point>> TargetPoints() const;
    std::vector<Weight> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 float norm_order, bool use_cache) override;

   private:
    float approximation_ratio_{Global::kDefaultApproximationRatio};
    std::unique_ptr<AnnSearcher> ann_searcher_;
};

// --------------------------------------------------
// QuadtreeWeightsGenerator Definition
// --------------------------------------------------
class QuadtreeWeightsGenerator : public WeightsGenerator {
   public:
    static constexpr unsigned int kDefaultMaxLevels = 3;

    explicit QuadtreeWeightsGenerator(unsigned int max_levels = kDefaultMaxLevels);
    [[nodiscard]] bool RequiresIndex() const override;
    std::vector<Weight> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 float norm_order, bool use_cache) override;

   private:
    unsigned int max_levels_{kDefaultMaxLevels};
};
