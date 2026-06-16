#pragma once

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
class QalshWeightsGenerator : public WeightsGenerator {
   public:
    explicit QalshWeightsGenerator(float approximation_ratio = Global::kDefaultApproximationRatio);
    [[nodiscard]] bool RequiresIndex() const override;
    [[nodiscard]] QalshConfig BuildConfig(unsigned int num_points, float norm_order) const;
    [[nodiscard]] std::shared_ptr<const std::vector<Point>> TargetPoints() const;
    std::vector<Weight> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 float norm_order, bool use_cache) override;

   private:
    float approximation_ratio_{Global::kDefaultApproximationRatio};
    std::unique_ptr<AnnSearcher> ann_searcher_;
};
