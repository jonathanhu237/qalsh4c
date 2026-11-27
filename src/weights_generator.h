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
    // True when every weight is the distance from its source point to some target point, so it bounds the nearest
    // distance from above and the exact stage may use it to prune. The exact stage treats it only as a hint: a sampled
    // point is assigned the distance of an actual target, and a point with no target within its hint is scanned again
    // without a bound.
    [[nodiscard]] virtual bool WeightsAreUpperBounds() const { return false; }
    // The target points the last Generate loaded, for the exact stage to reuse; empty when it loaded none.
    [[nodiscard]] virtual std::shared_ptr<const std::vector<Point>> TargetPoints() const { return {}; }
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
    // `upper_bounds`: the file holds distances to actual target points (QALSH or ANN crude distances).
    explicit PrecomputedWeightsGenerator(std::filesystem::path path, bool upper_bounds = false);
    [[nodiscard]] bool RequiresIndex() const override;
    std::vector<Weight> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 float norm_order, bool use_cache) override;
    [[nodiscard]] bool WeightsAreUpperBounds() const override { return upper_bounds_; }

   private:
    std::filesystem::path path_;
    bool upper_bounds_{false};
};

class QalshWeightsGenerator : public WeightsGenerator {
   public:
    explicit QalshWeightsGenerator(float approximation_ratio = Global::kDefaultApproximationRatio);
    [[nodiscard]] bool RequiresIndex() const override;
    [[nodiscard]] QalshConfig BuildConfig(const PointSetMetadata& from_metadata,
                                          const PointSetMetadata& to_metadata, float norm_order,
                                          bool use_cache) const;
    [[nodiscard]] std::shared_ptr<const std::vector<Point>> TargetPoints() const override;
    std::vector<Weight> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 float norm_order, bool use_cache) override;
    // QALSH crude distances are distances to target points.
    [[nodiscard]] bool WeightsAreUpperBounds() const override { return true; }

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
    // A quadtree weight is the distance to the next target in depth-first order; the floor that replaces a zero is
    // still above the true distance 0, and the exact stage only uses the weights as hints.
    [[nodiscard]] bool WeightsAreUpperBounds() const override { return true; }
    [[nodiscard]] std::shared_ptr<const std::vector<Point>> TargetPoints() const override { return target_points_; }

   private:
    unsigned int max_levels_{kDefaultMaxLevels};
    std::shared_ptr<const std::vector<Point>> target_points_;
};
