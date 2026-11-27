#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <qalsh/qalsh.h>

#include "config.h"
#include "global.h"
#include "kappa.h"
#include "point_set.h"

struct AnnResult {
    Distance distance{0.0F};
    unsigned int point_id{0};
};

struct CompareAnnResult {
    bool operator()(const AnnResult& a, const AnnResult& b) const { return a.distance > b.distance; }
};

class AnnSearcher {
   public:
    virtual ~AnnSearcher() = default;
    [[nodiscard]] virtual bool RequiresIndex() const = 0;
    [[nodiscard]] virtual std::unique_ptr<AnnSearcher> Clone() const = 0;
    virtual void Init(const PointSetMetadata& metadata, float norm_order) = 0;
    virtual AnnResult Search(const Point& query_point) = 0;
};

class LinearScanAnnSearcher : public AnnSearcher {
   public:
    LinearScanAnnSearcher() = default;
    [[nodiscard]] bool RequiresIndex() const override;
    [[nodiscard]] std::unique_ptr<AnnSearcher> Clone() const override;
    void Init(const PointSetMetadata& metadata, float norm_order) override;
    void InitLoaded(std::shared_ptr<const std::vector<Point>> base_points, unsigned int num_dimensions,
                    float norm_order);
    AnnResult Search(const Point& query_point) override;
    AnnResult SearchWithUpperBound(const Point& query_point, Distance initial_upper_bound);
    std::vector<AnnResult> SearchBatchWithUpperBounds(const std::vector<Point>& query_points,
                                                      const std::vector<Distance>& initial_upper_bounds);
    // True when exact batch searches go through ExactScan: L1 and L2, unless --scalar-exact-scan is set.
    [[nodiscard]] bool UsesBatchedScan() const;
    // Same results as SearchBatchWithUpperBounds for all queries at once, through ExactScan. It runs in parallel over
    // target blocks, so call it outside parallel regions.
    [[nodiscard]] std::vector<AnnResult> SearchBatched(const std::vector<Point>& query_points,
                                                       const std::vector<Distance>& initial_upper_bounds) const;

   private:
    float norm_order_{0.0F};
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    std::shared_ptr<const std::vector<Point>> base_points_;
};

struct QalshAnnIndexData {
    float norm_order{0.0F};
    unsigned int num_points{0};
    unsigned int num_dimensions{0};
    QalshConfig qalsh_config;
    std::shared_ptr<const std::vector<Point>> base_points;
    std::shared_ptr<const qalsh::ProjectionIndex> index;
};

class QalshAnnSearcher : public AnnSearcher {
   public:
    explicit QalshAnnSearcher(float approximation_ratio = Global::kDefaultApproximationRatio);
    QalshAnnSearcher(float approximation_ratio, std::mt19937::result_type random_seed);
    [[nodiscard]] bool RequiresIndex() const override;
    [[nodiscard]] std::unique_ptr<AnnSearcher> Clone() const override;
    [[nodiscard]] std::shared_ptr<const std::vector<Point>> TargetPoints() const;
    void Init(const PointSetMetadata& metadata, float norm_order) override;
    AnnResult Search(const Point& query_point) override;

   private:
    float approximation_ratio_{Global::kDefaultApproximationRatio};
    std::mt19937::result_type random_seed_{Global::kDefaultSeed};
    std::shared_ptr<const QalshAnnIndexData> index_data_;
    std::unique_ptr<qalsh::SearchEngine> engine_;
};

// The stopping ratio gamma of the continuous rule: --qalsh-continuous-ratio if set, else c^1.5.
float QalshContinuousRatio(float approximation_ratio);

// The factor gamma of Theorem 2 for the stopping rule that the in-memory QALSH search uses: the search returns a
// candidate only when its distance is at most gamma times a radius up to which every projection line is processed.
double QalshStoppingFactor(float approximation_ratio);

// The inputs of the certified bound on kappa for a QALSH configuration under the current stopping rule.
kappa::Parameters MakeKappaParameters(const QalshConfig& config, float norm_order);
