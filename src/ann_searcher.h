#pragma once

#include <memory>
#include <random>
#include <vector>

#include "config.h"
#include "global.h"
#include "point_set.h"
#include "qalsh_index.h"

struct AnnResult {
    double distance{0.0};
    unsigned int point_id{0};
};

struct CompareAnnResult {
    bool operator()(const AnnResult& a, const AnnResult& b) const { return a.distance > b.distance; }
};

// ---------------------------------------------
// AnnSearcher Definition
// ---------------------------------------------
class AnnSearcher {
   public:
    virtual ~AnnSearcher() = default;
    [[nodiscard]] virtual bool RequiresIndex() const = 0;
    [[nodiscard]] virtual std::unique_ptr<AnnSearcher> Clone() const = 0;
    virtual void Init(const PointSetMetadata& metadata, double norm_order) = 0;
    virtual AnnResult Search(const Point& query_point) = 0;
};

// ---------------------------------------------
// LinearScanAnnSearcher Definition
// ---------------------------------------------
class LinearScanAnnSearcher : public AnnSearcher {
   public:
    LinearScanAnnSearcher() = default;
    [[nodiscard]] bool RequiresIndex() const override;
    [[nodiscard]] std::unique_ptr<AnnSearcher> Clone() const override;
    void Init(const PointSetMetadata& metadata, double norm_order) override;
    AnnResult Search(const Point& query_point) override;
    AnnResult SearchWithUpperBound(const Point& query_point, double initial_upper_bound);
    std::vector<AnnResult> SearchBatchWithUpperBounds(const std::vector<Point>& query_points,
                                                      const std::vector<double>& initial_upper_bounds);

   private:
    double norm_order_{0.0};
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    std::vector<Point> base_points_;
};

// ---------------------------------------------
// QalshAnnSearcher Definition
// ---------------------------------------------
class QalshAnnSearcher : public AnnSearcher {
   public:
    explicit QalshAnnSearcher(double approximation_ratio = Global::kDefaultApproximationRatio);
    QalshAnnSearcher(double approximation_ratio, std::mt19937::result_type random_seed);
    [[nodiscard]] bool RequiresIndex() const override;
    [[nodiscard]] std::unique_ptr<AnnSearcher> Clone() const override;
    void Init(const PointSetMetadata& metadata, double norm_order) override;
    AnnResult Search(const Point& query_point) override;

   private:
    double approximation_ratio_{Global::kDefaultApproximationRatio};
    std::mt19937::result_type random_seed_{Global::kDefaultSeed};
    double norm_order_{0.0};
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    QalshConfig qalsh_config_;
    std::vector<Point> base_points_;
    std::vector<Point> dot_vectors_;
    std::vector<InMemoryQalshTable> searchers_;
    std::vector<unsigned int> collision_counts_;
    std::vector<unsigned int> collision_stamps_;
    std::vector<unsigned int> visited_stamps_;
    unsigned int search_epoch_{0};

    unsigned int NextSearchEpoch();
};
