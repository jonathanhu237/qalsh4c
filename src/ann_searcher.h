#pragma once

#include <memory>
#include <vector>

#include "b_plus_tree.h"
#include "config.h"
#include "point_set.h"

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

   private:
    PointSet base_;
    double norm_order_{0.0};
};

// ---------------------------------------------
// QalshAnnSearcher Definition
// ---------------------------------------------
class QalshAnnSearcher : public AnnSearcher {
   public:
    QalshAnnSearcher() = default;
    [[nodiscard]] bool RequiresIndex() const override;
    [[nodiscard]] std::unique_ptr<AnnSearcher> Clone() const override;
    void Init(const PointSetMetadata& metadata, double norm_order) override;
    AnnResult Search(const Point& query_point) override;

   private:
    PointSet base_;
    double norm_order_{0.0};
    QalshConfig qalsh_config_;
    std::vector<Point> dot_vectors_;
    std::vector<BPlusTreeSearcher> searchers_;
};
