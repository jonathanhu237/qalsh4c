#ifndef ANN_SEARCHER_H_
#define ANN_SEARCHER_H_

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
    virtual void Init(const PointSetMetadata& metadata, double norm_order) = 0;
    virtual AnnResult Search(const Point& query_point) = 0;
};

// ---------------------------------------------
// LinearScanAnnSearcher Definition
// ---------------------------------------------
class LinearScanAnnSearcher : public AnnSearcher {
   public:
    LinearScanAnnSearcher() = default;
    void Init(const PointSetMetadata& metadata, double norm_order) override;
    AnnResult Search(const Point& query_point) override;

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
    void Init(const PointSetMetadata& metadata, double norm_order) override;
    AnnResult Search(const Point& query_point) override;

   private:
    PointSet base_;
    double norm_order_{0.0};
    QalshConfig qalsh_config_;
    std::vector<Point> dot_vectors_;
    std::vector<BPlusTreeSearcher> searchers_;
};

#endif