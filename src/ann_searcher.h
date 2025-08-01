#ifndef ANN_SEARCHER_H_
#define ANN_SEARCHER_H_

#include <memory>
#include <queue>
#include <vector>

#include "point_set.h"
#include "qalsh_config.h"
#include "qalsh_hash_table.h"

struct AnnResult {
    unsigned int point_id{0};
    double distance{0.0};

    bool operator<(const AnnResult& rhs) const { return distance < rhs.distance; }
};

class AnnSearcher {
   public:
    virtual ~AnnSearcher() = default;

    virtual void Init(PointSetMetadata point_set_metadata, bool in_memory) = 0;
    virtual AnnResult Search(const Point& query_point) = 0;
    virtual void Reset() = 0;
};

class LinearScanAnnSearcher : public AnnSearcher {
   public:
    LinearScanAnnSearcher() = default;

    void Init(PointSetMetadata point_set_metadata, bool in_memory) override;
    AnnResult Search(const Point& query_point) override;
    void Reset() override;

   private:
    std::unique_ptr<PointSet> base_set_;
};

class QalshAnnSearcher : public AnnSearcher {
   public:
    QalshAnnSearcher(double approximation_ratio);

    void Init(PointSetMetadata point_set_metadata, bool in_memory) override;
    AnnResult Search(const Point& query_point) override;
    void Reset() override;

   private:
    [[nodiscard]] bool shouldTerminate(const std::priority_queue<AnnResult, std::vector<AnnResult>>& candidates,
                                       double search_radius) const;

    std::unique_ptr<PointSet> base_set_;

    QalshConfig qalsh_config_;
    std::vector<Point> dot_vectors_;
    std::vector<std::unique_ptr<QalshHashTable>> hash_tables;
};

#endif