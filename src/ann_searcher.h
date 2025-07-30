#ifndef ANN_SEARCHER_H_
#define ANN_SEARCHER_H_

#include <memory>

#include "point_set.h"

struct AnnResult {
    unsigned int point_id{0};
    double distance{0.0};

    bool operator<(const AnnResult& rhs) const { return distance < rhs.distance; }
};

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class AnnSearcher {
   public:
    virtual ~AnnSearcher() = default;

    virtual void Init(PointSetMetadata point_set_metadata) = 0;
    virtual AnnResult Search(const Point& query_point) = 0;
    virtual void Reset() = 0;
};

class LinearScanAnnSearcher : public AnnSearcher {
   public:
    LinearScanAnnSearcher(bool in_memory);

    void Init(PointSetMetadata point_set_metadata) override;
    AnnResult Search(const Point& query_point) override;
    void Reset() override;

   private:
    bool in_memory_;
    std::unique_ptr<PointSet> base_set_;
};

#endif