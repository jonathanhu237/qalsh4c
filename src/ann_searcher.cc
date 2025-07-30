#include "ann_searcher.h"

#include <spdlog/spdlog.h>

#include <memory>

#include "point_set.h"
#include "utils.h"

// ---------------------------------------------
// LinearScanAnnSearcher Implementation
// ---------------------------------------------
void LinearScanAnnSearcher::Init(PointSetMetadata point_set_metadata, bool in_memory) {
    if (in_memory) {
        base_set_ = std::make_unique<InMemoryPointSet>(point_set_metadata);
    } else {
        base_set_ = std::make_unique<DiskPointSet>(point_set_metadata);
    }
}

AnnResult LinearScanAnnSearcher::Search(const Point& query_point) {
    AnnResult result{.point_id = 0, .distance = std::numeric_limits<double>::max()};

    unsigned int base_num_points = base_set_->get_num_points();

    for (unsigned int i = 0; i < base_num_points; i++) {
        Point base_point = base_set_->GetPoint(i);
        double distance = Utils::L1Distance(base_point, query_point);
        if (distance < result.distance) {
            result.point_id = i;
            result.distance = distance;
        }
    }

    return result;
}

void LinearScanAnnSearcher::Reset() { base_set_.reset(); }