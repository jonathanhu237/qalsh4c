#include "ann_searcher.h"

#include <variant>

#include "point_set.h"
#include "types.h"
#include "utils.h"

// ---------------------------------------------
// LinearScanAnnSearcher Implementation
// ---------------------------------------------

LinearScanAnnSearcher::LinearScanAnnSearcher(PointSetReader* base_reader) : base_reader_(base_reader) {}

AnnResult LinearScanAnnSearcher::Search(const PointVariant& query_point) {
    AnnResult result{.point_id = 0, .distance = std::numeric_limits<double>::max()};

    unsigned int base_num_points = base_reader_->get_num_points();
    for (unsigned int i = 0; i < base_num_points; i++) {
        PointVariant base_point = base_reader_->GetPoint(i);
        double distance = 0.0;

        std::visit(
            [&](const auto& concrete_base_point, const auto& concrete_query_point) {
                distance = Utils::CalculateL1Distance(concrete_base_point, concrete_query_point);
            },
            base_point, query_point);

        if (distance < result.distance) {
            result.point_id = i;
            result.distance = distance;
        }
    }

    return result;
}