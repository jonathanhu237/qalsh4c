#include "estimator.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "ann_searcher.h"

namespace {

void WritePointFile(const std::filesystem::path& path, const std::vector<Point>& points) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    for (const Point& point : points) {
        output.write(reinterpret_cast<const char*>(point.data()),
                     static_cast<std::streamsize>(point.size() * sizeof(Coordinate)));
    }
}

bool AlmostEqual(double lhs, double rhs) {
    return std::abs(lhs - rhs) < 1e-9;
}

}  // namespace

int main() {
    const std::filesystem::path temp_dir = std::filesystem::current_path() / "disk_linear_scan_test_tmp";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    const std::vector<Point> queries{{0.0, 0.0}, {5.0, 0.0}, {10.0, 0.0}};
    const std::vector<Point> targets{{1.0, 0.0}, {6.0, 0.0}, {11.0, 0.0}, {100.0, 100.0}};
    WritePointFile(temp_dir / "queries.bin", queries);
    WritePointFile(temp_dir / "targets.bin", targets);

    const PointSetMetadata query_metadata{
        .file_path = temp_dir / "queries.bin",
        .num_points = static_cast<unsigned int>(queries.size()),
        .num_dimensions = 2,
    };
    const PointSetMetadata target_metadata{
        .file_path = temp_dir / "targets.bin",
        .num_points = static_cast<unsigned int>(targets.size()),
        .num_dimensions = 2,
    };

    AnnEstimator disk_estimator(std::make_unique<LinearScanAnnSearcher>(), true, 2, 2);
    const double disk_estimate = disk_estimator.EstimateDistance(query_metadata, target_metadata, 2.0);
    if (!AlmostEqual(disk_estimate, 3.0)) {
        std::filesystem::remove_all(temp_dir);
        return 1;
    }

    AnnEstimator memory_estimator(std::make_unique<LinearScanAnnSearcher>());
    const double memory_estimate = memory_estimator.EstimateDistance(query_metadata, target_metadata, 2.0);
    if (!AlmostEqual(disk_estimate, memory_estimate)) {
        std::filesystem::remove_all(temp_dir);
        return 2;
    }

    DiskBasedLinearScanAnnSearcher disk_searcher(2);
    disk_searcher.Init(target_metadata, 2.0);
    const AnnResult tie_result = disk_searcher.Search(Point{0.0, 0.0});
    if (tie_result.point_id != 0 || !AlmostEqual(tie_result.distance, 1.0)) {
        std::filesystem::remove_all(temp_dir);
        return 3;
    }

    std::filesystem::remove_all(temp_dir);
    return 0;
}
