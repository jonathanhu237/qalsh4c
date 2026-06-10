#include "ann_searcher.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "global.h"

namespace {
void WritePointFile(const std::filesystem::path& path, const std::vector<Point>& points) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    for (const Point& point : points) {
        ofs.write(reinterpret_cast<const char*>(point.data()),
                  static_cast<std::streamsize>(point.size() * sizeof(Coordinate)));
    }
}
}  // namespace

int main() {
    Global::kUseFixedSeed = true;

    const std::filesystem::path temp_dir = std::filesystem::current_path() / "ann_searcher_clone_test_tmp";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    std::vector<Point> base_points;
    base_points.reserve(150);
    base_points.push_back(Point{0.0});
    for (unsigned int point_id = 1; point_id < 150; ++point_id) {
        base_points.push_back(Point{1000.0 + static_cast<double>(point_id)});
    }
    WritePointFile(temp_dir / "base.bin", base_points);

    const PointSetMetadata metadata{
        .file_path = temp_dir / "base.bin",
        .num_points = static_cast<unsigned int>(base_points.size()),
        .num_dimensions = 1,
    };
    const Point query_point{0.0};

    LinearScanAnnSearcher linear;
    linear.Init(metadata, 2.0);
    std::unique_ptr<AnnSearcher> linear_clone = linear.Clone();
    linear_clone->Init(metadata, 2.0);
    if (const AnnResult result = linear_clone->Search(query_point); result.point_id != 0 || result.distance != 0.0) {
        std::filesystem::remove_all(temp_dir);
        return 1;
    }

    QalshAnnSearcher qalsh(2.0);
    qalsh.Init(metadata, 2.0);
    std::unique_ptr<AnnSearcher> qalsh_clone = qalsh.Clone();
    qalsh_clone->Init(metadata, 2.0);
    if (const AnnResult result = qalsh_clone->Search(query_point); result.point_id != 0 || result.distance != 0.0) {
        std::filesystem::remove_all(temp_dir);
        return 2;
    }

    if (std::filesystem::exists(temp_dir / "index")) {
        std::filesystem::remove_all(temp_dir);
        return 3;
    }

    std::filesystem::remove_all(temp_dir);
    return 0;
}
