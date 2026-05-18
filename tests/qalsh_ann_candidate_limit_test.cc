#include "ann_searcher.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "config.h"

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
    const std::filesystem::path temp_dir = std::filesystem::current_path() / "qalsh_ann_candidate_limit_test_tmp";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    std::vector<Point> base_points(150, Point{1000.0});
    for (unsigned int point_id = 0; point_id < 149; ++point_id) {
        base_points[point_id][0] = 1000.0 + static_cast<double>(point_id);
    }
    base_points[149][0] = 1.5;
    WritePointFile(temp_dir / "base.bin", base_points);

    const PointSetMetadata metadata{
        .file_path = temp_dir / "base.bin",
        .num_points = static_cast<unsigned int>(base_points.size()),
        .num_dimensions = 1,
    };
    std::vector<ProjectionPointIdPair> data;
    data.reserve(base_points.size());
    for (unsigned int point_id = 0; point_id < base_points.size(); ++point_id) {
        data.push_back(ProjectionPointIdPair{
            .dot_product = static_cast<double>(point_id + 1),
            .point_id = point_id,
        });
    }
    auto index = std::make_shared<InMemoryQalshIndex>();
    index->config = QalshConfig{
        .approximation_ratio = 2.0,
        .bucket_width = 1000.0,
        .error_probability = 0.1,
        .num_hash_tables = 1,
        .collision_threshold = 1,
        .page_size = 128,
    };
    index->metadata = metadata;
    index->points = base_points;
    index->dot_vectors = {Point{1.0}};
    index->tables = {std::move(data)};

    const Point query_point{0.0};

    AnnResult unlimited_result;
    AnnResult limited_result;
    {
        QalshAnnSearcher unlimited(index, false);
        unlimited.Init(metadata, 2.0);
        unlimited_result = unlimited.Search(query_point);

        QalshAnnSearcher limited(index, true);
        limited.Init(metadata, 2.0);
        limited_result = limited.Search(query_point);
    }

    std::filesystem::remove_all(temp_dir);

    if (unlimited_result.point_id != 149 || unlimited_result.distance != 1.5) {
        return 1;
    }
    if (limited_result.point_id == unlimited_result.point_id) {
        return 1;
    }
    if (limited_result.distance <= unlimited_result.distance) {
        return 1;
    }

    return 0;
}
