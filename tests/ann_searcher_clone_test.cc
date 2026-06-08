#include "ann_searcher.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "b_plus_tree.h"
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
    const std::filesystem::path temp_dir = std::filesystem::current_path() / "ann_searcher_clone_test_tmp";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir / "index" / "l2" / "base" / "b_plus_trees");

    std::vector<Point> base_points(150, Point{1000.0});
    for (unsigned int point_id = 0; point_id < 149; ++point_id) {
        base_points[point_id][0] = 1000.0 + static_cast<double>(point_id);
    }
    base_points[149][0] = 1.5;
    WritePointFile(temp_dir / "base.bin", base_points);

    QalshConfig config{
        .approximation_ratio = 2.0,
        .bucket_width = 1000.0,
        .error_probability = 0.1,
        .num_hash_tables = 1,
        .collision_threshold = 1,
        .page_size = 128,
    };
    config.Save(temp_dir / "index" / "l2" / "base" / "config.json");

    const std::vector<Point> dot_vectors = {Point{1.0}};
    WritePointFile(temp_dir / "index" / "l2" / "base" / "dot_vectors.bin", dot_vectors);

    std::vector<DotProductPointIdPair> data;
    data.reserve(base_points.size());
    for (unsigned int point_id = 0; point_id < base_points.size(); ++point_id) {
        data.push_back(DotProductPointIdPair{
            .dot_product = static_cast<double>(point_id + 1),
            .point_id = point_id,
        });
    }
    {
        BPlusTreeBulkLoader loader(temp_dir / "index" / "l2" / "base" / "b_plus_trees" / "0.bin", config.page_size);
        loader.Build(data);
    }

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
    if (const AnnResult result = linear_clone->Search(query_point); result.point_id != 149 || result.distance != 1.5) {
        std::filesystem::remove_all(temp_dir);
        return 1;
    }

    QalshAnnSearcher qalsh;
    qalsh.Init(metadata, 2.0);
    std::unique_ptr<AnnSearcher> qalsh_clone = qalsh.Clone();
    qalsh_clone->Init(metadata, 2.0);
    if (const AnnResult result = qalsh_clone->Search(query_point); result.point_id != 149 || result.distance != 1.5) {
        std::filesystem::remove_all(temp_dir);
        return 1;
    }

    std::filesystem::remove_all(temp_dir);
    return 0;
}
