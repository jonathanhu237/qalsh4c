#include "ann_searcher.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <vector>

#include "config.h"
#include "disk_qalsh_index.h"
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
        base_points.push_back(Point{1000.0F + static_cast<float>(point_id)});
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
    std::filesystem::remove(temp_dir / "base.bin");
    if (const AnnResult result = linear.Search(query_point); result.point_id != 0 || result.distance != 0.0) {
        std::filesystem::remove_all(temp_dir);
        return 1;
    }
    WritePointFile(temp_dir / "base.bin", base_points);

    std::unique_ptr<AnnSearcher> linear_clone = linear.Clone();
    linear_clone->Init(metadata, 2.0);
    if (const AnnResult result = linear_clone->Search(query_point); result.point_id != 0 || result.distance != 0.0) {
        std::filesystem::remove_all(temp_dir);
        return 2;
    }

    QalshAnnSearcher qalsh(2.0);
    qalsh.Init(metadata, 2.0);
    std::unique_ptr<AnnSearcher> qalsh_clone = qalsh.Clone();
    qalsh_clone->Init(metadata, 2.0);
    if (const AnnResult result = qalsh_clone->Search(query_point); result.point_id != 0 || result.distance != 0.0) {
        std::filesystem::remove_all(temp_dir);
        return 3;
    }

    if (std::filesystem::exists(temp_dir / "index")) {
        std::filesystem::remove_all(temp_dir);
        return 4;
    }

    const std::vector<Point> query_points{
        Point{0.0F, 0.0F}, Point{1.0F, 1.0F}, Point{5.0F, -1.0F}, Point{1003.0F, 1002.0F},
    };
    std::vector<Point> disk_base_points;
    disk_base_points.reserve(180);
    for (unsigned int point_id = 0; point_id < 180; ++point_id) {
        disk_base_points.push_back(Point{static_cast<float>(point_id), static_cast<float>(point_id % 7U)});
    }
    WritePointFile(temp_dir / "A.bin", query_points);
    WritePointFile(temp_dir / "B.bin", disk_base_points);
    DatasetMetadata dataset_metadata{
        .num_points_a = static_cast<unsigned int>(query_points.size()),
        .num_points_b = static_cast<unsigned int>(disk_base_points.size()),
        .num_dimensions = 2,
    };
    dataset_metadata.Save(temp_dir / "metadata.json");

    DiskQalsh::BuildIndexes(DiskQalsh::BuildRequest{.dataset_directory = temp_dir,
                                                    .norm_order = 2.0F,
                                                    .target_selection = DiskQalsh::TargetSelection::kB,
                                                    .seed = 42,
                                                    .approximation_ratio = 2.0F,
                                                    .overwrite = true});

    const PointSetMetadata disk_target_metadata{
        .file_path = temp_dir / "B.bin",
        .num_points = static_cast<unsigned int>(disk_base_points.size()),
        .num_dimensions = 2,
    };
    DiskBasedQalshAnnSearcher disk_searcher(std::optional<float>{2.0F});
    disk_searcher.Init(disk_target_metadata, 2.0F);

    std::vector<AnnResult> single_results;
    single_results.reserve(query_points.size());
    for (const Point& query : query_points) {
        single_results.push_back(disk_searcher.Search(query));
    }

    std::vector<Coordinate> flat_queries;
    flat_queries.reserve(query_points.size() * dataset_metadata.num_dimensions);
    for (const Point& query : query_points) {
        flat_queries.insert(flat_queries.end(), query.begin(), query.end());
    }

    std::vector<AnnResult> batch_results;
    disk_searcher.SearchBatchContiguous(flat_queries.data(), static_cast<unsigned int>(query_points.size()),
                                        dataset_metadata.num_dimensions, batch_results);
    if (batch_results.size() != single_results.size()) {
        std::filesystem::remove_all(temp_dir);
        return 5;
    }
    for (size_t i = 0; i < single_results.size(); ++i) {
        if (batch_results[i].point_id != single_results[i].point_id ||
            batch_results[i].distance != single_results[i].distance) {
            std::filesystem::remove_all(temp_dir);
            return 6;
        }
    }

    std::filesystem::remove_all(temp_dir);
    return 0;
}
