#include "dataset.h"

#include <cmath>
#include <filesystem>
#include <vector>

bool Near(float actual, float expected) {
    return std::abs(actual - expected) < 1e-9;
}

int main() {
    std::vector<DatasetPrep::GroundTruthTask> tasks = DatasetPrep::ParseGroundTruthTasks(" l2:ab,l1:ba,l2:ab ");
    if (tasks.size() != 2 || !Near(tasks[0].norm_order, 2.0) || tasks[0].direction != ChamferDirection::kAb ||
        !Near(tasks[1].norm_order, 1.0) || tasks[1].direction != ChamferDirection::kBa) {
        return 1;
    }

    const std::vector<Point> a = {Point{0.0, 0.0}, Point{2.0, 0.0}};
    const std::vector<Point> b = {Point{1.0, 0.0}, Point{4.0, 0.0}};
    if (!Near(DatasetPrep::ComputeDirectedChamfer(a, b, 2.0, "test l2:ab", false), 2.0)) {
        return 2;
    }
    if (!Near(DatasetPrep::ComputeDirectedChamfer(b, a, 2.0, "test l2:ba", false), 3.0)) {
        return 3;
    }
    if (!Near(DatasetPrep::ComputeDirectedChamfer(a, b, 1.0, "test l1:ab", false), 2.0)) {
        return 4;
    }

    const std::filesystem::path temp_dir = std::filesystem::current_path() / "dataset_core_test_tmp";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    DatasetPrep::WritePointSet(temp_dir / "A.bin", a);
    std::vector<Point> loaded = PointSet::LoadPointsFromFile(temp_dir / "A.bin", 2, 2);
    if (loaded != a) {
        return 5;
    }

    DatasetMetadata metadata;
    metadata.num_dimensions = 2;
    metadata.num_points_a = 2;
    metadata.num_points_b = 2;
    metadata.chamfer_distance_l2_ab = 2.0;
    metadata.Save(temp_dir / "metadata.json");

    DatasetMetadata loaded_metadata;
    loaded_metadata.Load(temp_dir / "metadata.json");
    if (!loaded_metadata.chamfer_distance_l2_ab.has_value() || !Near(*loaded_metadata.chamfer_distance_l2_ab, 2.0)) {
        return 6;
    }
    if (loaded_metadata.chamfer_distance_l1_ab.has_value() || loaded_metadata.GroundTruth(1.0, ChamferDirection::kAb).has_value()) {
        return 7;
    }
    if (loaded_metadata.GroundTruth(2.0, ChamferDirection::kBoth).has_value()) {
        return 8;
    }

    std::filesystem::remove_all(temp_dir);
    return 0;
}
