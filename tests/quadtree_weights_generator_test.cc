#include "weights_generator.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

void WritePointFile(const std::filesystem::path& path, const std::vector<Point>& points) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    for (const Point& point : points) {
        ofs.write(reinterpret_cast<const char*>(point.data()),
                  static_cast<std::streamsize>(point.size() * sizeof(Coordinate)));
    }
}

bool AllPositiveFinite(const std::vector<double>& weights) {
    for (double weight : weights) {
        if (!std::isfinite(weight) || weight <= 0.0) {
            return false;
        }
    }
    return true;
}

bool Equals(const std::vector<double>& actual, const std::vector<double>& expected) {
    if (actual.size() != expected.size()) {
        return false;
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != expected[index]) {
            return false;
        }
    }
    return true;
}

void WriteWeights(const std::filesystem::path& path, const std::vector<double>& weights) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    ofs.write(reinterpret_cast<const char*>(weights.data()),
              static_cast<std::streamsize>(weights.size() * sizeof(double)));
}

}  // namespace

int main() {
    const std::filesystem::path temp_dir = std::filesystem::current_path() / "quadtree_weights_generator_test_tmp";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    const std::vector<Point> points_a{
        {0.0, 0.0},
        {10.0, 0.0},
        {0.0, 10.0},
        {5.0, 5.0},
    };
    const std::vector<Point> points_b{
        {1.0, 0.0},
        {9.0, 0.0},
        {0.0, 9.0},
    };
    WritePointFile(temp_dir / "A.bin", points_a);
    WritePointFile(temp_dir / "B.bin", points_b);

    const PointSetMetadata metadata_a{
        .file_path = temp_dir / "A.bin",
        .num_points = static_cast<unsigned int>(points_a.size()),
        .num_dimensions = 2,
    };
    const PointSetMetadata metadata_b{
        .file_path = temp_dir / "B.bin",
        .num_points = static_cast<unsigned int>(points_b.size()),
        .num_dimensions = 2,
    };

    QuadtreeWeightsGenerator generator;
    if (generator.RequiresIndex()) {
        std::cerr << "quadtree weights should not require an index\n";
        return 1;
    }

    const std::vector<double> l1_weights = generator.Generate(metadata_a, metadata_b, 1.0, false);
    if (l1_weights.size() != points_a.size() || !AllPositiveFinite(l1_weights)) {
        std::cerr << "invalid L1 A->B weights\n";
        return 1;
    }

    const std::vector<double> l2_weights = generator.Generate(metadata_a, metadata_b, 2.0, false);
    if (l2_weights.size() != points_a.size() || !AllPositiveFinite(l2_weights)) {
        std::cerr << "invalid L2 A->B weights\n";
        return 1;
    }

    const std::vector<double> reverse_weights = generator.Generate(metadata_b, metadata_a, 1.0, false);
    if (reverse_weights.size() != points_b.size() || !AllPositiveFinite(reverse_weights)) {
        std::cerr << "invalid B->A weights\n";
        return 1;
    }

    QuadtreeWeightsGenerator shallow_generator(1);
    const std::vector<double> shallow_weights = shallow_generator.Generate(metadata_a, metadata_b, 1.0, false);
    if (shallow_weights.size() != points_a.size() || !AllPositiveFinite(shallow_weights)) {
        std::cerr << "invalid max-levels=1 weights\n";
        return 1;
    }

    const std::vector<Point> degenerate_a{{0.0, 0.0}, {0.0, 0.0}};
    const std::vector<Point> degenerate_b{{0.0, 0.0}};
    WritePointFile(temp_dir / "C.bin", degenerate_a);
    WritePointFile(temp_dir / "D.bin", degenerate_b);
    const PointSetMetadata metadata_c{
        .file_path = temp_dir / "C.bin",
        .num_points = static_cast<unsigned int>(degenerate_a.size()),
        .num_dimensions = 2,
    };
    const PointSetMetadata metadata_d{
        .file_path = temp_dir / "D.bin",
        .num_points = static_cast<unsigned int>(degenerate_b.size()),
        .num_dimensions = 2,
    };

    const std::vector<double> degenerate_weights = generator.Generate(metadata_c, metadata_d, 1.0, false);
    if (!Equals(degenerate_weights, std::vector<double>{1.0, 1.0})) {
        std::cerr << "degenerate weights should fall back to uniform weights\n";
        return 1;
    }

    const std::vector<double> cached_weights = generator.Generate(metadata_a, metadata_b, 1.0, true);
    const std::filesystem::path l1_cache_path = temp_dir / "l1_quadtree_l3_weights_A.bin";
    if (cached_weights.size() != points_a.size() || !std::filesystem::exists(l1_cache_path)) {
        std::cerr << "missing L1 quadtree cache file\n";
        return 1;
    }

    const std::vector<double> sentinel_weights{7.0, 11.0, 13.0, 17.0};
    WriteWeights(l1_cache_path, sentinel_weights);
    const std::vector<double> cache_hit_weights = generator.Generate(metadata_a, metadata_b, 1.0, true);
    if (!Equals(cache_hit_weights, sentinel_weights)) {
        std::cerr << "quadtree cache hit did not read cached weights\n";
        return 1;
    }

    (void)generator.Generate(metadata_a, metadata_b, 2.0, true);
    if (!std::filesystem::exists(temp_dir / "l2_quadtree_l3_weights_A.bin")) {
        std::cerr << "missing L2 quadtree cache file\n";
        return 1;
    }

    std::filesystem::remove_all(temp_dir);
    return 0;
}
