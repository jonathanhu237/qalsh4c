#include "weights_generator.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

void WritePointFile(const std::filesystem::path& path, const std::vector<Point>& points) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    for (const Point& point : points) {
        output.write(reinterpret_cast<const char*>(point.data()),
                     static_cast<std::streamsize>(point.size() * sizeof(Coordinate)));
    }
}

void WriteWeights(const std::filesystem::path& path, const std::vector<Weight>& weights) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(weights.data()),
                 static_cast<std::streamsize>(weights.size() * sizeof(Weight)));
}

bool NearlyEqual(Weight lhs, Weight rhs) {
    return std::abs(lhs - rhs) < 1e-5F;
}

bool Equals(const std::vector<Weight>& actual, const std::vector<Weight>& expected) {
    if (actual.size() != expected.size()) {
        return false;
    }
    for (size_t index = 0; index < actual.size(); ++index) {
        if (!NearlyEqual(actual[index], expected[index])) {
            return false;
        }
    }
    return true;
}

bool AllPositiveFinite(const std::vector<Weight>& weights) {
    for (Weight weight : weights) {
        if (!std::isfinite(weight) || weight <= 0.0F) {
            return false;
        }
    }
    return true;
}

PointSetMetadata Metadata(const std::filesystem::path& path, unsigned int num_points, unsigned int num_dimensions) {
    return PointSetMetadata{
        .file_path = path,
        .num_points = num_points,
        .num_dimensions = num_dimensions,
    };
}

}  // namespace

int main() {
    const std::filesystem::path temp_dir = std::filesystem::current_path() / "quadtree_weights_generator_test_tmp";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    const std::vector<Point> points_a{
        {0.0F, 0.0F},
        {10.0F, 0.0F},
        {4.0F, 0.0F},
    };
    const std::vector<Point> points_b{
        {3.0F, 4.0F},
        {8.0F, 0.0F},
    };
    WritePointFile(temp_dir / "A.bin", points_a);
    WritePointFile(temp_dir / "B.bin", points_b);

    const PointSetMetadata metadata_a =
        Metadata(temp_dir / "A.bin", static_cast<unsigned int>(points_a.size()), 2);
    const PointSetMetadata metadata_b =
        Metadata(temp_dir / "B.bin", static_cast<unsigned int>(points_b.size()), 2);

    QuadtreeWeightsGenerator generator(2);
    if (generator.RequiresIndex()) {
        std::cerr << "quadtree weights should not require an index\n";
        return 1;
    }

    const std::vector<Weight> l1_weights = generator.Generate(metadata_a, metadata_b, 1.0F, false);
    const std::vector<Weight> expected_l1{8.0F, 2.0F, 4.0F};
    if (!Equals(l1_weights, expected_l1)) {
        std::cerr << "unexpected L1 weights with stack-reversed DFS\n";
        return 2;
    }

    const std::vector<Weight> reverse_l1_weights = generator.Generate(metadata_b, metadata_a, 1.0F, false);
    const std::vector<Weight> expected_reverse_l1{11.0F, 2.0F};
    if (!Equals(reverse_l1_weights, expected_reverse_l1)) {
        std::cerr << "unexpected reverse-direction L1 weights\n";
        return 3;
    }

    const std::vector<Point> l2_points_a{{4.0F, 0.0F}};
    const std::vector<Point> l2_points_b{{0.0F, 3.0F}, {8.0F, 0.0F}};
    WritePointFile(temp_dir / "C.bin", l2_points_a);
    WritePointFile(temp_dir / "D.bin", l2_points_b);
    const PointSetMetadata metadata_c = Metadata(temp_dir / "C.bin", 1, 2);
    const PointSetMetadata metadata_d = Metadata(temp_dir / "D.bin", 2, 2);

    QuadtreeWeightsGenerator shallow_generator(1);
    const std::vector<Weight> l2_weights = shallow_generator.Generate(metadata_c, metadata_d, 2.0F, false);
    if (!Equals(l2_weights, std::vector<Weight>{5.0F})) {
        std::cerr << "unexpected L2 weight\n";
        return 4;
    }

    const std::vector<Point> partial_zero_a{{0.0F, 0.0F}, {2.0F, 0.0F}};
    const std::vector<Point> partial_zero_b{{0.0F, 0.0F}};
    WritePointFile(temp_dir / "E.bin", partial_zero_a);
    WritePointFile(temp_dir / "F.bin", partial_zero_b);
    const PointSetMetadata metadata_e = Metadata(temp_dir / "E.bin", 2, 2);
    const PointSetMetadata metadata_f = Metadata(temp_dir / "F.bin", 1, 2);
    const std::vector<Weight> partial_zero_weights =
        shallow_generator.Generate(metadata_e, metadata_f, 1.0F, false);
    if (partial_zero_weights.size() != 2 || !AllPositiveFinite(partial_zero_weights) ||
        partial_zero_weights[0] >= 1e-6F || !NearlyEqual(partial_zero_weights[1], 2.0F)) {
        std::cerr << "partial zero weights should be floored without changing positive weights\n";
        return 5;
    }

    const std::vector<Point> degenerate_a{{0.0F, 0.0F}, {0.0F, 0.0F}};
    const std::vector<Point> degenerate_b{{0.0F, 0.0F}};
    WritePointFile(temp_dir / "G.bin", degenerate_a);
    WritePointFile(temp_dir / "H.bin", degenerate_b);
    const PointSetMetadata metadata_g = Metadata(temp_dir / "G.bin", 2, 2);
    const PointSetMetadata metadata_h = Metadata(temp_dir / "H.bin", 1, 2);
    const std::vector<Weight> degenerate_weights = generator.Generate(metadata_g, metadata_h, 1.0F, false);
    if (!Equals(degenerate_weights, std::vector<Weight>{1.0F, 1.0F})) {
        std::cerr << "all-degenerate weights should fall back to uniform weights\n";
        return 6;
    }

    const std::vector<Weight> cached_weights = generator.Generate(metadata_a, metadata_b, 1.0F, true);
    const std::filesystem::path cache_path = temp_dir / "l1_quadtree_l2_weights_A.bin";
    if (cached_weights.size() != points_a.size() || !std::filesystem::exists(cache_path)) {
        std::cerr << "missing quadtree cache file\n";
        return 7;
    }

    const std::vector<Weight> sentinel_weights{7.0F, 11.0F, 13.0F};
    WriteWeights(cache_path, sentinel_weights);
    const std::vector<Weight> cache_hit_weights = generator.Generate(metadata_a, metadata_b, 1.0F, true);
    if (!Equals(cache_hit_weights, sentinel_weights)) {
        std::cerr << "quadtree cache hit did not read cached weights\n";
        return 8;
    }

    (void)generator.Generate(metadata_a, metadata_b, 2.0F, true);
    if (!std::filesystem::exists(temp_dir / "l2_quadtree_l2_weights_A.bin")) {
        std::cerr << "missing norm-specific quadtree cache file\n";
        return 9;
    }

    QuadtreeWeightsGenerator deeper_generator(3);
    (void)deeper_generator.Generate(metadata_a, metadata_b, 1.0F, true);
    if (!std::filesystem::exists(temp_dir / "l1_quadtree_l3_weights_A.bin")) {
        std::cerr << "missing max-level-specific quadtree cache file\n";
        return 10;
    }

    std::filesystem::remove_all(temp_dir);
    return 0;
}
