#include "static_c2lsh.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "global.h"
#include "utils.h"

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

// Random bucket ids, negative ones included, drawn around a few cluster centers so that cells are shared at many
// levels.
std::vector<std::int64_t> RandomBucketIds(size_t num_points, unsigned int m, std::int64_t spread, std::mt19937& gen) {
    std::uniform_int_distribution<std::int64_t> center(-spread, spread);
    std::uniform_int_distribution<std::int64_t> jitter(-spread / 64 - 1, spread / 64 + 1);
    std::uniform_int_distribution<int> cluster(0, 3);
    std::vector<std::vector<std::int64_t>> centers(4, std::vector<std::int64_t>(m));
    for (auto& row : centers) {
        for (auto& value : row) {
            value = center(gen);
        }
    }
    std::vector<std::int64_t> ids(num_points * m);
    for (size_t point = 0; point < num_points; ++point) {
        const auto& row = centers[static_cast<size_t>(cluster(gen))];
        for (unsigned int i = 0; i < m; ++i) {
            ids[point * m + i] = row[i] + jitter(gen);
        }
    }
    return ids;
}

void TestDefaultNumProjections() {
    // The sizes of the MNIST-A20k, Enron-A20k, P53 and Quora target sets.
    Check(StaticC2lsh::DefaultNumProjections(49900) == 16, "m for |B| = 49,900");
    Check(StaticC2lsh::DefaultNumProjections(75300) == 17, "m for |B| = 75,300");
    Check(StaticC2lsh::DefaultNumProjections(26100) == 16, "m for |B| = 26,100");
    Check(StaticC2lsh::DefaultNumProjections(522931) == 19, "m for |B| = 522,931");
}

void TestPairLevel() {
    const std::vector<std::int64_t> x{0, 5, -3};
    const std::vector<std::int64_t> y{0, 4, -4};
    // XOR bit lengths are 0, 1 and 3 (-3 ^ -4 = 7): two projections agree from level 1 up, all three from level 3.
    Check(StaticC2lsh::PairLevel(x.data(), y.data(), 3) == 1, "pair level of a hand-made pair");
    Check(StaticC2lsh::PairLevel(x.data(), x.data(), 3) == 0, "pair level of a point with itself");
}

// The first levels must equal a brute-force minimum over every target, and coarsened keys report max(level, dropped).
void TestMortonNeighborsAgainstBruteForce() {
    std::mt19937 gen(7);
    struct Case {
        unsigned int m;
        unsigned int window;
        std::int64_t spread;
    };
    const std::vector<Case> cases{{2, 1, 50},      {3, 1, 1000},    {5, 2, 1 << 12}, {8, 1, 1 << 20},
                                  {17, 3, 1 << 9}, {32, 1, 1LL << 40}, {24, 2, 1LL << 50}};
    for (const Case& test_case : cases) {
        for (int trial = 0; trial < 20; ++trial) {
            const unsigned int num_sources = 37;
            const unsigned int num_targets = 53;
            const std::vector<std::int64_t> ids =
                RandomBucketIds(num_sources + num_targets, test_case.m, test_case.spread, gen);
            const StaticC2lsh::Neighbors neighbors =
                StaticC2lsh::MortonNeighbors(ids, num_sources, num_targets, test_case.m, test_case.window);
            const size_t row_size = 2 * static_cast<size_t>(test_case.window) * test_case.m;
            for (unsigned int source = 0; source < num_sources; ++source) {
                unsigned int brute = std::numeric_limits<unsigned int>::max();
                for (unsigned int target = 0; target < num_targets; ++target) {
                    brute = std::min(brute, StaticC2lsh::PairLevel(ids.data() + source * test_case.m,
                                                                   ids.data() + (num_sources + target) * test_case.m,
                                                                   test_case.m));
                }
                const unsigned int expected = std::max(brute, neighbors.dropped_bits);
                Check(neighbors.first_level[source] == expected,
                      "first level m=" + std::to_string(test_case.m) + " source " + std::to_string(source) + ": got " +
                          std::to_string(neighbors.first_level[source]) + ", expected " + std::to_string(expected));
                for (size_t slot = 0; slot < row_size; ++slot) {
                    const unsigned int target = neighbors.targets[source * row_size + slot];
                    Check(target == StaticC2lsh::kNoTarget || target < num_targets, "neighbor id in range");
                }
            }
            if (test_case.spread >= (1LL << 40)) {
                Check(neighbors.dropped_bits > 0, "wide bucket ids drop low bits");
            }
        }
    }
}

std::vector<Point> GaussianPoints(size_t count, unsigned int d, float scale, std::mt19937& gen) {
    std::normal_distribution<float> normal(0.0F, scale);
    std::vector<Point> points(count, Point(d));
    for (Point& point : points) {
        for (Coordinate& value : point) {
            value = normal(gen);
        }
    }
    return points;
}

Distance NearestDistance(const Point& source, const std::vector<Point>& targets, float norm_order) {
    Distance best = std::numeric_limits<Distance>::infinity();
    for (const Point& target : targets) {
        best = std::min(best, Calc::LpDistance(source, target, norm_order));
    }
    return best;
}

bool IsTargetDistance(double value, const Point& source, const std::vector<Point>& targets, float norm_order) {
    return std::ranges::any_of(targets, [&](const Point& target) {
        return std::abs(static_cast<double>(Calc::LpDistance(source, target, norm_order)) - value) <= 1e-5 * value;
    });
}

// Tightened bounds lie between the nearest-neighbor distance and U_a, are distances to real targets, and keep zero
// bounds at zero. A window as large as B makes every target a neighbor, so the bounds become exact.
void TestUpperBounds() {
    std::mt19937 data_gen(11);
    const unsigned int d = 8;
    const std::vector<Point> sources = GaussianPoints(150, d, 1.0F, data_gen);
    std::vector<Point> targets = GaussianPoints(400, d, 1.0F, data_gen);
    for (const float norm_order : {1.0F, 2.0F}) {
        std::vector<double> upper_bounds(sources.size());
        std::vector<Distance> nearest(sources.size());
        for (size_t source = 0; source < sources.size(); ++source) {
            nearest[source] = NearestDistance(sources[source], targets, norm_order);
            // The distance to a fixed target is a valid, loose upper bound.
            upper_bounds[source] = Calc::LpDistance(sources[source], targets[source % 7], norm_order);
        }
        upper_bounds[3] = 0.0;

        for (const unsigned int repetitions : {1U, 3U}) {
            std::mt19937 gen(5);
            StaticC2lshStats stats;
            const StaticC2lshOptions options{.num_projections = 0, .num_repetitions = repetitions, .window = 2};
            const std::vector<double> tightened =
                StaticC2lshUpperBounds(sources, targets, upper_bounds, norm_order, options, gen, stats);
            Check(tightened[3] == 0.0, "zero bound stays zero");
            Check(stats.num_projections == StaticC2lsh::DefaultNumProjections(targets.size()), "default m");
            Check(stats.candidates_per_source > 0.0, "neighbors were checked");
            for (size_t source = 0; source < sources.size(); ++source) {
                if (source == 3) {
                    continue;
                }
                Check(tightened[source] <= upper_bounds[source], "tightened bound is at most U_a");
                Check(tightened[source] >= static_cast<double>(nearest[source]) * (1.0 - 1e-6),
                      "tightened bound is at least the nearest-neighbor distance");
                Check(IsTargetDistance(tightened[source], sources[source], targets, norm_order),
                      "tightened bound is a distance to a target");
            }
        }

        std::mt19937 gen(9);
        StaticC2lshStats stats;
        const StaticC2lshOptions all{.num_projections = 6,
                                     .num_repetitions = 1,
                                     .window = static_cast<unsigned int>(targets.size())};
        const std::vector<double> exact = StaticC2lshUpperBounds(sources, targets, upper_bounds, norm_order, all, gen, stats);
        for (size_t source = 0; source < sources.size(); ++source) {
            if (source != 3) {
                Check(std::abs(exact[source] - static_cast<double>(nearest[source])) <= 1e-6 * exact[source],
                      "a window as large as B finds the nearest neighbor");
            }
        }
    }
}

void WritePointFile(const std::filesystem::path& path, const std::vector<Point>& points) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    for (const Point& point : points) {
        output.write(reinterpret_cast<const char*>(point.data()),
                     static_cast<std::streamsize>(point.size() * sizeof(Coordinate)));
    }
}

// The disk variant must return exactly the in-memory bounds for the same generator state, with projection chunks that
// straddle the end of A, several verification chunks, and target blocks that hold no neighbor.
void TestDiskMatchesMemory() {
    const std::filesystem::path directory = std::filesystem::temp_directory_path() / "static_c2lsh_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    std::mt19937 data_gen(17);
    // At d = 300 a projection chunk holds 873 points, so 400 sources and 1,500 targets span three chunks.
    const unsigned int d = 300;
    const std::vector<Point> sources = GaussianPoints(400, d, 1.0F, data_gen);
    const std::vector<Point> targets = GaussianPoints(1500, d, 1.0F, data_gen);
    WritePointFile(directory / "A.bin", sources);
    WritePointFile(directory / "B.bin", targets);
    const PointSetMetadata source_metadata{.file_path = directory / "A.bin", .num_points = 400, .num_dimensions = d};
    const PointSetMetadata target_metadata{.file_path = directory / "B.bin", .num_points = 1500, .num_dimensions = d};

    for (const float norm_order : {1.0F, 2.0F}) {
        std::vector<double> upper_bounds(sources.size());
        for (size_t source = 0; source < sources.size(); ++source) {
            upper_bounds[source] = Calc::LpDistance(sources[source], targets[source % 11], norm_order);
        }
        upper_bounds[5] = 0.0;
        // Chunks of 16 sources check about 600 neighbors spread over 150 target blocks of 10 points, so some blocks
        // hold none and are skipped.
        const StaticC2lshOptions options{.num_projections = 0, .num_repetitions = 2, .window = 2, .disk_chunk_size = 16};

        std::mt19937 memory_gen(23);
        StaticC2lshStats memory_stats;
        const std::vector<double> memory =
            StaticC2lshUpperBounds(sources, targets, upper_bounds, norm_order, options, memory_gen, memory_stats);
        std::mt19937 disk_gen(23);
        StaticC2lshStats disk_stats;
        const StaticC2lshDiskOptions disk_options{.block_size = 10, .index = nullptr, .target_block_hook = {}};
        const std::vector<double> disk = StaticC2lshUpperBoundsDisk(
            source_metadata, target_metadata, upper_bounds, norm_order, options, disk_options, disk_gen, disk_stats);
        Check(memory == disk, "disk bounds equal the in-memory bounds for p = " + std::to_string(norm_order));
        Check(memory_stats.candidates_per_source == disk_stats.candidates_per_source,
              "disk and memory check the same neighbors");
        Check(disk_stats.verify_passes == 25, "400 sources in chunks of 16 take 25 passes over B");
        Check(disk_stats.read_mib > 0.0, "the disk variant reports its reads");
    }
    std::filesystem::remove_all(directory);
}

// A persistent index replaces B's coordinates by its projections onto vectors drawn from the index's seed, so a disk
// run on the index must equal an in-memory run that draws its vectors from the same seed. A block hook must see every
// block of B once, in order, in the same pass that verifies the neighbors.
void TestIndexMatchesMemory() {
    const std::filesystem::path directory = std::filesystem::temp_directory_path() / "static_c2lsh_index_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    std::mt19937 data_gen(29);
    const unsigned int d = 40;
    const std::vector<Point> sources = GaussianPoints(300, d, 1.0F, data_gen);
    const std::vector<Point> targets = GaussianPoints(900, d, 1.0F, data_gen);
    WritePointFile(directory / "A.bin", sources);
    WritePointFile(directory / "B.bin", targets);
    const PointSetMetadata source_metadata{.file_path = directory / "A.bin", .num_points = 300, .num_dimensions = d};
    const PointSetMetadata target_metadata{.file_path = directory / "B.bin", .num_points = 900, .num_dimensions = d};

    for (const float norm_order : {1.0F, 2.0F}) {
        StaticC2lsh::BuildIndex(target_metadata, norm_order, 0, 2, 99, 64, true);
        const StaticC2lsh::Index index = StaticC2lsh::OpenIndex(target_metadata, norm_order);
        Check(index.directory ==
                  directory / "index" / "static" / (norm_order == 1.0F ? "l1" : "l2") / "b",
              "the index lives next to its point set");
        Check(index.num_repetitions == 2 && index.seed == 99, "the index records its parameters");
        Check(index.num_projections == StaticC2lsh::DefaultNumProjections(900), "the index uses the default m");

        std::vector<double> upper_bounds(sources.size());
        for (size_t source = 0; source < sources.size(); ++source) {
            upper_bounds[source] = Calc::LpDistance(sources[source], targets[source % 13], norm_order);
        }
        StaticC2lshOptions memory_options{.num_projections = 0, .num_repetitions = 2, .window = 1};
        memory_options.vector_seed = 99;
        std::mt19937 memory_gen(31);
        StaticC2lshStats memory_stats;
        const std::vector<double> memory =
            StaticC2lshUpperBounds(sources, targets, upper_bounds, norm_order, memory_options, memory_gen, memory_stats);

        unsigned int blocks_seen = 0;
        unsigned int next_first = 0;
        bool in_order = true;
        StaticC2lshDiskOptions disk_options{.block_size = 128, .index = &index, .target_block_hook = {}};
        disk_options.target_block_hook = [&](const Coordinate*, unsigned int count, unsigned int first_id) {
            in_order = in_order && first_id == next_first;
            next_first = first_id + count;
            ++blocks_seen;
        };
        const StaticC2lshOptions disk_run{.num_projections = 0, .num_repetitions = 2, .window = 1, .disk_index = true};
        std::mt19937 disk_gen(31);
        StaticC2lshStats disk_stats;
        const std::vector<double> disk = StaticC2lshUpperBoundsDisk(
            source_metadata, target_metadata, upper_bounds, norm_order, disk_run, disk_options, disk_gen, disk_stats);
        Check(memory == disk, "bounds on the index equal the in-memory bounds for p = " + std::to_string(norm_order));
        Check(in_order && next_first == 900 && blocks_seen == 8, "the hook sees the eight blocks of B once, in order");
        Check(disk_stats.verify_passes == 1, "the hook rides along with the only verification pass");
    }
    std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
    TestDefaultNumProjections();
    TestPairLevel();
    TestMortonNeighborsAgainstBruteForce();
    TestUpperBounds();
    TestDiskMatchesMemory();
    TestIndexMatchesMemory();
    if (failures != 0) {
        std::cerr << failures << " checks failed\n";
        return 1;
    }
    std::cout << "static_c2lsh_test passed\n";
    return 0;
}
