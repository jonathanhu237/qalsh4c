#include "static_c2lsh.h"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numbers>
#include <random>
#include <utility>
#include <vector>

#include "global.h"
#include "log.h"
#include "utils.h"

namespace {

using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

constexpr unsigned int kBitsPerWord = 64;
constexpr unsigned int kMaxKeyWords = StaticC2lsh::kMaxKeyBits / kBitsPerWord;
// Coordinates per block of points in a projection product, a (rows x d) by (d x mR) matrix product on one thread;
// a block this size stays in cache between its copy and the product.
constexpr size_t kProjectionBlockCoordinates = size_t{1} << 18;
constexpr size_t kMinProjectionRows = 32;
constexpr size_t kMaxProjectionRows = 2048;
// Bucket ids stay well inside int64 so that XORs and shifts never meet the sign bit's edge cases.
constexpr double kMaxBucketMagnitude = 0x1p62;
// Levels above this are never resolved: their cells are more than 2^47 finest buckets wide.
constexpr double kMaxTopLevel = 47.0;

double ElapsedMilliseconds(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

unsigned int BitLength(std::int64_t x, std::int64_t y) {
    return static_cast<unsigned int>(std::bit_width(static_cast<std::uint64_t>(x ^ y)));
}

// Appends the low `bits` bits of `value` to a big-endian bit string at bit offset `position`.
void AppendBits(std::uint64_t* words, unsigned int& position, std::uint64_t value, unsigned int bits) {
    const unsigned int word = position / kBitsPerWord;
    const unsigned int room = kBitsPerWord - position % kBitsPerWord;
    if (bits <= room) {
        words[word] |= value << (room - bits);
    } else {
        words[word] |= value >> (bits - room);
        words[word + 1] |= value << (kBitsPerWord - (bits - room));
    }
    position += bits;
}

// Removes bit j from a mask of m <= 32 bits and closes the gap.
std::uint64_t DeleteBit(std::uint32_t mask, unsigned int j) {
    const std::uint64_t wide = mask;
    return (wide & ((std::uint64_t{1} << j) - 1U)) | ((wide >> (j + 1U)) << j);
}

struct SweepInput {
    // Per point, key_bits masks of m bits, the bit planes of its bucket ids from the highest kept bit down: bit i of a
    // mask is the plane's bit of projection i.
    const std::vector<std::uint32_t>& planes;
    unsigned int key_bits;
    unsigned int num_sources;
    unsigned int num_targets;
    unsigned int num_projections;
    unsigned int window;
};

template <unsigned int Words>
struct Record {
    std::array<std::uint64_t, Words> key;
    unsigned int id;
};

// For every omitted projection j: sorts all points by the Morton key of the other m - 1 bucket ids, plane by plane
// from the highest bit, and records the nearest `window` targets on each side of every source.
template <unsigned int Words>
void SortAndSweep(const SweepInput& input, std::vector<unsigned int>& targets) {
    const unsigned int m = input.num_projections;
    const unsigned int window = input.window;
    const size_t num_points = static_cast<size_t>(input.num_sources) + input.num_targets;
    const size_t row_size = 2 * static_cast<size_t>(window) * m;

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (long long omitted = 0; omitted < static_cast<long long>(m); ++omitted) {
        const auto j = static_cast<unsigned int>(omitted);
        std::vector<Record<Words>> records(num_points);
        for (size_t point = 0; point < num_points; ++point) {
            Record<Words>& record = records[point];
            record.key.fill(0);
            record.id = static_cast<unsigned int>(point);
            const std::uint32_t* masks = input.planes.data() + point * input.key_bits;
            unsigned int position = 0;
            for (unsigned int plane = 0; plane < input.key_bits; ++plane) {
                AppendBits(record.key.data(), position, DeleteBit(masks[plane], j), m - 1);
            }
        }
        std::ranges::sort(records, [](const Record<Words>& lhs, const Record<Words>& rhs) {
            for (unsigned int word = 0; word < Words; ++word) {
                if (lhs.key[word] != rhs.key[word]) {
                    return lhs.key[word] < rhs.key[word];
                }
            }
            return false;
        });

        // Each sweep keeps the last `window` targets it passed in a ring, the newest at (seen - 1) % window.
        std::vector<unsigned int> recent(window, StaticC2lsh::kNoTarget);
        const auto sweep = [&](size_t position, unsigned int side, size_t& seen) {
            const unsigned int id = records[position].id;
            if (id >= input.num_sources) {
                recent[seen % window] = id - input.num_sources;
                ++seen;
                return;
            }
            unsigned int* slots = targets.data() + id * row_size + (2 * static_cast<size_t>(j) + side) * window;
            const size_t available = std::min<size_t>(seen, window);
            for (size_t rank = 0; rank < available; ++rank) {
                slots[rank] = recent[(seen - 1 - rank) % window];
            }
        };
        size_t seen = 0;
        for (size_t position = 0; position < num_points; ++position) {
            sweep(position, 0, seen);
        }
        seen = 0;
        for (size_t position = num_points; position > 0; --position) {
            sweep(position - 1, 1, seen);
        }
    }
}

using SweepFunction = void (*)(const SweepInput&, std::vector<unsigned int>&);

template <unsigned int... Indices>
constexpr std::array<SweepFunction, sizeof...(Indices)> MakeSweepTable(std::integer_sequence<unsigned int, Indices...>) {
    return {&SortAndSweep<Indices + 1>...};
}

// Entry k sorts keys of k + 1 words.
constexpr std::array<SweepFunction, kMaxKeyWords> kSweepTable =
    MakeSweepTable(std::make_integer_sequence<unsigned int, kMaxKeyWords>{});

bool IsSupportedNorm(float norm_order) {
    return std::abs(norm_order - 1.0F) < Global::kEpsilon || std::abs(norm_order - 2.0F) < Global::kEpsilon;
}

// Points per projection product. Chunks start at multiples of this over the sequence of sources and then targets, in
// memory and on disk alike, so that both compute every projection with the same matrix product.
size_t ProjectionChunkRows(Eigen::Index d) {
    return std::clamp(kProjectionBlockCoordinates / static_cast<size_t>(d), kMinProjectionRows, kMaxProjectionRows);
}

// Projections of `rows` row-major points onto every column of `projections`, written row-major to `output`.
void ProjectChunk(const Coordinate* points, size_t rows, const Eigen::MatrixXf& projections, float* output) {
    const Eigen::Map<const RowMajorMatrix> block(points, static_cast<Eigen::Index>(rows), projections.rows());
    Eigen::Map<RowMajorMatrix> result(output, static_cast<Eigen::Index>(rows), projections.cols());
    result.noalias() = block * projections;
}

// Projections a . o of every source and then every target on every column of `projections`, row-major with one row
// per point. One pass over the points serves every repetition.
std::vector<float> ProjectPoints(const std::vector<Point>& sources, const std::vector<Point>& targets,
                                 const Eigen::MatrixXf& projections) {
    const Eigen::Index d = projections.rows();
    const auto columns = static_cast<size_t>(projections.cols());
    const size_t num_sources = sources.size();
    const size_t num_points = num_sources + targets.size();
    const size_t chunk_rows = ProjectionChunkRows(d);
    const size_t num_chunks = (num_points + chunk_rows - 1) / chunk_rows;
    std::vector<float> projected(num_points * columns);
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel
#endif
    {
        std::vector<Coordinate> block;
#ifdef QALSH4C_USE_OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (long long chunk = 0; chunk < static_cast<long long>(num_chunks); ++chunk) {
            const size_t begin = static_cast<size_t>(chunk) * chunk_rows;
            const size_t rows = std::min(chunk_rows, num_points - begin);
            block.resize(rows * static_cast<size_t>(d));
            for (size_t row = 0; row < rows; ++row) {
                const size_t point = begin + row;
                const Point& coordinates = point < num_sources ? sources[point] : targets[point - num_sources];
                std::ranges::copy(coordinates, block.data() + row * static_cast<size_t>(d));
            }
            ProjectChunk(block.data(), rows, projections, projected.data() + begin * columns);
        }
    }
    return projected;
}

void OpenPointFile(std::ifstream& input, const PointSetMetadata& metadata) {
    input.open(metadata.file_path, std::ios::binary);
    if (!input.is_open()) {
        Log::error("Could not open point set file: {}", metadata.file_path.string());
    }
}

// Reads `count` consecutive points starting at point `first` of a point file.
void ReadPoints(std::ifstream& input, const PointSetMetadata& metadata, size_t first, size_t count,
                Coordinate* output) {
    const size_t coordinates_per_point = metadata.num_dimensions;
    input.clear();
    input.seekg(static_cast<std::streamoff>(first * coordinates_per_point * sizeof(Coordinate)), std::ios::beg);
    const auto bytes = static_cast<std::streamsize>(count * coordinates_per_point * sizeof(Coordinate));
    input.read(reinterpret_cast<char*>(output), bytes);
    if (!input || input.gcount() != bytes) {
        Log::error("Failed to read {} points starting at point {} from {}.", count, first, metadata.file_path.string());
    }
}

// ProjectPoints for point files. The sources and then the targets are read sequentially in groups of whole chunks of
// at least block_size points, and the chunks of a group are projected in parallel.
std::vector<float> ProjectPointFiles(const PointSetMetadata& sources, const PointSetMetadata& targets,
                                     const Eigen::MatrixXf& projections, unsigned int block_size,
                                     double& read_bytes) {
    const Eigen::Index d = projections.rows();
    const auto columns = static_cast<size_t>(projections.cols());
    const size_t num_sources = sources.num_points;
    const size_t num_points = num_sources + targets.num_points;
    const size_t chunk_rows = ProjectionChunkRows(d);
    const size_t group_rows = std::max<size_t>(1, (block_size + chunk_rows - 1) / chunk_rows) * chunk_rows;
    std::ifstream source_input;
    std::ifstream target_input;
    OpenPointFile(source_input, sources);
    OpenPointFile(target_input, targets);
    std::vector<float> projected(num_points * columns);
    std::vector<Coordinate> group;
    for (size_t group_begin = 0; group_begin < num_points; group_begin += group_rows) {
        const size_t rows = std::min(group_rows, num_points - group_begin);
        const size_t source_rows = group_begin < num_sources ? std::min(rows, num_sources - group_begin) : 0;
        group.resize(rows * static_cast<size_t>(d));
        if (source_rows > 0) {
            ReadPoints(source_input, sources, group_begin, source_rows, group.data());
        }
        if (rows > source_rows) {
            ReadPoints(target_input, targets, group_begin + source_rows - num_sources, rows - source_rows,
                       group.data() + source_rows * static_cast<size_t>(d));
        }
        read_bytes += static_cast<double>(rows * static_cast<size_t>(d) * sizeof(Coordinate));
        const size_t num_chunks = (rows + chunk_rows - 1) / chunk_rows;
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (long long chunk = 0; chunk < static_cast<long long>(num_chunks); ++chunk) {
            const size_t begin = static_cast<size_t>(chunk) * chunk_rows;
            ProjectChunk(group.data() + begin * static_cast<size_t>(d), std::min(chunk_rows, rows - begin),
                         projections, projected.data() + (group_begin + begin) * columns);
        }
    }
    return projected;
}

// Bucket ids floor((a_i . o + b_i) / width) for the m projections that start at column `first_column`, m per point.
// Each id is then rebased below `top_level`: its level-top cell is renumbered from the smallest one in use. Cells at
// levels up to top_level stay exactly the same, and ids become small non-negative integers, so Morton keys need only
// a few bits above top_level instead of the 64 that ids on both sides of zero would differ in.
std::vector<std::int64_t> RebasedBucketIds(const std::vector<float>& projected, size_t num_columns,
                                           unsigned int first_column, unsigned int num_projections,
                                           const std::vector<double>& shifts, double width, unsigned int top_level) {
    const size_t num_points = projected.size() / num_columns;
    std::vector<std::int64_t> bucket_ids(num_points * num_projections);
    std::vector<std::int64_t> min_cells(num_projections, std::numeric_limits<std::int64_t>::max());
    bool overflow = false;
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel
#endif
    {
        std::vector<std::int64_t> local_min_cells(num_projections, std::numeric_limits<std::int64_t>::max());
#ifdef QALSH4C_USE_OPENMP
#pragma omp for schedule(static) reduction(|| : overflow)
#endif
        for (long long index = 0; index < static_cast<long long>(num_points); ++index) {
            const auto point = static_cast<size_t>(index);
            for (unsigned int i = 0; i < num_projections; ++i) {
                const double value =
                    std::floor((static_cast<double>(projected[point * num_columns + first_column + i]) + shifts[i]) /
                               width);
                if (!(std::abs(value) < kMaxBucketMagnitude)) {
                    overflow = true;
                    continue;
                }
                const auto id = static_cast<std::int64_t>(value);
                bucket_ids[point * num_projections + i] = id;
                local_min_cells[i] = std::min(local_min_cells[i], id >> top_level);
            }
        }
#ifdef QALSH4C_USE_OPENMP
#pragma omp critical(static_c2lsh_min_cells)
#endif
        for (unsigned int i = 0; i < num_projections; ++i) {
            min_cells[i] = std::min(min_cells[i], local_min_cells[i]);
        }
    }
    if (overflow) {
        Log::error("Static C2LSH bucket ids overflow with bucket width {}; the points or the width are out of range.",
                   width);
    }

    const std::int64_t low_mask = (std::int64_t{1} << top_level) - 1;
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long index = 0; index < static_cast<long long>(num_points); ++index) {
        std::int64_t* ids = bucket_ids.data() + static_cast<size_t>(index) * num_projections;
        for (unsigned int i = 0; i < num_projections; ++i) {
            ids[i] = (((ids[i] >> top_level) - min_cells[i]) << top_level) | (ids[i] & low_mask);
        }
    }
    return bucket_ids;
}

}  // namespace

namespace StaticC2lsh {

unsigned int DefaultNumProjections(size_t num_targets) {
    const double log_targets = std::log(static_cast<double>(std::max<size_t>(num_targets, 1)));
    const double log_collision = -0.5 * std::log(2.0 * std::numbers::pi);
    unsigned int m = 4;
    while (m < kMaxProjections &&
           log_targets + std::log(static_cast<double>(m)) + static_cast<double>(m - 1) * log_collision > 0.0) {
        ++m;
    }
    return m;
}

unsigned int PairLevel(const std::int64_t* x, const std::int64_t* y, unsigned int num_projections) {
    unsigned int largest = 0;
    unsigned int second = 0;
    for (unsigned int i = 0; i < num_projections; ++i) {
        const unsigned int length = BitLength(x[i], y[i]);
        if (length > largest) {
            second = largest;
            largest = length;
        } else if (length > second) {
            second = length;
        }
    }
    return second;
}

Neighbors MortonNeighbors(const std::vector<std::int64_t>& bucket_ids, unsigned int num_sources,
                          unsigned int num_targets, unsigned int num_projections, unsigned int window) {
    const unsigned int m = num_projections;
    const size_t num_points = static_cast<size_t>(num_sources) + num_targets;
    if (m < 2 || m > kMaxProjections) {
        Log::error("Static C2LSH needs between 2 and {} projections, got {}.", kMaxProjections, m);
    }
    if (window == 0) {
        Log::error("Static C2LSH needs a positive window.");
    }
    if (num_targets == 0) {
        Log::error("Static C2LSH needs a non-empty target point set.");
    }
    if (bucket_ids.size() != num_points * m) {
        Log::error("Static C2LSH got {} bucket ids for {} points and {} projections.", bucket_ids.size(), num_points,
                   m);
    }

    // Bits above the highest one in which two points differ are shared by every point, so they never split a cell.
    unsigned int key_bits = 1;
    for (unsigned int i = 0; i < m; ++i) {
        std::uint64_t spread = 0;
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(static) reduction(| : spread)
#endif
        for (long long point = 0; point < static_cast<long long>(num_points); ++point) {
            spread |= static_cast<std::uint64_t>(bucket_ids[static_cast<size_t>(point) * m + i] ^ bucket_ids[i]);
        }
        key_bits = std::max(key_bits, static_cast<unsigned int>(std::bit_width(spread)));
    }
    // Keys longer than kMaxKeyBits drop the lowest bits, which coarsens the finest levels.
    const unsigned int max_key_bits = kMaxKeyBits / (m - 1);
    const unsigned int dropped_bits = key_bits > max_key_bits ? key_bits - max_key_bits : 0;
    key_bits -= dropped_bits;

    std::vector<std::uint32_t> planes(num_points * key_bits, 0U);
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long index = 0; index < static_cast<long long>(num_points); ++index) {
        const auto point = static_cast<size_t>(index);
        std::uint32_t* masks = planes.data() + point * key_bits;
        for (unsigned int i = 0; i < m; ++i) {
            const auto value = static_cast<std::uint64_t>(bucket_ids[point * m + i] >> dropped_bits);
            for (unsigned int plane = 0; plane < key_bits; ++plane) {
                masks[plane] |= static_cast<std::uint32_t>((value >> (key_bits - 1 - plane)) & 1U) << i;
            }
        }
    }

    Neighbors neighbors;
    neighbors.key_bits = key_bits;
    neighbors.dropped_bits = dropped_bits;
    const size_t row_size = 2 * static_cast<size_t>(window) * m;
    neighbors.targets.assign(num_sources * row_size, kNoTarget);
    const unsigned int key_words = (key_bits * (m - 1) + kBitsPerWord - 1) / kBitsPerWord;
    const SweepInput input{.planes = planes,
                           .key_bits = key_bits,
                           .num_sources = num_sources,
                           .num_targets = num_targets,
                           .num_projections = m,
                           .window = window};
    kSweepTable[key_words - 1](input, neighbors.targets);

    // Levels count from the finest bucket ids given, so coarsened ids report their dropped levels as well.
    neighbors.first_level.assign(num_sources, 0);
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long index = 0; index < static_cast<long long>(num_sources); ++index) {
        const auto source = static_cast<size_t>(index);
        std::array<std::int64_t, kMaxProjections> source_ids{};
        std::array<std::int64_t, kMaxProjections> target_ids{};
        for (unsigned int i = 0; i < m; ++i) {
            source_ids[i] = bucket_ids[source * m + i] >> dropped_bits;
        }
        unsigned int level = key_bits;
        for (size_t slot = 0; slot < row_size; ++slot) {
            const unsigned int target = neighbors.targets[source * row_size + slot];
            if (target == kNoTarget) {
                continue;
            }
            const size_t target_row = (static_cast<size_t>(num_sources) + target) * m;
            for (unsigned int i = 0; i < m; ++i) {
                target_ids[i] = bucket_ids[target_row + i] >> dropped_bits;
            }
            level = std::min(level, PairLevel(source_ids.data(), target_ids.data(), m));
        }
        neighbors.first_level[source] = level + dropped_bits;
    }
    return neighbors;
}

}  // namespace StaticC2lsh

namespace {

void ValidateOptions(float norm_order, const StaticC2lshOptions& options) {
    if (!IsSupportedNorm(norm_order)) {
        Log::error("Static C2LSH supports only L1 and L2, got norm order {}.", norm_order);
    }
    if (options.num_repetitions == 0 || options.window == 0) {
        Log::error("Static C2LSH needs at least one repetition and a positive window.");
    }
    if (!(options.bucket_width >= 0.0) || !std::isfinite(options.bucket_width)) {
        Log::error("Static C2LSH needs a finite, non-negative bucket width, got {}.", options.bucket_width);
    }
}

struct BoundRange {
    double min_positive{std::numeric_limits<double>::infinity()};
    double max{0.0};
    size_t num_positive{0};
};

BoundRange PositiveBoundRange(const std::vector<double>& upper_bounds) {
    BoundRange range;
    for (const double bound : upper_bounds) {
        if (bound > 0.0) {
            range.min_positive = std::min(range.min_positive, bound);
            range.max = std::max(range.max, bound);
            ++range.num_positive;
        }
    }
    return range;
}

// The random draws of every structure: m projection vectors per repetition, standard Gaussian for L2 and standard
// Cauchy for L1 (the p-stable laws of the two norms), then each repetition's level phase and shifts.
struct Structures {
    Eigen::MatrixXf projections;  // d x mR; repetition r owns columns [r m, (r + 1) m)
    std::vector<double> widths;
    std::vector<unsigned int> top_levels;
    std::vector<std::vector<double>> shifts;
};

Structures DrawStructures(Eigen::Index d, unsigned int m, unsigned int repetitions, float norm_order,
                          double bucket_width, double max_bound, std::mt19937& gen) {
    Structures structures;
    const auto columns = static_cast<Eigen::Index>(static_cast<size_t>(m) * repetitions);
    if (std::abs(norm_order - 1.0F) < Global::kEpsilon) {
        std::cauchy_distribution<float> cauchy(0.0F, 1.0F);
        structures.projections = Eigen::MatrixXf::NullaryExpr(d, columns, [&]() { return cauchy(gen); });
    } else {
        std::normal_distribution<float> normal(0.0F, 1.0F);
        structures.projections = Eigen::MatrixXf::NullaryExpr(d, columns, [&]() { return normal(gen); });
    }
    // A random phase per structure puts the level widths at w 2^(k + U), U ~ U[0, 1), which spreads the gap between
    // an ideal width and the level below it; the shifts cover every level up to twice the largest bound.
    std::uniform_real_distribution<double> phase(0.0, 1.0);
    structures.widths.resize(repetitions);
    structures.top_levels.resize(repetitions);
    structures.shifts.assign(repetitions, std::vector<double>(m));
    for (unsigned int repetition = 0; repetition < repetitions; ++repetition) {
        const double width = bucket_width * std::exp2(phase(gen));
        const double levels = std::ceil(std::log2(max_bound / width));
        structures.widths[repetition] = width;
        structures.top_levels[repetition] = static_cast<unsigned int>(std::clamp(levels, 0.0, kMaxTopLevel)) + 1;
        std::uniform_real_distribution<double> shift(0.0, width * std::exp2(structures.top_levels[repetition]));
        for (double& value : structures.shifts[repetition]) {
            value = shift(gen);
        }
    }
    return structures;
}

// Every repetition's Morton neighbors of every source: num_sources rows of repetitions * 2 * window * m target ids,
// which are verified together once all structures are sorted.
std::vector<unsigned int> CollectNeighbors(const std::vector<float>& projected, const Structures& structures,
                                           unsigned int m, unsigned int num_sources, unsigned int num_targets,
                                           unsigned int window, const std::vector<double>& upper_bounds,
                                           size_t num_positive, StaticC2lshStats& stats) {
    using Clock = std::chrono::steady_clock;
    const auto repetitions = static_cast<unsigned int>(structures.widths.size());
    const size_t num_columns = static_cast<size_t>(m) * repetitions;
    const size_t row_size = 2 * static_cast<size_t>(window) * m;
    std::vector<unsigned int> candidates(static_cast<size_t>(num_sources) * repetitions * row_size);
    double level_sum = 0.0;
    for (unsigned int repetition = 0; repetition < repetitions; ++repetition) {
        const auto bucket_begin = Clock::now();
        const std::vector<std::int64_t> bucket_ids =
            RebasedBucketIds(projected, num_columns, repetition * m, m, structures.shifts[repetition],
                             structures.widths[repetition], structures.top_levels[repetition]);
        const auto sort_begin = Clock::now();
        const StaticC2lsh::Neighbors neighbors =
            StaticC2lsh::MortonNeighbors(bucket_ids, num_sources, num_targets, m, window);
        const auto sort_end = Clock::now();
        stats.projection_ms += ElapsedMilliseconds(bucket_begin, sort_begin);
        stats.sort_ms += ElapsedMilliseconds(sort_begin, sort_end);
        stats.key_bits = std::max(stats.key_bits, neighbors.key_bits);

        for (unsigned int source = 0; source < num_sources; ++source) {
            std::copy_n(neighbors.targets.begin() + static_cast<std::ptrdiff_t>(source * row_size), row_size,
                        candidates.begin() +
                            static_cast<std::ptrdiff_t>((static_cast<size_t>(source) * repetitions + repetition) *
                                                        row_size));
            if (upper_bounds[source] > 0.0) {
                level_sum += neighbors.first_level[source];
            }
        }
    }
    stats.mean_first_level = level_sum / (static_cast<double>(num_positive) * repetitions);
    return candidates;
}

// The distinct neighbors of one source over every repetition, in ascending order.
void DistinctNeighbors(const std::vector<unsigned int>& candidates, size_t source, size_t neighbors_per_source,
                       std::vector<unsigned int>& distinct) {
    const auto row = candidates.begin() + static_cast<std::ptrdiff_t>(source * neighbors_per_source);
    distinct.assign(row, row + static_cast<std::ptrdiff_t>(neighbors_per_source));
    std::erase(distinct, StaticC2lsh::kNoTarget);
    std::ranges::sort(distinct);
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
}

// The verification of StaticC2lshUpperBounds for point files, one chunk of sources at a time. A chunk is read once,
// and B is streamed in blocks of block_size points, reading only the blocks that hold a neighbor of the chunk. Each
// distance starts from its source's best bound at the start of the block; the minimum over the neighbors is exact
// either way, so the bounds equal those of the in-memory verification.
std::vector<double> VerifyFromFiles(const PointSetMetadata& sources, const PointSetMetadata& targets,
                                    const std::vector<double>& upper_bounds,
                                    const std::vector<unsigned int>& candidates, size_t neighbors_per_source,
                                    float norm_order, unsigned int chunk_size, unsigned int block_size,
                                    StaticC2lshStats& stats, size_t& num_checked) {
    const size_t d = sources.num_dimensions;
    std::ifstream source_input;
    std::ifstream target_input;
    OpenPointFile(source_input, sources);
    OpenPointFile(target_input, targets);
    std::vector<double> tightened = upper_bounds;
    std::vector<Coordinate> chunk;
    std::vector<Coordinate> block;
    std::vector<unsigned int> distinct;
    std::vector<std::pair<unsigned int, unsigned int>> pairs;  // (target, source in the chunk)
    std::vector<Distance> best;
    std::vector<Distance> distances;
    for (size_t chunk_begin = 0; chunk_begin < sources.num_points; chunk_begin += chunk_size) {
        const size_t chunk_end = std::min<size_t>(chunk_begin + chunk_size, sources.num_points);
        pairs.clear();
        for (size_t source = chunk_begin; source < chunk_end; ++source) {
            if (upper_bounds[source] <= 0.0) {
                continue;
            }
            DistinctNeighbors(candidates, source, neighbors_per_source, distinct);
            for (const unsigned int target : distinct) {
                pairs.emplace_back(target, static_cast<unsigned int>(source - chunk_begin));
            }
        }
        if (pairs.empty()) {
            continue;
        }
        num_checked += pairs.size();
        std::ranges::sort(pairs);
        chunk.resize((chunk_end - chunk_begin) * d);
        ReadPoints(source_input, sources, chunk_begin, chunk_end - chunk_begin, chunk.data());
        stats.read_mib += static_cast<double>(chunk.size() * sizeof(Coordinate)) / (1 << 20);
        ++stats.verify_passes;
        best.resize(chunk_end - chunk_begin);
        for (size_t source = chunk_begin; source < chunk_end; ++source) {
            best[source - chunk_begin] = static_cast<Distance>(upper_bounds[source]);
        }

        size_t pair_begin = 0;
        for (size_t block_begin = 0; block_begin < targets.num_points && pair_begin < pairs.size();
             block_begin += block_size) {
            const size_t block_end = std::min<size_t>(block_begin + block_size, targets.num_points);
            size_t pair_end = pair_begin;
            while (pair_end < pairs.size() && pairs[pair_end].first < block_end) {
                ++pair_end;
            }
            if (pair_end == pair_begin) {
                continue;  // No neighbor in this block: skip reading it.
            }
            block.resize((block_end - block_begin) * d);
            ReadPoints(target_input, targets, block_begin, block_end - block_begin, block.data());
            stats.read_mib += static_cast<double>(block.size() * sizeof(Coordinate)) / (1 << 20);
            distances.resize(pair_end - pair_begin);
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (long long index = 0; index < static_cast<long long>(pair_end - pair_begin); ++index) {
                const auto& [target, source] = pairs[pair_begin + static_cast<size_t>(index)];
                distances[static_cast<size_t>(index)] =
                    Calc::LpDistanceBoundedRaw(chunk.data() + source * d, block.data() + (target - block_begin) * d, d,
                                               norm_order, best[source]);
            }
            for (size_t index = pair_begin; index < pair_end; ++index) {
                Distance& source_best = best[pairs[index].second];
                source_best = std::min(source_best, distances[index - pair_begin]);
            }
            pair_begin = pair_end;
        }
        for (size_t source = chunk_begin; source < chunk_end; ++source) {
            tightened[source] = std::min(upper_bounds[source], static_cast<double>(best[source - chunk_begin]));
        }
    }
    return tightened;
}

}  // namespace

std::vector<double> StaticC2lshUpperBounds(const std::vector<Point>& sources, const std::vector<Point>& targets,
                                           const std::vector<double>& upper_bounds, float norm_order,
                                           const StaticC2lshOptions& options, std::mt19937& gen,
                                           StaticC2lshStats& stats) {
    using Clock = std::chrono::steady_clock;
    ValidateOptions(norm_order, options);
    if (targets.empty()) {
        Log::error("Static C2LSH needs a non-empty target point set.");
    }
    if (upper_bounds.size() != sources.size()) {
        Log::error("Static C2LSH got {} upper bounds for {} source points.", upper_bounds.size(), sources.size());
    }
    const unsigned int m =
        options.num_projections != 0 ? options.num_projections : StaticC2lsh::DefaultNumProjections(targets.size());
    const BoundRange range = PositiveBoundRange(upper_bounds);
    stats = StaticC2lshStats{.num_projections = m};
    if (range.num_positive == 0) {
        return upper_bounds;
    }
    stats.bucket_width = options.bucket_width > 0.0 ? options.bucket_width : range.min_positive;

    const auto projection_begin = Clock::now();
    const auto d = static_cast<Eigen::Index>(targets.front().size());
    const Structures structures =
        DrawStructures(d, m, options.num_repetitions, norm_order, stats.bucket_width, range.max, gen);
    const std::vector<float> projected = ProjectPoints(sources, targets, structures.projections);
    stats.projection_ms += ElapsedMilliseconds(projection_begin, Clock::now());
    const auto num_sources = static_cast<unsigned int>(sources.size());
    const std::vector<unsigned int> candidates =
        CollectNeighbors(projected, structures, m, num_sources, static_cast<unsigned int>(targets.size()),
                         options.window, upper_bounds, range.num_positive, stats);

    const auto verify_begin = Clock::now();
    const size_t neighbors_per_source = static_cast<size_t>(options.num_repetitions) * 2 * options.window * m;
    std::vector<double> tightened = upper_bounds;
    size_t num_checked = 0;
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel
#endif
    {
        std::vector<unsigned int> distinct;
#ifdef QALSH4C_USE_OPENMP
#pragma omp for schedule(dynamic) reduction(+ : num_checked)
#endif
        for (long long index = 0; index < static_cast<long long>(num_sources); ++index) {
            const auto source = static_cast<size_t>(index);
            if (upper_bounds[source] <= 0.0) {
                continue;
            }
            DistinctNeighbors(candidates, source, neighbors_per_source, distinct);
            auto best = static_cast<Distance>(upper_bounds[source]);
            for (const unsigned int target : distinct) {
                best = std::min(best, Calc::LpDistanceBounded(sources[source], targets[target], norm_order, best));
            }
            tightened[source] = std::min(upper_bounds[source], static_cast<double>(best));
            num_checked += distinct.size();
        }
    }
    stats.verify_ms = ElapsedMilliseconds(verify_begin, Clock::now());
    stats.candidates_per_source = static_cast<double>(num_checked) / static_cast<double>(range.num_positive);
    return tightened;
}

std::vector<double> StaticC2lshUpperBoundsDisk(const PointSetMetadata& sources, const PointSetMetadata& targets,
                                               const std::vector<double>& upper_bounds, float norm_order,
                                               const StaticC2lshOptions& options, unsigned int block_size,
                                               std::mt19937& gen, StaticC2lshStats& stats) {
    using Clock = std::chrono::steady_clock;
    ValidateOptions(norm_order, options);
    if (sources.num_dimensions != targets.num_dimensions || sources.num_dimensions == 0) {
        Log::error("Static C2LSH needs point sets of the same positive dimension, got {} and {}.",
                   sources.num_dimensions, targets.num_dimensions);
    }
    if (targets.num_points == 0) {
        Log::error("Static C2LSH needs a non-empty target point set.");
    }
    if (upper_bounds.size() != sources.num_points) {
        Log::error("Static C2LSH got {} upper bounds for {} source points.", upper_bounds.size(), sources.num_points);
    }
    if (block_size == 0) {
        Log::error("Static C2LSH needs a positive disk block size.");
    }
    const unsigned int m = options.num_projections != 0 ? options.num_projections
                                                        : StaticC2lsh::DefaultNumProjections(targets.num_points);
    const BoundRange range = PositiveBoundRange(upper_bounds);
    stats = StaticC2lshStats{.num_projections = m};
    if (range.num_positive == 0) {
        return upper_bounds;
    }
    stats.bucket_width = options.bucket_width > 0.0 ? options.bucket_width : range.min_positive;

    const auto projection_begin = Clock::now();
    const auto d = static_cast<Eigen::Index>(sources.num_dimensions);
    const Structures structures =
        DrawStructures(d, m, options.num_repetitions, norm_order, stats.bucket_width, range.max, gen);
    double read_bytes = 0.0;
    const std::vector<float> projected =
        ProjectPointFiles(sources, targets, structures.projections, block_size, read_bytes);
    stats.read_mib = read_bytes / (1 << 20);
    stats.projection_ms += ElapsedMilliseconds(projection_begin, Clock::now());
    const std::vector<unsigned int> candidates =
        CollectNeighbors(projected, structures, m, sources.num_points, targets.num_points, options.window,
                         upper_bounds, range.num_positive, stats);

    const auto verify_begin = Clock::now();
    const size_t point_bytes = static_cast<size_t>(sources.num_dimensions) * sizeof(Coordinate);
    const unsigned int chunk_size =
        options.disk_chunk_size != 0
            ? options.disk_chunk_size
            : static_cast<unsigned int>(std::max<size_t>(1, StaticC2lsh::kDefaultChunkBytes / point_bytes));
    const size_t neighbors_per_source = static_cast<size_t>(options.num_repetitions) * 2 * options.window * m;
    size_t num_checked = 0;
    std::vector<double> tightened = VerifyFromFiles(sources, targets, upper_bounds, candidates, neighbors_per_source,
                                                    norm_order, chunk_size, block_size, stats, num_checked);
    stats.verify_ms = ElapsedMilliseconds(verify_begin, Clock::now());
    stats.candidates_per_source = static_cast<double>(num_checked) / static_cast<double>(range.num_positive);
    return tightened;
}
