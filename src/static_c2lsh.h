#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <vector>

#include "point_set.h"

// Static C2LSH over A and B with collision threshold l = m - 1 and c = 2 (Gan et al., SIGMOD 2012, with integer
// virtual rehashing). Every point gets m bucket ids h_i = floor((a_i . o + b_i) / w), and level k merges them into
// floor(h_i / 2^k). A source and a target collide in at least m - 1 projections at level k exactly when they share
// the level-k cell of some delete-one key, the m - 1 bucket ids left after omitting projection j. Sorting A and B
// together in the Morton order of a delete-one key puts every such cell in one contiguous run, so the targets next
// to a source in that order include one that shares its smallest cell, as the quadtree DFS order does for
// hypercubes. One structure therefore needs m sorts and no per-point queries.
struct StaticC2lshOptions {
    unsigned int num_projections{0};  // m; 0 selects DefaultNumProjections(|B|)
    unsigned int num_repetitions{1};  // Independent structures; each source keeps its best witness over all of them
    unsigned int window{1};           // Targets checked on each side of a source in every Morton order
    double bucket_width{0.0};         // w; 0 selects the smallest positive upper bound
    // External memory only: source points held in memory while their neighbors are verified, one pass over B per
    // chunk; 0 selects as many as kDefaultChunkBytes holds.
    unsigned int disk_chunk_size{0};
    // External memory only: read B's projections from its persistent static index instead of its coordinates.
    bool disk_index{false};
    // Without an index: draw the projection vectors from std::mt19937(vector_seed), the way an index draws them,
    // instead of from the run's generator.
    std::optional<std::uint32_t> vector_seed{};
};

struct StaticC2lshStats {
    unsigned int num_projections{0};
    unsigned int key_bits{0};  // Largest number of low bits kept per bucket id in a Morton key
    double bucket_width{0.0};
    double projection_ms{0.0};  // Projections and bucket ids
    double sort_ms{0.0};        // Morton keys, sorts and neighbor sweeps
    double verify_ms{0.0};      // Distances to the distinct neighbors
    double candidates_per_source{0.0};  // Distinct neighbors per source point with a positive bound
    double mean_first_level{0.0};       // First collision level, averaged over repetitions and source points
    double read_mib{0.0};               // External memory only: point data read from the files
    unsigned int verify_passes{0};      // External memory only: passes over B to verify neighbors
};

namespace StaticC2lsh {

inline constexpr unsigned int kNoTarget = std::numeric_limits<unsigned int>::max();
// Morton keys hold at most this many bits; coarser bucket ids are kept when a key would be longer.
inline constexpr unsigned int kMaxKeyBits = 1024;
inline constexpr unsigned int kMaxProjections = 32;
// External memory: the default budget for the source points held while verifying neighbors.
inline constexpr size_t kDefaultChunkBytes = size_t{256} << 20;

// The smallest m >= 4 with |B| m (2 pi)^(-(m-1)/2) <= 1, which bounds the far-target term of the tail bound
// Pr[D_a / v_a > t] <= C_m t^(-2(m-1)/(m+1)).
unsigned int DefaultNumProjections(size_t num_targets);

// The first level at which two points share the bucket of at least m - 1 projections: the second largest bit length
// of h_i(x) XOR h_i(y), since floor(h / 2^k) agrees exactly when the XOR has no bit at position k or above.
unsigned int PairLevel(const std::int64_t* x, const std::int64_t* y, unsigned int num_projections);

struct Neighbors {
    // num_sources rows of 2 * window * m target ids (kNoTarget where an order runs out of targets): for every omitted
    // projection j, the nearest `window` targets before and after the source in that Morton order.
    std::vector<unsigned int> targets;
    // Per source, the smallest PairLevel over its neighbors, which is the first level at which any target collides
    // with it in at least m - 1 projections.
    std::vector<unsigned int> first_level;
    unsigned int key_bits{0};      // Low bits kept per bucket id in the Morton keys
    unsigned int dropped_bits{0};  // Lowest bits dropped to fit kMaxKeyBits, which coarsens the finest levels
};

// Morton neighbors for row-major bucket ids of num_sources sources followed by num_targets targets, m per point.
Neighbors MortonNeighbors(const std::vector<std::int64_t>& bucket_ids, unsigned int num_sources,
                          unsigned int num_targets, unsigned int num_projections, unsigned int window);

// A persistent static index of one point set, built once like the disk QALSH index: m R projection vectors drawn from
// a seed, and the set's projections onto them. An external-memory estimate then reads 4 m R bytes per target point
// instead of its coordinates, and draws only the level phases and shifts itself. It lives in
// <dataset>/index/static/l<p>/<a|b>.
struct Index {
    std::filesystem::path directory;
    float norm_order{0.0F};
    unsigned int num_points{0};
    unsigned int num_dimensions{0};
    unsigned int num_projections{0};  // m
    unsigned int num_repetitions{0};  // R the index holds; an estimate may use fewer
    std::uint32_t seed{0};
    std::vector<float> vectors;  // d x m R, column-major; repetition r owns columns [r m, (r + 1) m)
};

[[nodiscard]] std::filesystem::path IndexDirectory(const PointSetMetadata& points, float norm_order);
// num_projections 0 selects DefaultNumProjections(points.num_points). An index with the same parameters is kept.
void BuildIndex(const PointSetMetadata& points, float norm_order, unsigned int num_projections,
                unsigned int num_repetitions, std::uint32_t seed, unsigned int block_size, bool overwrite);
[[nodiscard]] Index OpenIndex(const PointSetMetadata& points, float norm_order);

}  // namespace StaticC2lsh

struct StaticC2lshDiskOptions {
    unsigned int block_size{0};  // Points per sequential read
    const StaticC2lsh::Index* index{nullptr};  // B's projections; null projects B from its point file
    // Called with every block of B, in id order, during the first verification pass, which then reads every block, so
    // that an estimator can scan its pilot queries in the same pass.
    std::function<void(const Coordinate* block, unsigned int count, unsigned int first_id)> target_block_hook;
};

// Tightens every positive upper bound U_a to min(U_a, distance to the nearest Morton neighbor over all repetitions).
// Every neighbor is a real target point, so the result is still an upper bound, and it never exceeds the distance to
// a first-collision witness, which the tail bound needs. Zero bounds, from exact target matches, stay zero.
std::vector<double> StaticC2lshUpperBounds(const std::vector<Point>& sources, const std::vector<Point>& targets,
                                           const std::vector<double>& upper_bounds, float norm_order,
                                           const StaticC2lshOptions& options, std::mt19937& gen,
                                           StaticC2lshStats& stats);

// The same bounds for point sets on disk. A is streamed from its file in blocks of disk.block_size points, and B too
// unless disk.index holds its projections; memory holds no point vectors beyond one block or one chunk of source
// points: the projections (m R floats per point), one repetition's bucket ids and Morton keys, and the neighbor lists.
// B is read once per chunk of source points to verify the neighbors, skipping blocks that hold no neighbor. With the
// same generator state, and the index's seed as options.vector_seed when an index is used, it returns exactly what
// StaticC2lshUpperBounds returns.
std::vector<double> StaticC2lshUpperBoundsDisk(const PointSetMetadata& sources, const PointSetMetadata& targets,
                                               const std::vector<double>& upper_bounds, float norm_order,
                                               const StaticC2lshOptions& options, const StaticC2lshDiskOptions& disk,
                                               std::mt19937& gen, StaticC2lshStats& stats);
