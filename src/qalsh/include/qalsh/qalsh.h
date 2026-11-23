#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace qalsh {

using PointId = std::uint32_t;
using Coordinate = float;
using Projection = float;
using Distance = float;

/** The first-version distance kernels supported by the projection index. */
enum class Metric : std::uint32_t {
    l1 = 1,
    l2 = 2,
};

/**
 * A borrowed view of one original point.
 *
 * A PointAccessor may return a view into an immutable data set or into a
 * reusable scratch buffer.  The library consumes the view before it invokes
 * the accessor again; callers must not retain the view after that call.
 */
using PointView = std::span<const Coordinate>;
using PointAccessor = std::function<PointView(PointId)>;

struct QalshParameters {
    float approximation_ratio{2.0F};
    float bucket_width{0.0F};
    float error_probability{1.0F / 2.71828182845904523536F};
    std::uint32_t num_hash_tables{0};
    std::uint32_t collision_threshold{0};
    /** Base beta*n budget: default top-k uses min(n, base+k-1); zero is invalid. */
    std::uint32_t candidate_budget{100};
    float initial_radius{1.0F};
    float radius_growth{2.0F};
    std::uint32_t scan_quantum{128};
};

struct IndexConfig {
    Metric metric{Metric::l2};
    std::uint32_t num_points{0};
    std::uint32_t num_dimensions{0};
    QalshParameters qalsh{};
    std::uint32_t seed{42};
    /** Row-major vectors, with exactly num_hash_tables*num_dimensions values. */
    std::vector<Coordinate> projection_vectors;
    /** Internal projection/sorting workers (1..1024). Accessor calls stay serial, on the calling thread and in
     * increasing id, unless concurrent_accessor is set. */
    std::uint32_t build_threads{1};
    /** The accessor may be called from several build workers at once (for example, it returns views of immutable
     * in-memory vectors). Workers then read, check and project points themselves; otherwise the calling thread projects
     * every point. The index is the same either way. */
    bool concurrent_accessor{false};
    /** Keep a bfloat16 copy of the points. Before an exact bounded distance, the search computes a rigorous lower bound
     * from the copy and skips the candidate only when that bound proves that the exact bounded distance would reject
     * it. Results are unchanged; the copy halves the bytes read for most rejected candidates. */
    bool compressed_filter{false};
    /** Also keep every point's projections (point-major), which the projection-distance scan order uses to place each
     * new candidate exactly. */
    bool point_projections{false};
};

struct IndexMetadata {
    Metric metric{Metric::l2};
    std::uint32_t num_points{0};
    std::uint32_t num_dimensions{0};
    std::uint32_t num_hash_tables{0};
    std::uint32_t seed{0};
    QalshParameters qalsh{};
};

/** Read-only view of a projection index. InMemoryIndex is its only implementation. */
class ProjectionIndex {
   public:
    virtual ~ProjectionIndex() = default;
    [[nodiscard]] virtual const IndexMetadata& metadata() const noexcept = 0;
    [[nodiscard]] virtual std::span<const Coordinate> projection_vectors() const noexcept = 0;
};

class InMemoryIndex final : public ProjectionIndex {
   public:
    /** Build an ephemeral sorted-array index. */
    static std::shared_ptr<InMemoryIndex> Build(IndexConfig config, const PointAccessor& accessor);

    [[nodiscard]] const IndexMetadata& metadata() const noexcept override;
    [[nodiscard]] std::span<const Coordinate> projection_vectors() const noexcept override;

   private:
    friend class SearchEngine;
    struct Entry {
        Projection value{0.0F};
        PointId point_id{0};
    };
    struct Table {
        std::vector<Entry> entries;
        // entries[i].point_id for every i, so that collision counting reads only ids.
        std::vector<PointId> ids;
    };
    IndexMetadata metadata_{};
    std::vector<Coordinate> projection_vectors_;
    std::vector<Table> tables_;
    // Optional bfloat16 copy of the points (see IndexConfig::compressed_filter), one row per point, written by the
    // build workers while they project the points. Place j of a row holds coordinate compressed_order_[j]: coordinates
    // come in decreasing order of their variance over a sample of points, so the lower bound grows fast and rejects
    // early. compressed_kind_ says per point whether its copy is exact (1), rounded (0) or unusable because a
    // coordinate rounds to infinity (2).
    std::unique_ptr<std::uint16_t[]> compressed_points_;
    std::vector<std::uint8_t> compressed_kind_;
    std::vector<std::uint32_t> compressed_order_;
    // Optional point-major projections: row i holds point i's values in all tables (IndexConfig::point_projections).
    std::unique_ptr<Projection[]> point_projections_;

    InMemoryIndex() = default;
};

struct Neighbor {
    Distance distance{0.0F};
    PointId point_id{0};
};

/** Why a search ended: a stopping rule of the strategy held, or every table ran out of entries. */
enum class TerminationReason : std::uint32_t {
    strategy,
    scan_exhausted,
};

struct SearchResult {
    std::vector<Neighbor> neighbors;
    TerminationReason reason{TerminationReason::strategy};
    std::size_t evaluated_candidates{0};
    std::size_t projection_hits{0};
    /** All k neighbors were returned; this is not a quality guarantee. */
    bool complete{false};
};

struct SearchOptions {
    /** Query only the first num_hash_tables tables (default: all). */
    std::optional<std::uint32_t> num_hash_tables;
    /** Entries per table visit in the quanta order (0: the index's QalshParameters::scan_quantum). */
    std::uint32_t scan_quantum{0};
    /** Stop a candidate's distance computation once it exceeds the k-th distance. */
    bool bounded_distance{true};
    /** Check every evaluated candidate for the right size and finite coordinates. A caller whose accessor returns
     * the immutable vectors the index was built from (and checked) may switch this off. */
    bool validate_points{true};
    /** Order in which the search visits window entries inside a radius round. `quanta` alternates between the
     * tables, scan_quantum entries per visit (half on each side); it needs a scan quantum of at least two.
     * `projection_distance` processes the entries of all tables in increasing distance from the query's projection, so
     * every table's window grows at the same rate and points become candidates in increasing order of the radius at
     * which they reach the collision threshold; it needs an index built with IndexConfig::point_projections.
     * Points that become candidates at the same distance are evaluated in increasing id.
     * `projection_distance_reference` is a plain implementation of the same order, one distance at a time, for tests. */
    enum class ScanOrder : std::uint32_t { quanta, projection_distance, projection_distance_reference };
    ScanOrder scan_order{ScanOrder::quanta};
    /** For the projection-distance order: the distance range of each round is split into this many equal slabs, each
     * counted in one pass; candidates are still evaluated in exact order, so the result does not depend on it. More
     * slabs count fewer entries past a stop, but every slab costs a run search on each table side. */
    std::uint32_t slabs_per_round{2};
};

/** Settings of the standard collision-count QALSH search: the collision threshold and the stopping rules.
 * By default the search stops once beta*n + k - 1 candidates are evaluated (capped by n), or once the k-th distance
 * is at most approximation_ratio times the radius, checked after every evaluation, at the end of every round and right
 * after the radius grows.
 */
class DefaultQalshStrategy final {
   public:
    /** A threshold of zero uses the index's QalshParameters::collision_threshold. */
    explicit DefaultQalshStrategy(std::uint32_t collision_threshold = 0);
    /** Replace the default budget: a positive count, or std::nullopt for no budget. */
    void set_candidate_budget(std::optional<std::size_t> budget);
    void set_check_after_evaluation(bool enabled);
    /** Also check the stopping condition right after the radius grows, against the new radius (the original
     * behavior). Switching it off checks only at the end of each round, against that round's radius, as Algorithm 1
     * of QALSH does. */
    void set_check_after_radius_advance(bool enabled);
    /** With a projection-distance scan order only: stop as soon as the k-th distance is at most approximation_ratio
     * times the radius of the current front (twice its distance from the query's projection over the bucket width),
     * in place of the round rules. The radius then grows continuously, and the returned distance is at most
     * approximation_ratio times the larger of the nearest neighbor's collision radius and its distance over
     * approximation_ratio. */
    void set_continuous_stop(bool enabled);
    /** Stop inside a round once a candidate lies within ratio * radius (0 disables). */
    void set_mid_round_ratio(float ratio);

   private:
    friend class SearchEngine;

    std::uint32_t collision_threshold_{0};
    std::optional<std::size_t> candidate_budget_override_;
    bool candidate_budget_override_set_{false};
    bool check_after_evaluation_{true};
    float mid_round_ratio_{0.0F};
    bool check_after_radius_advance_{true};
    bool continuous_stop_{false};
};

class SearchEngine {
   public:
    /** The index must be an InMemoryIndex. */
    SearchEngine(std::shared_ptr<const ProjectionIndex> index, PointAccessor accessor,
                 SearchOptions options = {});
    [[nodiscard]] SearchResult search(PointView query, std::uint32_t k, const DefaultQalshStrategy& strategy) const;

   private:
    std::shared_ptr<const InMemoryIndex> index_;
    PointAccessor accessor_;
    SearchOptions options_{};
};

[[nodiscard]] const char* ToString(TerminationReason reason) noexcept;

}  // namespace qalsh
