#include <qalsh/qalsh.h>

#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("query speed regression: " + message);
}

using Neighbors = std::vector<std::pair<qalsh::PointId, std::uint32_t>>;
Neighbors NeighborBits(std::span<const qalsh::Neighbor> neighbors) {
    Neighbors result;
    for (const auto& neighbor : neighbors) {
        result.emplace_back(neighbor.point_id, std::bit_cast<std::uint32_t>(neighbor.distance));
    }
    return result;
}

void CompareResult(const qalsh::SearchResult& lhs, const qalsh::SearchResult& rhs) {
    Check(lhs.reason == rhs.reason && lhs.complete == rhs.complete &&
              lhs.evaluated_candidates == rhs.evaluated_candidates &&
              lhs.projection_hits == rhs.projection_hits &&
              NeighborBits(lhs.neighbors) == NeighborBits(rhs.neighbors),
          "typed and erased search results differ");
}

using Observation = std::tuple<float, float, std::size_t, std::size_t, bool, bool, Neighbors>;

struct SearchTrace {
    std::vector<qalsh::PointId> candidates;
    std::vector<Observation> observations;
};

void CompareDefaultDispatch(const std::shared_ptr<const qalsh::ProjectionIndex>& index,
                            const std::vector<std::array<float, 2>>& points) {
    for (const auto quantum : {1U, 3U, 128U}) {
        qalsh::SearchOptions options;
        options.scan_quantum = quantum;
        for (const auto stop_after : {0U, 1U, 5U, 10000U}) {
            for (const bool custom_candidate : {false, true}) {
                SearchTrace typed_trace, erased_trace;
                const auto accessor = [&](SearchTrace& trace) -> qalsh::PointAccessor {
                    return [&, trace_ptr = &trace](qalsh::PointId id) -> qalsh::PointView {
                        trace_ptr->candidates.push_back(id);
                        return points.at(id);
                    };
                };
                const qalsh::SearchEngine typed_engine(index, accessor(typed_trace), options);
                const qalsh::SearchEngine erased_engine(index, accessor(erased_trace), options);
                qalsh::DefaultQalshStrategy typed, erased;
                const auto configure = [&](qalsh::DefaultQalshStrategy& strategy, SearchTrace& trace) {
                    strategy.set_candidate_budget(std::nullopt);
                    if (custom_candidate) {
                        strategy.set_candidate_rule([](const qalsh::CollisionObservation& hit) {
                            return hit.collision_count >= 2U;
                        });
                    }
                    if (stop_after != 0U) {
                        strategy.set_termination_rule([&, trace_ptr = &trace](const qalsh::QuerySnapshot& state) {
                            trace_ptr->observations.emplace_back(
                                state.radius, state.bound, state.evaluated_candidates,
                                state.projection_hits, state.current_round_complete,
                                state.all_tables_exhausted, NeighborBits(state.neighbors));
                            return state.evaluated_candidates >= stop_after;
                        });
                    }
                };
                configure(typed, typed_trace);
                configure(erased, erased_trace);
                qalsh::SearchStrategy& erased_base = erased;
                for (const auto query : {std::array<float, 2>{0.25F, -0.75F},
                                         std::array<float, 2>{-30.0F, 25.0F}}) {
                    typed_trace = {};
                    erased_trace = {};
                    const auto a = typed_engine.search(query, 7U, typed);
                    const auto b = erased_engine.search(query, 7U, erased_base);
                    CompareResult(a, b);
                    Check(typed_trace.candidates == erased_trace.candidates &&
                              typed_trace.observations == erased_trace.observations,
                          "candidate order or stop-predicate observations differ");
                    if (stop_after == 10000U) {
                        Check(a.evaluated_candidates == points.size() &&
                                  a.projection_hits == points.size() * index->metadata().num_hash_tables,
                              "exhaustive query missed accepted-zero counter reuse");
                    }
                    if (stop_after == 1U && quantum == 128U) {
                        Check(a.evaluated_candidates == 1U && a.projection_hits > 1U,
                              "early stop lost the consumed scan boundary");
                    }
                }
            }
        }
    }
}

using ScanStep = std::tuple<std::vector<std::pair<qalsh::PointId, float>>,
                            std::size_t, bool, bool>;

std::vector<ScanStep> ScanTrace(const qalsh::ProjectionIndex& index, float query,
                               float bound, std::size_t quantum, qalsh::ScanScope scope) {
    auto cursor = index.make_cursor(0U, query);
    std::vector<ScanStep> trace;
    for (std::size_t step = 0; step <= index.metadata().num_points; ++step) {
        std::vector<std::pair<qalsh::PointId, float>> hits;
        const auto report = index.scan(
            *cursor, bound, quantum, &hits,
            [](void* context, const qalsh::ProjectionHit& hit) {
                auto& hits = *static_cast<std::vector<std::pair<qalsh::PointId, float>>*>(context);
                hits.emplace_back(hit.point_id, hit.projected_value);
            }, scope);
        trace.emplace_back(std::move(hits), report.entries_visited,
                           report.window_exhausted, report.table_exhausted);
        if (report.window_exhausted || report.table_exhausted) return trace;
    }
    throw std::runtime_error("query speed regression: scan did not complete");
}

void CheckSeparatorDescent(const std::filesystem::path& directory) {
    // At 512 bytes, leaves hold 60 entries and internal nodes 40 children.
    // More than 2400 points forces two internal levels; repeated keys cross
    // both leaf and internal-node boundaries.
    std::vector<std::array<float, 1>> points(10001U);
    for (std::size_t i = 0; i < points.size(); ++i) {
        points[i][0] = static_cast<float>(i / 300U) - 16.0F;
    }
    qalsh::IndexConfig config;
    config.num_points = static_cast<std::uint32_t>(points.size());
    config.num_dimensions = 1U;
    config.qalsh.num_hash_tables = 1U;
    config.qalsh.collision_threshold = 1U;
    config.qalsh.bucket_width = 2.0F;
    config.projection_vectors = {1.0F};
    const qalsh::PointAccessor accessor = [&](qalsh::PointId id) -> qalsh::PointView {
        return points.at(id);
    };
    const auto memory = qalsh::InMemoryIndex::Build(config, accessor);
    const auto tree_path = directory / "query-speed-tree.qalsh";
    const auto array_path = directory / "query-speed-array.qalsh";
    qalsh::BPlusTreeIndex::Build(tree_path.string(), config, accessor, 512U, true);
    qalsh::SortedArrayIndex::Build(array_path.string(), config, accessor, 512U, true);
    const auto tree = qalsh::BPlusTreeIndex::Open(tree_path.string());
    const auto array = qalsh::SortedArrayIndex::Open(array_path.string());
    const auto infinity = std::numeric_limits<float>::infinity();
    for (const float query : {-17.0F, -16.0F, -15.0F, -10.0F, -3.0F,
                              std::nextafter(0.0F, -infinity), 0.0F,
                              std::nextafter(0.0F, infinity), 4.0F, 10.0F, 16.0F, 17.0F, 18.0F}) {
        for (const auto bound : {0.0F, 1.0F, 100.0F}) {
            for (const auto quantum : {1U, 17U, 128U}) {
                const auto actual = ScanTrace(*tree, query, bound, quantum, qalsh::ScanScope::quantum);
                Check(actual == ScanTrace(*memory, query, bound, quantum, qalsh::ScanScope::quantum),
                      "B+ separator descent differs from the in-memory lower bound");
                Check(actual == ScanTrace(*array, query, bound, quantum, qalsh::ScanScope::quantum),
                      "B+ separator descent differs from the sorted-array lower bound");
            }
            Check(ScanTrace(*tree, query, bound, 128U, qalsh::ScanScope::range) ==
                      ScanTrace(*array, query, bound, 128U, qalsh::ScanScope::range),
                  "B+ range boundaries changed for duplicate separator keys");
        }
    }
    std::filesystem::remove(tree_path);
    std::filesystem::remove(array_path);
}
}  // namespace

int main(int argc, char** argv) {
    Check(argc == 2, "expected a temporary directory");
    const std::filesystem::path directory(argv[1]);
    std::vector<std::array<float, 2>> points(257U);
    for (std::size_t i = 0; i < points.size(); ++i) {
        points[i] = {static_cast<float>(static_cast<int>(i % 17U) - 8),
                     static_cast<float>(static_cast<int>(i % 11U) - 5)};
    }
    qalsh::IndexConfig config;
    config.num_points = static_cast<std::uint32_t>(points.size());
    config.num_dimensions = 2U;
    config.qalsh.num_hash_tables = 3U;
    config.qalsh.collision_threshold = 2U;
    config.qalsh.bucket_width = 2.0F;
    config.projection_vectors = {1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
    const qalsh::PointAccessor accessor = [&](qalsh::PointId id) -> qalsh::PointView {
        return points.at(id);
    };
    const auto tree_path = directory / "query-speed-dispatch-tree.qalsh";
    const auto array_path = directory / "query-speed-dispatch-array.qalsh";
    for (const auto metric : {qalsh::Metric::l1, qalsh::Metric::l2}) {
        config.metric = metric;
        CompareDefaultDispatch(qalsh::InMemoryIndex::Build(config, accessor), points);
        qalsh::BPlusTreeIndex::Build(tree_path.string(), config, accessor, 512U, true);
        qalsh::SortedArrayIndex::Build(array_path.string(), config, accessor, 512U, true);
        CompareDefaultDispatch(qalsh::BPlusTreeIndex::Open(tree_path.string()), points);
        CompareDefaultDispatch(qalsh::SortedArrayIndex::Open(array_path.string()), points);
    }
    std::filesystem::remove(tree_path);
    std::filesystem::remove(array_path);
    CheckSeparatorDescent(directory);
}
