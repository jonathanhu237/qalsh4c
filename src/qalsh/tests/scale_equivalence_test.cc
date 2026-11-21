#include "qalsh/qalsh.h"

#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Multiplying every point by a power of two while keeping the initial radius is the same search as keeping the
// points and dividing the initial radius by that power: projections, windows, and distances all scale exactly in
// IEEE arithmetic, so both searches visit the same entries, evaluate the same candidates, and return the same
// neighbor at exactly the scaled distance.

namespace {

using qalsh::Coordinate;
using qalsh::PointId;
using qalsh::PointView;

constexpr std::uint32_t kNumPoints = 3000;
constexpr std::uint32_t kNumDimensions = 48;
constexpr std::uint32_t kNumQueries = 60;
constexpr float kScale = 4096.0F;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("scale equivalence test failed: " + message);
    }
}

std::vector<std::vector<Coordinate>> RandomPoints(std::mt19937& gen, std::uint32_t count) {
    std::normal_distribution<float> coordinate(0.0F, 1.0F);
    std::vector<std::vector<Coordinate>> points(count, std::vector<Coordinate>(kNumDimensions));
    for (auto& point : points) {
        for (auto& value : point) {
            value = coordinate(gen);
        }
    }
    return points;
}

std::vector<std::vector<Coordinate>> Scaled(const std::vector<std::vector<Coordinate>>& points) {
    auto scaled = points;
    for (auto& point : scaled) {
        for (auto& value : point) {
            value *= kScale;
        }
    }
    return scaled;
}

qalsh::IndexConfig Config(float initial_radius) {
    qalsh::IndexConfig config;
    config.metric = qalsh::Metric::l2;
    config.num_points = kNumPoints;
    config.num_dimensions = kNumDimensions;
    config.qalsh = qalsh::QalshParameters{.approximation_ratio = 2.0F,
                                          .bucket_width = 2.719F,
                                          .error_probability = 0.1F,
                                          .num_hash_tables = 24,
                                          .collision_threshold = 9,
                                          .candidate_budget = 100,
                                          .initial_radius = initial_radius,
                                          .radius_growth = 2.0F,
                                          .scan_quantum = 64};
    config.seed = 7;
    return config;
}

qalsh::SearchResult Search(const qalsh::SearchEngine& engine, const std::vector<Coordinate>& query) {
    qalsh::DefaultQalshStrategy strategy(9);
    strategy.set_candidate_budget(std::nullopt);
    return engine.search(PointView(query), 1, strategy);
}

}  // namespace

int main() {
    try {
        std::mt19937 gen(2026);
        const auto points = RandomPoints(gen, kNumPoints);
        const auto queries = RandomPoints(gen, kNumQueries);
        const auto scaled_points = Scaled(points);
        const auto scaled_queries = Scaled(queries);

        const qalsh::PointAccessor accessor = [&points](PointId id) -> PointView { return points.at(id); };
        const qalsh::PointAccessor scaled_accessor = [&scaled_points](PointId id) -> PointView {
            return scaled_points.at(id);
        };
        const qalsh::SearchEngine engine(qalsh::InMemoryIndex::Build(Config(1.0F / kScale), accessor), accessor);
        const qalsh::SearchEngine scaled_engine(qalsh::InMemoryIndex::Build(Config(1.0F), scaled_accessor),
                                                scaled_accessor);

        for (std::uint32_t query = 0; query < kNumQueries; ++query) {
            const qalsh::SearchResult result = Search(engine, queries[query]);
            const qalsh::SearchResult scaled_result = Search(scaled_engine, scaled_queries[query]);
            const std::string label = "query " + std::to_string(query);
            Check(result.neighbors.size() == 1 && scaled_result.neighbors.size() == 1, label + " found no neighbor");
            Check(result.neighbors[0].point_id == scaled_result.neighbors[0].point_id, label + " neighbor differs");
            Check(result.neighbors[0].distance * kScale == scaled_result.neighbors[0].distance,
                  label + " distance is not exactly scaled");
            Check(result.evaluated_candidates == scaled_result.evaluated_candidates,
                  label + " evaluated a different number of candidates");
            Check(result.projection_hits == scaled_result.projection_hits,
                  label + " visited a different number of projection entries");
            Check(result.reason == scaled_result.reason, label + " stopped for a different reason");
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
