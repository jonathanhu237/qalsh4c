#include <qalsh/qalsh.h>
#include <cstdint>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

void Check(bool b, std::source_location location = std::source_location::current()) {
    if (!b) throw std::runtime_error("distance regression failed at line " + std::to_string(location.line()));
}

// A search engine over the given points and a one-table index with the given projection vector.
qalsh::SearchEngine Engine(const std::vector<std::vector<float>>& points, std::vector<float> projection_vector,
                           qalsh::Metric metric = qalsh::Metric::l2) {
    qalsh::IndexConfig config;
    config.metric = metric;
    config.num_points = static_cast<std::uint32_t>(points.size());
    config.num_dimensions = static_cast<std::uint32_t>(projection_vector.size());
    config.qalsh.num_hash_tables = 1;
    config.qalsh.collision_threshold = 1;
    config.qalsh.bucket_width = 2.5F;
    config.projection_vectors = std::move(projection_vector);
    const qalsh::PointAccessor accessor = [&points](qalsh::PointId id) -> qalsh::PointView { return points.at(id); };
    return qalsh::SearchEngine(qalsh::InMemoryIndex::Build(config, accessor), accessor);
}

int main() {
    const qalsh::DefaultQalshStrategy strategy;
    // Distances whose squares do not fit in float32 are still exact.
    for (float value : {1e-30F, 1e-22F, 1e20F}) {
        const std::vector<std::vector<float>> points{{value}};
        const auto result = Engine(points, {1}).search(std::vector<float>{0}, 1, strategy);
        Check(result.complete && result.neighbors.size() == 1 && result.neighbors[0].distance == value);
    }
    // width*radius may overflow float even when width*radius/2 fits.
    const float far = 2e38F;
    {
        const std::vector<std::vector<float>> points{{far}};
        const auto result = Engine(points, {1}).search(std::vector<float>{0}, 1, strategy);
        Check(result.complete && result.neighbors.size() == 1 && result.neighbors[0].distance == far);
    }

    // All points project to zero, so every point is a candidate in the first round, in increasing id.
    const float maximum = std::numeric_limits<float>::max();
    const std::vector<float> query{-maximum, 0};
    for (auto norm : {qalsh::Metric::l1, qalsh::Metric::l2}) {
        // Without a bound, a distance above the float32 range is an error.
        {
            const std::vector<std::vector<float>> points{{maximum, 0}};
            const auto engine = Engine(points, {0, 1}, norm);
            bool threw = false;
            try {
                (void)engine.search(query, 1, strategy);
            } catch (const std::overflow_error&) {
                threw = true;
            }
            Check(threw);
        }
        // With the bound of the first neighbor, at distance maximum, the same candidate is rejected instead.
        {
            const std::vector<std::vector<float>> points{{0, 0}, {maximum, 0}};
            const auto result = Engine(points, {0, 1}, norm).search(query, 1, strategy);
            Check(result.neighbors.size() == 1 && result.neighbors[0].point_id == 0 &&
                  result.neighbors[0].distance == maximum && result.evaluated_candidates == 2);
        }
    }
}
