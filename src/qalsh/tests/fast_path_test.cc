// The search must not depend on how the index was built or on the bfloat16 candidate filter: an index built with a
// serial accessor and no filter, and one built by concurrent workers with the filter, must give the same neighbors
// (distances and ids), candidate count, hit count, termination reason and completeness, for the quanta order and for
// the one-entry-at-a-time reference of the projection-distance order. The slab implementation of the
// projection-distance order must evaluate the same candidates as that reference, for several slab counts. The test runs
// on random data for both metrics and several index and strategy settings.

#include "qalsh/qalsh.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qalsh::Coordinate;
using qalsh::PointId;
using qalsh::PointView;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("fast path test failed: " + message);
    }
}

enum class Values { gaussian, pixels, mixed_scales };

std::vector<std::vector<Coordinate>> MakePoints(std::size_t count, std::size_t dimensions, Values values,
                                                std::mt19937& generator) {
    std::normal_distribution<float> gaussian(0.0F, 1.0F);
    std::uniform_int_distribution<int> pixel(0, 255);
    std::vector<std::vector<Coordinate>> points(count, std::vector<Coordinate>(dimensions));
    for (auto& point : points) {
        for (std::size_t i = 0; i < dimensions; ++i) {
            switch (values) {
                case Values::gaussian:
                    point[i] = gaussian(generator);
                    break;
                case Values::pixels:
                    point[i] = static_cast<float>(pixel(generator));
                    break;
                case Values::mixed_scales:
                    // A few wide coordinates, many narrow ones and some constant ones, like sparse features.
                    point[i] = i % 7 == 0 ? 100.0F * gaussian(generator)
                                          : (i % 5 == 0 ? 0.0F : 0.01F * gaussian(generator));
                    break;
            }
        }
    }
    // Exact duplicates give distance-zero neighbors and ties in every table.
    for (std::size_t copy = 0; copy + 1 < count && copy < 8; ++copy) {
        points[count - 1 - copy] = points[copy];
    }
    return points;
}

// QALSH's table rule for approximation ratio 2, error probability 1/e and beta = 1 (QALSH4C's setting).
qalsh::QalshParameters StandardParameters(qalsh::Metric metric) {
    constexpr double c = 2.0;
    const double term1 = std::sqrt(std::log(2.0));
    const double term2 = 1.0;
    const double eta = term1 / term2;
    double bucket_width = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;
    if (metric == qalsh::Metric::l1) {
        bucket_width = 2.0 * std::sqrt(c);
        p1 = 2.0 / std::numbers::pi * std::atan(bucket_width / 2.0);
        p2 = 2.0 / std::numbers::pi * std::atan(bucket_width / (2.0 * c));
    } else {
        bucket_width = std::sqrt((8.0 * c * c * std::log(c)) / (c * c - 1.0));
        p1 = std::erf(bucket_width / 2.0 / std::sqrt(2.0));
        p2 = std::erf(bucket_width / (2.0 * c) / std::sqrt(2.0));
    }
    const auto tables =
        static_cast<std::uint32_t>(std::ceil(std::pow(term1 + term2, 2.0) / (2.0 * (p1 - p2) * (p1 - p2))));
    qalsh::QalshParameters parameters;
    parameters.approximation_ratio = static_cast<float>(c);
    parameters.bucket_width = static_cast<float>(bucket_width);
    parameters.num_hash_tables = tables;
    parameters.collision_threshold = static_cast<std::uint32_t>(std::ceil((eta * p1 + p2) / (1.0 + eta) * tables));
    parameters.radius_growth = static_cast<float>(c);
    return parameters;
}

struct Case {
    qalsh::Metric metric{qalsh::Metric::l2};
    Values values{Values::gaussian};
    std::size_t num_points{0};
    std::size_t dimensions{0};
    std::uint32_t scan_quantum{256};
    float initial_radius{1.0F};
};

void CheckSame(const qalsh::SearchResult& actual, const qalsh::SearchResult& expected, const std::string& where) {
    Check(actual.neighbors.size() == expected.neighbors.size(), where + ": neighbor count");
    for (std::size_t i = 0; i < actual.neighbors.size(); ++i) {
        Check(actual.neighbors[i].point_id == expected.neighbors[i].point_id, where + ": neighbor id");
        Check(std::bit_cast<std::uint32_t>(actual.neighbors[i].distance) ==
                  std::bit_cast<std::uint32_t>(expected.neighbors[i].distance),
              where + ": neighbor distance");
    }
    Check(actual.evaluated_candidates == expected.evaluated_candidates, where + ": evaluated candidates");
    Check(actual.projection_hits == expected.projection_hits, where + ": projection hits");
    Check(actual.reason == expected.reason, where + ": termination reason");
    Check(actual.complete == expected.complete, where + ": completeness");
}

// The slab implementation counts every entry of the slab in which it stops, so only the hit counts may differ.
void CheckSameCandidates(const qalsh::SearchResult& slabs, const qalsh::SearchResult& reference,
                         const std::string& where) {
    Check(slabs.neighbors.size() == reference.neighbors.size(), where + ": neighbor count");
    for (std::size_t i = 0; i < slabs.neighbors.size(); ++i) {
        Check(slabs.neighbors[i].point_id == reference.neighbors[i].point_id, where + ": neighbor id");
        Check(std::bit_cast<std::uint32_t>(slabs.neighbors[i].distance) ==
                  std::bit_cast<std::uint32_t>(reference.neighbors[i].distance),
              where + ": neighbor distance");
    }
    Check(slabs.evaluated_candidates == reference.evaluated_candidates, where + ": evaluated candidates");
    Check(slabs.complete == reference.complete, where + ": completeness");
}

std::size_t RunCase(const Case& test_case, std::uint32_t seed) {
    std::mt19937 generator(seed);
    const auto points = MakePoints(test_case.num_points, test_case.dimensions, test_case.values, generator);
    auto queries = MakePoints(10, test_case.dimensions, test_case.values, generator);
    // Queries that are target points themselves, and one far away.
    queries.push_back(points[3]);
    queries.push_back(points[test_case.num_points / 2]);
    queries.push_back(std::vector<Coordinate>(test_case.dimensions, 1000.0F));

    const qalsh::PointAccessor accessor = [&points](PointId id) -> PointView { return points[id]; };
    const auto num_points = static_cast<std::uint32_t>(test_case.num_points);
    qalsh::IndexConfig config;
    config.metric = test_case.metric;
    config.num_points = num_points;
    config.num_dimensions = static_cast<std::uint32_t>(test_case.dimensions);
    config.qalsh = StandardParameters(test_case.metric);
    config.qalsh.candidate_budget = num_points;
    config.qalsh.scan_quantum = test_case.scan_quantum;
    config.qalsh.initial_radius = test_case.initial_radius;
    config.seed = seed;
    config.build_threads = 4;
    config.point_projections = true;
    // The plain index: serial accessor, no compressed copy. The filtered index: concurrent build with the copy.
    const auto plain_index = qalsh::InMemoryIndex::Build(config, accessor);
    config.concurrent_accessor = true;
    config.compressed_filter = true;
    const auto filtered_index = qalsh::InMemoryIndex::Build(config, accessor);

    const qalsh::SearchEngine plain(plain_index, accessor);
    const qalsh::SearchEngine filtered(filtered_index, accessor);
    // The projection-distance order: the one-at-a-time reference on both indexes, and the slab implementation with
    // several slab counts on the filtered index.
    qalsh::SearchOptions reference_options;
    reference_options.scan_order = qalsh::SearchOptions::ScanOrder::projection_distance_reference;
    const qalsh::SearchEngine distance_reference(plain_index, accessor, reference_options);
    const qalsh::SearchEngine filtered_distance_reference(filtered_index, accessor, reference_options);
    std::vector<std::unique_ptr<qalsh::SearchEngine>> distance_engines;
    for (const std::uint32_t slabs : {1U, 2U, 16U}) {
        qalsh::SearchOptions options;
        options.scan_order = qalsh::SearchOptions::ScanOrder::projection_distance;
        options.slabs_per_round = slabs;
        distance_engines.push_back(std::make_unique<qalsh::SearchEngine>(filtered_index, accessor, options));
    }

    struct StrategySetting {
        const char* name;
        bool check_after_evaluation;
        float mid_round_ratio;
        std::optional<std::size_t> budget;
        bool budget_override;
        bool check_after_radius_advance;
        bool continuous{false};
    };
    const StrategySetting settings[] = {
        {"library default", true, 0.0F, std::nullopt, false, true},
        {"qalsh4c hybrid", false, 1.0F, std::nullopt, true, true},
        {"qalsh4c hybrid, round-end check only", false, 1.0F, std::nullopt, true, false},
        {"round end", false, 0.0F, std::nullopt, true, true},
        {"round end only", false, 0.0F, std::nullopt, true, false},
        {"within round", true, 0.0F, std::nullopt, true, true},
        {"budget", false, 1.0F, std::size_t{5}, true, true},
        {"continuous", false, 1.0F, std::nullopt, true, true, true},
    };
    std::size_t compared = 0;
    for (const auto& setting : settings) {
        for (const std::uint32_t k : {1U, 3U}) {
            for (std::size_t query = 0; query < queries.size(); ++query) {
                const auto make_strategy = [&] {
                    qalsh::DefaultQalshStrategy strategy;
                    if (setting.budget_override) strategy.set_candidate_budget(setting.budget);
                    strategy.set_check_after_evaluation(setting.check_after_evaluation);
                    strategy.set_mid_round_ratio(setting.mid_round_ratio);
                    strategy.set_check_after_radius_advance(setting.check_after_radius_advance);
                    strategy.set_continuous_stop(setting.continuous);
                    return strategy;
                };
                const std::string where = std::string(test_case.metric == qalsh::Metric::l1 ? "l1" : "l2") + " n=" +
                                          std::to_string(test_case.num_points) + " d=" +
                                          std::to_string(test_case.dimensions) + " seed=" + std::to_string(seed) +
                                          " " + setting.name + " k=" + std::to_string(k) + " query " +
                                          std::to_string(query);
                const auto strategy = make_strategy();
                if (!setting.continuous) {
                    const auto expected = plain.search(queries[query], k, strategy);
                    CheckSame(filtered.search(queries[query], k, strategy), expected, where + " (quanta, filtered)");
                    ++compared;
                }

                const auto by_distance = distance_reference.search(queries[query], k, strategy);
                CheckSame(filtered_distance_reference.search(queries[query], k, strategy), by_distance,
                          where + " (distance reference, filtered)");
                ++compared;
                for (std::size_t engine = 0; engine < distance_engines.size(); ++engine) {
                    CheckSameCandidates(distance_engines[engine]->search(queries[query], k, strategy), by_distance,
                                        where + " (distance order, engine " + std::to_string(engine) + ")");
                    ++compared;
                }
            }
        }
    }
    return compared;
}

}  // namespace

int main() {
    try {
        std::size_t compared = 0;
        const Case cases[] = {
            {qalsh::Metric::l2, Values::gaussian, 1500, 48, 256, 0x1p-149F},
            {qalsh::Metric::l1, Values::gaussian, 1500, 48, 256, 0x1p-149F},
            {qalsh::Metric::l2, Values::pixels, 1500, 64, 256, 1.0F},
            {qalsh::Metric::l1, Values::pixels, 1500, 64, 7, 1.0F},
            {qalsh::Metric::l2, Values::mixed_scales, 1500, 200, 2, 0x1p-149F},
            {qalsh::Metric::l1, Values::mixed_scales, 1500, 200, 3, 0x1p-149F},
            {qalsh::Metric::l2, Values::gaussian, 300, 5, 256, 1.0F},
        };
        for (const auto& test_case : cases) {
            for (const std::uint32_t seed : {1U, 2U}) {
                compared += RunCase(test_case, seed);
            }
        }
        std::printf("%zu search comparisons agree\n", compared);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
