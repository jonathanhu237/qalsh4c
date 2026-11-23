#include <qalsh/qalsh.h>

#include <algorithm>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void Check(bool value, std::source_location where = std::source_location::current()) {
    if (!value) throw std::runtime_error("parallel build invariant failed at line " + std::to_string(where.line()));
}

// Every call overwrites the same storage. Retaining borrowed point views or
// invoking this accessor from a worker violates the serial build contract.
class ScratchPoints {
   public:
    explicit ScratchPoints(std::size_t dimensions) : point_(dimensions) {}

    qalsh::PointView Read(qalsh::PointId id) {
        Check(std::this_thread::get_id() == owner_);
        for (std::size_t dimension = 0; dimension < point_.size(); ++dimension) {
            point_[dimension] =
                static_cast<float>((id * 17U + dimension * 13U) % (97U + dimension % 31U)) * .125F;
        }
        return point_;
    }

    qalsh::PointAccessor BuildAccessor() {
        calls = 0;
        return [this](qalsh::PointId id) {
            Check(id == calls);
            ++calls;
            return Read(id);
        };
    }

    // Copies of the first `count` points, for an accessor that concurrent build workers may call.
    std::vector<std::vector<float>> Stored(std::uint32_t count) {
        std::vector<std::vector<float>> stored;
        stored.reserve(count);
        for (qalsh::PointId id = 0; id < count; ++id) {
            const auto point = Read(id);
            stored.emplace_back(point.begin(), point.end());
        }
        return stored;
    }

    std::size_t calls{0};

   private:
    const std::thread::id owner_{std::this_thread::get_id()};
    std::vector<float> point_;
};

qalsh::PointAccessor StoredAccessor(const std::vector<std::vector<float>>& stored) {
    return [&stored](qalsh::PointId id) -> qalsh::PointView { return stored.at(id); };
}

qalsh::IndexConfig Config(std::uint32_t num_points, std::uint32_t dimensions,
                           qalsh::Metric metric = qalsh::Metric::l2) {
    qalsh::IndexConfig config;
    config.metric = metric;
    config.num_points = num_points;
    config.num_dimensions = dimensions;
    config.qalsh.num_hash_tables = 9;
    config.qalsh.collision_threshold = 3;
    config.qalsh.bucket_width = 2;
    config.point_projections = true;
    return config;
}

// The tables are private, so two indexes are compared through their searches: the same neighbors, candidates, hits
// and termination reasons in both scan orders.
void CheckSameSearches(const std::shared_ptr<qalsh::InMemoryIndex>& expected,
                       const std::shared_ptr<qalsh::InMemoryIndex>& actual, ScratchPoints& points) {
    Check(std::ranges::equal(expected->projection_vectors(), actual->projection_vectors()));
    const qalsh::PointAccessor access = [&](qalsh::PointId id) { return points.Read(id); };
    for (const auto order : {qalsh::SearchOptions::ScanOrder::quanta,
                             qalsh::SearchOptions::ScanOrder::projection_distance}) {
        qalsh::SearchOptions options;
        options.scan_order = order;
        qalsh::SearchEngine first(expected, access, options), second(actual, access, options);
        const auto k = std::min(5U, expected->metadata().num_points);
        for (unsigned i = 0; i < 20; ++i) {
            const auto point = points.Read(i);
            std::vector<float> query(point.begin(), point.end());
            query.front() += .1F;
            const qalsh::DefaultQalshStrategy strategy;
            const auto x = first.search(query, k, strategy), y = second.search(query, k, strategy);
            Check(x.projection_hits == y.projection_hits && x.evaluated_candidates == y.evaluated_candidates &&
                  x.reason == y.reason && x.neighbors.size() == y.neighbors.size());
            for (std::size_t j = 0; j < x.neighbors.size(); ++j) {
                Check(x.neighbors[j].point_id == y.neighbors[j].point_id &&
                      x.neighbors[j].distance == y.neighbors[j].distance);
            }
        }
    }
}

void CheckBuildParity(qalsh::IndexConfig config) {
    ScratchPoints points(config.num_dimensions);
    auto serial = qalsh::InMemoryIndex::Build(config, points.BuildAccessor());
    Check(points.calls == config.num_points);
    config.build_threads = 4;
    auto parallel = qalsh::InMemoryIndex::Build(config, points.BuildAccessor());
    Check(points.calls == config.num_points);
    CheckSameSearches(serial, parallel, points);

    // Concurrent workers read and project the points themselves.
    const auto stored = points.Stored(config.num_points);
    config.concurrent_accessor = true;
    auto concurrent = qalsh::InMemoryIndex::Build(config, StoredAccessor(stored));
    CheckSameSearches(serial, concurrent, points);
}

void CheckBuildFailures() {
    auto config = Config(4217, 257);
    config.build_threads = 4;
    ScratchPoints points(config.num_dimensions);
    (void)qalsh::InMemoryIndex::Build(config, points.BuildAccessor());

    // An accessor failure stops the serial build and reaches the caller.
    const auto ordered_accessor = points.BuildAccessor();
    bool rejected = false;
    try {
        (void)qalsh::InMemoryIndex::Build(config, [&](qalsh::PointId id) -> qalsh::PointView {
            const auto point = ordered_accessor(id);
            if (id == 4100) throw std::runtime_error("accessor failure");
            return point;
        });
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()) == "accessor failure";
    }
    Check(rejected && points.calls == 4101);

    const auto invalid_accessor = points.BuildAccessor();
    rejected = false;
    try {
        (void)qalsh::InMemoryIndex::Build(config, [&](qalsh::PointId id) {
            const auto point = invalid_accessor(id);
            return id == 4100 ? point.first(point.size() - 1) : point;
        });
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Check(rejected && points.calls == 4101);

    // Finite inputs whose projection overflows fail inside projection work.
    auto overflowing = config;
    overflowing.projection_vectors.assign(
        static_cast<std::size_t>(config.num_dimensions) * config.qalsh.num_hash_tables, 1.0F);
    const std::vector<float> overflow_point(config.num_dimensions, std::numeric_limits<float>::max());
    const auto overflow_accessor = points.BuildAccessor();
    rejected = false;
    try {
        (void)qalsh::InMemoryIndex::Build(overflowing, [&](qalsh::PointId id) -> qalsh::PointView {
            const auto point = overflow_accessor(id);
            return id == 3000 ? qalsh::PointView(overflow_point) : point;
        });
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Check(rejected && points.calls > 0 && points.calls <= config.num_points);

    (void)qalsh::InMemoryIndex::Build(config, points.BuildAccessor());
    Check(points.calls == config.num_points);

    // The same failures inside concurrent workers reach the caller without terminating or leaving workers behind.
    const auto stored = points.Stored(config.num_points);
    const auto stored_accessor = StoredAccessor(stored);
    config.concurrent_accessor = true;
    overflowing.concurrent_accessor = true;
    rejected = false;
    try {
        (void)qalsh::InMemoryIndex::Build(config, [&](qalsh::PointId id) -> qalsh::PointView {
            if (id == 4100) throw std::runtime_error("accessor failure");
            return stored_accessor(id);
        });
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()) == "accessor failure";
    }
    Check(rejected);

    rejected = false;
    try {
        (void)qalsh::InMemoryIndex::Build(config, [&](qalsh::PointId id) {
            const auto point = stored_accessor(id);
            return id == 4100 ? point.first(point.size() - 1) : point;
        });
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Check(rejected);

    rejected = false;
    try {
        (void)qalsh::InMemoryIndex::Build(overflowing, [&](qalsh::PointId id) -> qalsh::PointView {
            return id == 3000 ? qalsh::PointView(overflow_point) : stored_accessor(id);
        });
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Check(rejected);

    (void)qalsh::InMemoryIndex::Build(config, stored_accessor);
}
}  // namespace

int main() {
    CheckBuildParity(Config(3, 1));
    CheckBuildParity(Config(1000, 3));
    // Several point tasks per worker, an uneven tail, and SIMD dimension tails.
    for (const auto metric : {qalsh::Metric::l1, qalsh::Metric::l2}) {
        CheckBuildParity(Config(4217, 257, metric));
    }
    CheckBuildFailures();
    auto config = Config(1000, 3);
    ScratchPoints points(config.num_dimensions);
    for (const auto invalid : {0U, 1025U}) {
        config.build_threads = invalid;
        bool rejected = false;
        try {
            (void)qalsh::InMemoryIndex::Build(config, points.BuildAccessor());
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        Check(rejected);
    }
}
