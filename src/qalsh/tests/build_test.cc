#include <qalsh/qalsh.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
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

std::vector<char> Bytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    Check(input.is_open());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

// Every call overwrites the same storage. Retaining borrowed point views or
// invoking this accessor from a worker violates the supported build contract.
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

    std::size_t calls{0};

   private:
    const std::thread::id owner_{std::this_thread::get_id()};
    std::vector<float> point_;
};

qalsh::IndexConfig Config(std::uint32_t num_points, std::uint32_t dimensions,
                           qalsh::Metric metric = qalsh::Metric::l2) {
    qalsh::IndexConfig config;
    config.metric = metric;
    config.num_points = num_points;
    config.num_dimensions = dimensions;
    config.qalsh.num_hash_tables = 9;
    config.qalsh.collision_threshold = 3;
    config.qalsh.bucket_width = 2;
    return config;
}

std::vector<qalsh::ProjectionHit> TableHits(const qalsh::ProjectionIndex& index, std::uint32_t table) {
    auto cursor = index.make_cursor(table, 0);
    std::vector<qalsh::ProjectionHit> hits;
    const auto report = index.scan(*cursor, std::numeric_limits<float>::max(),
                                   static_cast<std::size_t>(index.metadata().num_points) * 2, &hits,
                                   [](void* context, const qalsh::ProjectionHit& hit) {
                                       static_cast<std::vector<qalsh::ProjectionHit>*>(context)->push_back(hit);
                                   });
    Check(report.table_exhausted && hits.size() == index.metadata().num_points);
    return hits;
}

void CheckMemoryIndexes(const std::shared_ptr<qalsh::InMemoryIndex>& serial,
                         const std::shared_ptr<qalsh::InMemoryIndex>& parallel, ScratchPoints& points) {
    Check(std::ranges::equal(serial->projection_vectors(), parallel->projection_vectors()));
    for (std::uint32_t table = 0; table < serial->metadata().num_hash_tables; ++table) {
        const auto first = TableHits(*serial, table);
        const auto second = TableHits(*parallel, table);
        Check(first.size() == second.size());
        for (std::size_t i = 0; i < first.size(); ++i) {
            Check(first[i].point_id == second[i].point_id &&
                  first[i].projected_value == second[i].projected_value);
        }
    }
    const qalsh::PointAccessor access = [&](qalsh::PointId id) { return points.Read(id); };
    qalsh::SearchEngine first(serial, access), second(parallel, access);
    const auto k = std::min(5U, serial->metadata().num_points);
    for (unsigned i = 0; i < 20; ++i) {
        const auto point = points.Read(i);
        std::vector<float> query(point.begin(), point.end());
        query.front() += .1F;
        qalsh::DefaultQalshStrategy a, b;
        const auto x = first.search(query, k, a), y = second.search(query, k, b);
        Check(x.projection_hits == y.projection_hits && x.neighbors.size() == y.neighbors.size());
        for (std::size_t j = 0; j < x.neighbors.size(); ++j) {
            Check(x.neighbors[j].point_id == y.neighbors[j].point_id &&
                  x.neighbors[j].distance == y.neighbors[j].distance);
        }
    }
}

void CheckBuildParity(const std::filesystem::path& directory, const std::string& name,
                       qalsh::IndexConfig config) {
    ScratchPoints points(config.num_dimensions);
    auto serial = qalsh::InMemoryIndex::Build(config, points.BuildAccessor());
    Check(points.calls == config.num_points);
    config.build_threads = 4;
    auto parallel = qalsh::InMemoryIndex::Build(config, points.BuildAccessor());
    Check(points.calls == config.num_points);
    CheckMemoryIndexes(serial, parallel, points);

    for (const auto layout : {qalsh::IndexLayout::b_plus_tree, qalsh::IndexLayout::sorted_array}) {
        const std::string prefix = name + "-" + std::to_string(static_cast<unsigned>(layout));
        const std::string serial_path = (directory / (prefix + "-serial.qalsh")).string();
        const std::string parallel_path = (directory / (prefix + "-parallel.qalsh")).string();
        const qalsh::PersistentBuildOptions options{.layout = layout, .page_size = 512, .overwrite = true};
        config.build_threads = 1;
        qalsh::PersistentIndex::Build(serial_path, config, points.BuildAccessor(), options);
        Check(points.calls == config.num_points);
        config.build_threads = 4;
        qalsh::PersistentIndex::Build(parallel_path, config, points.BuildAccessor(), options);
        Check(points.calls == config.num_points);
        Check(Bytes(serial_path) == Bytes(parallel_path));
        std::filesystem::remove(serial_path);
        std::filesystem::remove(parallel_path);
    }
}

void CheckBuildFailures(const std::filesystem::path& directory) {
    auto config = Config(4217, 257);
    config.build_threads = 4;
    ScratchPoints points(config.num_dimensions);
    for (const auto layout : {qalsh::IndexLayout::in_memory, qalsh::IndexLayout::b_plus_tree,
                              qalsh::IndexLayout::sorted_array}) {
        const std::string path = (directory / ("failed-build-" +
                                                std::to_string(static_cast<unsigned>(layout)) + ".qalsh"))
                                     .string();
        const auto build = [&](const qalsh::IndexConfig& build_config, const qalsh::PointAccessor& accessor) {
            if (layout == qalsh::IndexLayout::in_memory) {
                (void)qalsh::InMemoryIndex::Build(build_config, accessor);
            } else {
                qalsh::PersistentIndex::Build(
                    path, build_config, accessor,
                    qalsh::PersistentBuildOptions{.layout = layout, .page_size = 512, .overwrite = true});
            }
        };
        build(config, points.BuildAccessor());
        const auto original = layout == qalsh::IndexLayout::in_memory ? std::vector<char>{} : Bytes(path);
        const auto check_preserved = [&] {
            if (layout != qalsh::IndexLayout::in_memory) Check(Bytes(path) == original);
        };

        // Fail on the owning thread after at least one completed point batch.
        const auto ordered_accessor = points.BuildAccessor();
        bool rejected = false;
        try {
            build(config, [&](qalsh::PointId id) -> qalsh::PointView {
                const auto point = ordered_accessor(id);
                if (id == 4100) throw std::runtime_error("accessor failure");
                return point;
            });
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()) == "accessor failure";
        }
        Check(rejected && points.calls == 4101);
        check_preserved();

        const auto invalid_accessor = points.BuildAccessor();
        rejected = false;
        try {
            build(config, [&](qalsh::PointId id) {
                const auto point = invalid_accessor(id);
                return id == 4100 ? point.first(point.size() - 1) : point;
            });
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        Check(rejected && points.calls == 4101);
        check_preserved();

        // Finite inputs whose projection overflows fail inside projection work.
        // The worker exception must reach the caller without terminating or
        // leaving workers waiting for another batch.
        auto overflowing = config;
        overflowing.projection_vectors.assign(
            static_cast<std::size_t>(config.num_dimensions) * config.qalsh.num_hash_tables, 1.0F);
        const std::vector<float> overflow_point(config.num_dimensions, std::numeric_limits<float>::max());
        const auto overflow_accessor = points.BuildAccessor();
        rejected = false;
        try {
            build(overflowing, [&](qalsh::PointId id) -> qalsh::PointView {
                const auto point = overflow_accessor(id);
                return id == 3000 ? qalsh::PointView(overflow_point) : point;
            });
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        Check(rejected && points.calls > 0 && points.calls <= config.num_points);
        check_preserved();

        build(config, points.BuildAccessor());
        Check(points.calls == config.num_points);
        check_preserved();
        if (layout != qalsh::IndexLayout::in_memory) std::filesystem::remove(path);
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::filesystem::path directory(argv[1]);
    CheckBuildParity(directory, "tiny", Config(3, 1));
    CheckBuildParity(directory, "small", Config(1000, 3));
    // More than one copied batch, an uneven tail, and SIMD dimension tails.
    for (const auto metric : {qalsh::Metric::l1, qalsh::Metric::l2}) {
        CheckBuildParity(directory, "batched-" + std::to_string(static_cast<unsigned>(metric)),
                           Config(4217, 257, metric));
    }
    CheckBuildFailures(directory);
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
