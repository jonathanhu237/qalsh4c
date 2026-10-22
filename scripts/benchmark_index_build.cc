// Compile this same source separately against each version's public headers
// and static library. benchmark_index_build.py records both compile commands.
#include <qalsh/qalsh.h>

#include <bit>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <vector>

namespace {

std::uint32_t ParseInteger(const char* argument, std::uint32_t minimum,
                           std::uint32_t maximum, std::string_view name) {
    const std::string_view text(argument);
    std::uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() ||
        value < minimum || value > maximum) {
        throw std::invalid_argument("invalid " + std::string(name));
    }
    return value;
}

std::uint64_t ResidentKiB() {
    // smaps_rollup reports resident pages directly; VmRSS in /proc/self/status
    // can lag because Linux batches updates to some process memory counters.
    std::ifstream memory("/proc/self/smaps_rollup");
    if (!memory) throw std::runtime_error("cannot read /proc/self/smaps_rollup");
    std::string line;
    while (std::getline(memory, line)) {
        if (!line.starts_with("Rss:")) continue;
        std::istringstream values(line);
        std::string key, unit;
        std::uint64_t resident = 0;
        if (!(values >> key >> resident >> unit) || unit != "kB") {
            throw std::runtime_error("invalid Linux Rss record");
        }
        return resident;
    }
    throw std::runtime_error("Linux smaps_rollup has no Rss record");
}

std::uint64_t PeakKiB() {
    rusage usage{};
    if (::getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) {
        throw std::runtime_error("cannot read process peak RSS");
    }
    // Linux reports ru_maxrss in KiB. Each invocation builds only one index.
    return static_cast<std::uint64_t>(usage.ru_maxrss);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 7) {
            throw std::invalid_argument("usage: index-build B.bin points dimensions norm threads seed");
        }
#ifndef __linux__
        throw std::runtime_error("index build RSS measurements require Linux");
#endif
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
        if (std::endian::native != std::endian::little) {
            throw std::runtime_error("canonical float32 inputs require a little-endian host");
        }
        const auto count = ParseInteger(argv[2], 1, 1U << 30U, "point count");
        const auto dimensions = ParseInteger(argv[3], 1, 1U << 20U, "dimension count");
        const auto norm = ParseInteger(argv[4], 1, 2, "norm");
        const auto threads = ParseInteger(argv[5], 1, 1024, "build thread count");
        const auto seed = ParseInteger(argv[6], 0, std::numeric_limits<std::uint32_t>::max(), "seed");
        const std::uint64_t input_bytes = static_cast<std::uint64_t>(count) * dimensions * sizeof(float);
        if (std::filesystem::file_size(argv[1]) != input_bytes) {
            throw std::invalid_argument("B.bin size does not match float32 point metadata");
        }

        // Match the application's immutable vector-of-points accessor. Loading
        // and allocation happen before the build timer and prebuild RSS sample.
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open B.bin");
        std::vector<std::vector<float>> points;
        points.reserve(count);
        for (std::uint32_t point = 0; point < count; ++point) {
            points.emplace_back(dimensions);
            input.read(reinterpret_cast<char*>(points.back().data()),
                       static_cast<std::streamsize>(static_cast<std::size_t>(dimensions) * sizeof(float)));
            if (!input) throw std::runtime_error("short read from B.bin");
        }
        input.close();

        qalsh::IndexConfig config;
        config.metric = norm == 1 ? qalsh::Metric::l1 : qalsh::Metric::l2;
        config.num_points = count;
        config.num_dimensions = dimensions;
        config.qalsh = qalsh::DeriveQalshParameters(config.metric, count, 2.0F);
        config.seed = seed;
        config.build_threads = threads;
        const qalsh::PointAccessor accessor = [&](qalsh::PointId id) -> qalsh::PointView {
            return points[id];
        };
        const auto prebuild_rss = ResidentKiB();
        const auto prebuild_peak = PeakKiB();
        const auto start = std::chrono::steady_clock::now();
        const auto index = qalsh::InMemoryIndex::Build(config, accessor);
        const auto finish = std::chrono::steady_clock::now();
        const double build_ms = std::chrono::duration<double, std::milli>(finish - start).count();
        const auto steady_rss = ResidentKiB();
        const auto peak_rss = PeakKiB();
        const auto& metadata = index->metadata();

        // Keep both input points and the completed index alive. Do not trim
        // allocator arenas: retained build storage is part of observed RSS.
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"points\":" << count << ",\"dimensions\":" << dimensions
                  << ",\"norm\":" << norm << ",\"threads\":" << threads
                  << ",\"seed\":" << seed << ",\"input_bytes\":" << input_bytes
                  << ",\"tables\":" << metadata.num_hash_tables
                  << ",\"collision_threshold\":" << metadata.qalsh.collision_threshold
                  << ",\"build_ms\":" << build_ms
                  << ",\"prebuild_rss_kib\":" << prebuild_rss
                  << ",\"steady_rss_kib\":" << steady_rss
                  << ",\"rss_delta_kib\":"
                  << static_cast<std::int64_t>(steady_rss) - static_cast<std::int64_t>(prebuild_rss)
                  << ",\"prebuild_peak_rss_kib\":" << prebuild_peak
                  << ",\"peak_rss_kib\":" << peak_rss << "}\n";
    } catch (const std::exception& error) {
        std::cerr << "index build benchmark: " << error.what() << '\n';
        return 1;
    }
}
