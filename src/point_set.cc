#include "point_set.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <fcntl.h>
#include <limits>
#include <sys/uio.h>
#include <unistd.h>

#include "log.h"

void PointSet::Init(const PointSetMetadata &metadata) {
    num_points_ = metadata.num_points;
    num_dimensions_ = metadata.num_dimensions;

    if (ifs_.is_open()) {
        ifs_.close();
    }

    ifs_.open(metadata.file_path, std::ios::binary);
    if (!ifs_.is_open()) {
        Log::error("Could not open point set file: {}", metadata.file_path.string());
    }
}

Point PointSet::ReadPoint(unsigned int point_id) {
    Point point(num_dimensions_);
    ReadPointInto(point_id, point);
    return point;
}

void PointSet::ReadPointInto(unsigned int point_id, Point& out) {
    SeekPoint(point_id);
    if (!ReadNextPointInto(out)) {
        Log::error("Failed to read point {}.", point_id);
    }
}

void PointSet::SeekPoint(unsigned int point_id) {
    const auto offset = static_cast<std::uint64_t>(point_id) * static_cast<std::uint64_t>(num_dimensions_) *
                        static_cast<std::uint64_t>(sizeof(Coordinate));
    ifs_.clear();
    ifs_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
}

bool PointSet::ReadNextPointInto(Point& out) {
    if (out.size() != num_dimensions_) {
        out.resize(num_dimensions_);
    }
    ifs_.read(reinterpret_cast<char *>(out.data()),
              static_cast<std::streamsize>(static_cast<size_t>(num_dimensions_) * sizeof(Coordinate)));
    return static_cast<bool>(ifs_);
}

namespace {

// Reads `count` rows from `offset` into the buffers of `rows`, retrying partial reads; false at end of file or on an
// error.
bool ReadRows(int fd, iovec* rows, std::size_t count, std::uint64_t offset) {
    std::size_t first = 0;
    while (first < count) {
        const int batch = static_cast<int>(std::min<std::size_t>(count - first, IOV_MAX));
        const ssize_t bytes = ::preadv(fd, rows + first, batch, static_cast<off_t>(offset));
        if (bytes < 0 && errno == EINTR) {
            continue;
        }
        if (bytes <= 0) {
            return false;
        }
        offset += static_cast<std::uint64_t>(bytes);
        auto remaining = static_cast<std::size_t>(bytes);
        while (first < count && remaining >= rows[first].iov_len) {
            remaining -= rows[first].iov_len;
            ++first;
        }
        if (remaining != 0) {
            rows[first].iov_base = static_cast<char*>(rows[first].iov_base) + remaining;
            rows[first].iov_len -= remaining;
        }
    }
    return true;
}

}  // namespace

std::vector<Point> PointSet::LoadPointsFromFile(const std::filesystem::path &file_path, unsigned int num_points,
                                                unsigned int num_dimensions) {
    const int fd = ::open(file_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        Log::error("Could not open base points file: {}", file_path.string());
        return {};
    }

    // Threads read separate runs of points with one scatter read per run, straight into the points' own buffers.
    std::vector<Point> points(num_points);
    const std::size_t row_bytes = static_cast<std::size_t>(num_dimensions) * sizeof(Coordinate);
    constexpr std::size_t kRowsPerRead = 256;
    const auto num_runs = static_cast<long long>((static_cast<std::size_t>(num_points) + kRowsPerRead - 1) / kRowsPerRead);
    std::atomic<unsigned long long> first_failure{std::numeric_limits<unsigned long long>::max()};
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (long long run = 0; run < num_runs; ++run) {
        const std::size_t begin = static_cast<std::size_t>(run) * kRowsPerRead;
        const std::size_t end = std::min<std::size_t>(begin + kRowsPerRead, num_points);
        std::array<iovec, kRowsPerRead> rows{};
        for (std::size_t point = begin; point < end; ++point) {
            points[point].resize(num_dimensions);
            rows[point - begin] = iovec{.iov_base = points[point].data(), .iov_len = row_bytes};
        }
        if (row_bytes != 0 && !ReadRows(fd, rows.data(), end - begin, static_cast<std::uint64_t>(begin) * row_bytes)) {
            unsigned long long expected = first_failure.load();
            while (begin < expected && !first_failure.compare_exchange_weak(expected, begin)) {
            }
        }
    }
    ::close(fd);
    if (first_failure.load() != std::numeric_limits<unsigned long long>::max()) {
        Log::error("Failed to read point {} from {}.", first_failure.load(), file_path.string());
    }
    return points;
}
