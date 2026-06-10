#include "point_set.h"

#include <cstdint>

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

std::vector<Point> PointSet::LoadPointsFromFile(const std::filesystem::path &file_path, unsigned int num_points,
                                                unsigned int num_dimensions) {
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs.is_open()) {
        Log::error("Could not open base points file: {}", file_path.string());
        return {};
    }

    std::vector<Point> points;
    points.reserve(num_points);
    for (unsigned int i = 0; i < num_points; ++i) {
        Point point(num_dimensions);
        ifs.read(reinterpret_cast<char *>(point.data()),
                 static_cast<std::streamsize>(static_cast<size_t>(num_dimensions) * sizeof(Coordinate)));
        if (!ifs) {
            Log::error("Failed to read point {} from {}.", i, file_path.string());
        }
        points.push_back(std::move(point));
    }

    return points;
}
