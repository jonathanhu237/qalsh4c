#include "point_set.h"

#include <spdlog/spdlog.h>

void PointSet::Init(const std::filesystem::path &file_path, unsigned int num_points, unsigned int num_dimensions) {
    num_points_ = num_points;
    num_dimensions_ = num_dimensions;

    ifs_.open(file_path, std::ios::binary);
    if (!ifs_.is_open()) {
        throw std::runtime_error("Could not open point set file: " + file_path.string());
    }
}

Point PointSet::ReadPoint(unsigned int point_id) {
    ifs_.seekg(static_cast<std::streamoff>(static_cast<unsigned long>(point_id * num_dimensions_) * sizeof(Coordinate)),
               std::ios::beg);
    Point point(num_dimensions_);
    ifs_.read(reinterpret_cast<char *>(point.data()),
              static_cast<std::streamoff>(num_dimensions_ * sizeof(Coordinate)));
    return point;
}

std::vector<Point> PointSet::LoadPointsFromFile(const std::filesystem::path &file_path, unsigned int num_points,
                                                unsigned int num_dimensions) {
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) {
        spdlog::error("Could not open base points file: {}", file_path.string());
        return {};
    }

    unsigned int total_doubles = num_points * num_dimensions;
    std::vector<double> buffer(total_doubles);
    ifs.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamoff>(total_doubles * sizeof(Coordinate)));

    std::vector<Point> points;
    points.reserve(num_points);
    for (unsigned int i = 0; i < num_points; ++i) {
        auto start_offset = i * num_dimensions;
        auto end_offset = start_offset + num_dimensions;
        auto start_it = buffer.begin() + start_offset;
        auto end_it = buffer.begin() + end_offset;
        points.emplace_back(start_it, end_it);
    }

    return points;
}