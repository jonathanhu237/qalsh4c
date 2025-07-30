#include "point_set.h"

#include <ios>

// ---------------------------------------------
// InMemoryPointSetReader Implementation
// ---------------------------------------------
InMemoryPointSet::InMemoryPointSet(const PointSetMetadata& point_set_metadata)
    : num_points_(point_set_metadata.num_points), num_dimensions_(point_set_metadata.num_dimensions) {
    std::ifstream ifs(point_set_metadata.file_path, std::ios::binary);
    if (!ifs.is_open()) {
        spdlog::error(std::format("Failed to open file: {}", point_set_metadata.file_path.string()));
    }

    points_.resize(num_points_);
    for (unsigned int i = 0; i < num_points_; ++i) {
        points_[i].resize(num_dimensions_);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        ifs.read(reinterpret_cast<char*>(points_[i].data()),
                 static_cast<std::streamoff>(sizeof(Coordinate)) * num_dimensions_);
    }
}

unsigned int InMemoryPointSet::get_num_points() const { return num_points_; }

unsigned int InMemoryPointSet::get_num_dimensions() const { return num_dimensions_; }

Point InMemoryPointSet::GetPoint(unsigned int index) { return points_.at(index); }

// ---------------------------------------------
// DiskPointSetReader Implementation
// ---------------------------------------------
DiskPointSet::DiskPointSet(const PointSetMetadata& point_set_metadata)
    : num_points_(point_set_metadata.num_points), num_dimensions_(point_set_metadata.num_dimensions) {
    ifs_.open(point_set_metadata.file_path, std::ios::binary);
    if (!ifs_.is_open()) {
        spdlog::error(std::format("Failed to open file: {}", point_set_metadata.file_path.string()));
    }
}

unsigned int DiskPointSet::get_num_points() const { return num_points_; }

unsigned int DiskPointSet::get_num_dimensions() const { return num_dimensions_; }

Point DiskPointSet::GetPoint(unsigned int index) {
    ifs_.seekg(index * static_cast<std::streamoff>(sizeof(Coordinate)) * num_dimensions_, std::ios::beg);
    Point point(num_dimensions_);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    ifs_.read(reinterpret_cast<char*>(point.data()), static_cast<std::streamoff>(sizeof(Coordinate)) * num_dimensions_);

    return point;
}