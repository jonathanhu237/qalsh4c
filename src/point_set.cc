#include "point_set.h"

#include <fstream>
#include <ios>
#include <utility>

// ---------------------------------------------
// InMemoryPointSetReader Implementation
// ---------------------------------------------
InMemoryPointSetReader::InMemoryPointSetReader(const PointSetMetadata& point_set_metadata)
    : num_points_(point_set_metadata.num_points), num_dimensions_(point_set_metadata.num_dimensions) {
    std::ifstream ifs(point_set_metadata.file_path, std::ios::binary);
    if (!ifs.is_open()) {
        spdlog::error(std::format("Failed to open file: {}", point_set_metadata.file_path.string()));
    }

    points_.resize(num_points_);
    for (unsigned int i = 0; i < num_points_; ++i) {
        points_[i].resize(num_dimensions_);
        ifs.read(reinterpret_cast<char*>(points_[i].data()),
                 static_cast<std::streamoff>(sizeof(Coordinate)) * num_dimensions_);
    }
}

unsigned int InMemoryPointSetReader::get_num_points() const { return num_points_; }

unsigned int InMemoryPointSetReader::get_num_dimensions() const { return num_dimensions_; }

Point InMemoryPointSetReader::GetPoint(unsigned int index) { return points_.at(index); }

// ---------------------------------------------
// DiskPointSetReader Implementation
// ---------------------------------------------
DiskPointSetReader::DiskPointSetReader(const PointSetMetadata& point_set_metadata)
    : num_points_(point_set_metadata.num_points), num_dimensions_(point_set_metadata.num_dimensions) {
    ifs_.open(point_set_metadata.file_path, std::ios::binary);
    if (!ifs_.is_open()) {
        spdlog::error(std::format("Failed to open file: {}", point_set_metadata.file_path.string()));
    }
}

unsigned int DiskPointSetReader::get_num_points() const { return num_points_; }

unsigned int DiskPointSetReader::get_num_dimensions() const { return num_dimensions_; }

Point DiskPointSetReader::GetPoint(unsigned int index) {
    ifs_.seekg(index * static_cast<std::streamoff>(sizeof(Coordinate)) * num_dimensions_, std::ios::beg);
    Point point(num_dimensions_);

    ifs_.read(reinterpret_cast<char*>(point.data()), static_cast<std::streamoff>(sizeof(Coordinate)) * num_dimensions_);

    return point;
}

// ---------------------------------------------
// InMemoryPointSetWriter Implementation
// ---------------------------------------------
InMemoryPointSetWriter::InMemoryPointSetWriter(std::filesystem::path file_path) : file_path_(std::move(file_path)) {}

void InMemoryPointSetWriter::AddPoint(const Point& point) { points_.emplace_back(point); }

void InMemoryPointSetWriter::Flush() {
    std::ofstream ofs(file_path_);
    for (auto point : points_) {
        ofs.write(reinterpret_cast<const char*>(point.data()),
                  static_cast<std::streamoff>(sizeof(Coordinate) * point.size()));
    }
}

// ---------------------------------------------
// DiskPointSetWriter Implementation
// ---------------------------------------------
DiskPointSetWriter::DiskPointSetWriter(const std::filesystem::path& file_path) {
    ofs_.open(file_path, std::ios::binary | std::ios::trunc);
}

void DiskPointSetWriter::AddPoint(const Point& point) {
    ofs_.write(reinterpret_cast<const char*>(point.data()),
               static_cast<std::streamoff>(sizeof(Coordinate) * point.size()));
}

void DiskPointSetWriter::Flush() { ofs_.flush(); }