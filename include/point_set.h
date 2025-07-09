#ifndef POINT_SET_H_
#define POINT_SET_H_

#include <filesystem>
#include <format>
#include <fstream>
#include <vector>

#include "data_type.h"

namespace fs = std::filesystem;

namespace qalsh_chamfer {

// ---------- Declaration ----------
template <typename T>
using Point = std::vector<T>;

template <typename T>
class PointSet {
   public:
    PointSet(const fs::path& file_path, unsigned int num_dimensions);  // Create a new point set

    PointSet(const PointSet&) = delete;
    auto operator=(const PointSet&) -> PointSet& = delete;
    PointSet(PointSet&&) noexcept = default;
    auto operator=(PointSet&&) noexcept -> PointSet& = default;

    ~PointSet();

    auto Add(const Point<T>& point) -> void;
    auto Flush() -> void;

   private:
    auto WriteHeader() -> void;

    DataType data_type_;
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};

    std::fstream fs_;
};

// ---------- Implementation ----------
template <typename T>
PointSet<T>::PointSet(const fs::path& file_path, unsigned int num_dimensions)
    : data_type_(GetDataType<T>()), num_dimensions_(num_dimensions) {
    fs_.open(file_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!fs_.is_open()) {
        throw std::runtime_error(std::format("Failed to open file: {}", file_path.string()));
    }
    WriteHeader();
}

template <typename T>
PointSet<T>::~PointSet() {
    if (fs_.is_open()) {
        WriteHeader();
        fs_.close();
    }
}

template <typename T>
auto PointSet<T>::WriteHeader() -> void {
    fs_.seekp(0);
    fs_.write(reinterpret_cast<const char*>(&data_type_), sizeof(data_type_));
    fs_.write(reinterpret_cast<const char*>(&num_points_), sizeof(num_points_));
    fs_.write(reinterpret_cast<const char*>(&num_dimensions_), sizeof(num_dimensions_));
}

template <typename T>
auto PointSet<T>::Add(const Point<T>& point) -> void {
    if (point.size() != num_dimensions_) {
        throw std::runtime_error(std::format("Point dimension does not match the set dimension, expected {}, got {}",
                                             num_dimensions_, point.size()));
    }

    fs_.seekp(0, std::ios::end);
    fs_.write(reinterpret_cast<const char*>(point.data()), sizeof(T) * point.size());
    num_points_++;
}

}  // namespace qalsh_chamfer

#endif