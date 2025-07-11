#ifndef POINT_SET_H_
#define POINT_SET_H_

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <vector>

#include "utils.h"

// ---------------------------------------------
// Definition of PointSet-related classes
// ---------------------------------------------

template <typename T>
class PointSetWriter {
   public:
    explicit PointSetWriter(const std::filesystem::path& file_path);
    ~PointSetWriter();

    PointSetWriter(const PointSetWriter&) = delete;
    auto operator=(const PointSetWriter&) -> PointSetWriter& = delete;
    PointSetWriter(PointSetWriter&&) noexcept = default;
    auto operator=(PointSetWriter&&) noexcept -> PointSetWriter& = default;

    auto AddPoint(const std::vector<T>& point) -> void;

   private:
    std::ofstream ofs_;
};

template <typename T>
class PointSetReader {
   public:
    explicit PointSetReader(const std::filesystem::path& file_path, unsigned int num_dimensions);
    ~PointSetReader();

    PointSetReader(const PointSetReader&) = delete;
    auto operator=(const PointSetReader&) -> PointSetReader& = delete;
    PointSetReader(PointSetReader&&) noexcept = default;
    auto operator=(PointSetReader&&) noexcept -> PointSetReader& = default;

    auto GetPoint(unsigned int index) -> std::vector<T>;
    auto CalculateDistance(std::vector<T> query) -> double;

   private:
    std::ifstream ifs_;
    unsigned int num_dimensions_;
};

// ---------------------------------------------
// PointSetWriter Implementation
// ---------------------------------------------

template <typename T>
PointSetWriter<T>::PointSetWriter(const std::filesystem::path& file_path) {
    ofs_.open(file_path, std::ios::binary | std::ios::trunc);
    if (!ofs_.is_open()) {
        throw std::runtime_error(std::format("Failed to open file: {}", file_path.string()));
    }
}

template <typename T>
PointSetWriter<T>::~PointSetWriter() {
    if (ofs_.is_open()) {
        ofs_.close();
    }
}

template <typename T>
auto PointSetWriter<T>::AddPoint(const std::vector<T>& point) -> void {
    ofs_.seekp(0, std::ios::end);
    ofs_.write(reinterpret_cast<const char*>(point.data()), sizeof(T) * point.size());
}

// ---------------------------------------------
// PointSetReader Implementation
// ---------------------------------------------

template <typename T>
PointSetReader<T>::PointSetReader(const std::filesystem::path& file_path, unsigned int num_dimensions)
    : num_dimensions_(num_dimensions) {
    ifs_.open(file_path, std::ios::binary);
    if (!ifs_.is_open()) {
        throw std::runtime_error(std::format("Failed to open file: {}", file_path.string()));
    }
}

template <typename T>
PointSetReader<T>::~PointSetReader() {
    if (ifs_.is_open()) {
        ifs_.close();
    }
}

template <typename T>
auto PointSetReader<T>::GetPoint(unsigned int index) -> std::vector<T> {
    ifs_.seekg(index * sizeof(T) * num_dimensions_, std::ios::beg);
    std::vector<T> point(num_dimensions_);
    ifs_.read(reinterpret_cast<char*>(point.data()), sizeof(T) * num_dimensions_);
    return point;
}

template <typename T>
auto PointSetReader<T>::CalculateDistance(std::vector<T> query) -> double {
    if (query.size() != num_dimensions_) {
        throw std::invalid_argument("Query point dimensions do not match the set dimensions.");
    }

    double distance = std::numeric_limits<double>::max();
    std::vector<T> point(num_dimensions_);

    ifs_.seekg(0, std::ios::beg);
    while (ifs_.read(reinterpret_cast<char*>(point.data()), sizeof(T) * num_dimensions_)) {
        double curr_distance = Utils::CalculateL1Distance(point, query);
        distance = std::min(curr_distance, distance);
    }

    return distance;
}

#endif