#ifndef POINT_SET_H_
#define POINT_SET_H_

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utils.h"

template <typename T>
using Point = std::vector<T>;

using PointVariant = std::variant<Point<uint8_t>, Point<int>, Point<double>>;

// ---------------------------------------------
// PointSetWriter Definition
// ---------------------------------------------

class PointSetWriter {
   public:
    virtual ~PointSetWriter() = default;
    virtual auto AddPoint(const PointVariant& point) -> void = 0;
    virtual auto Flush() -> void = 0;
};

// ---------------------------------------------
// InMemoryPointSetWriter Definition
// ---------------------------------------------

template <typename T>
class InMemoryPointSetWriter : public PointSetWriter {
   public:
    InMemoryPointSetWriter(std::filesystem::path file_path, unsigned int num_dimensions);

    auto AddPoint(const PointVariant& point) -> void override;
    auto Flush() -> void override;

   private:
    std::filesystem::path file_path_;
    unsigned int num_dimensions_;
    std::vector<Point<T>> points_;
};

// ---------------------------------------------
// DiskPointSetWriter Definition
// ---------------------------------------------

template <typename T>
class DiskPointSetWriter : public PointSetWriter {
   public:
    DiskPointSetWriter(const std::filesystem::path& file_path, unsigned int num_dimensions);
    ~DiskPointSetWriter();

    DiskPointSetWriter(const DiskPointSetWriter&) = delete;
    auto operator=(const DiskPointSetWriter&) -> DiskPointSetWriter& = delete;
    DiskPointSetWriter(DiskPointSetWriter&&) noexcept = default;
    auto operator=(DiskPointSetWriter&&) noexcept -> DiskPointSetWriter& = default;

    auto AddPoint(const PointVariant& point) -> void override;
    auto Flush() -> void override;

   private:
    std::ofstream ofs_;
    unsigned int num_dimensions_;
};

// ---------------------------------------------
// PointSetWriterFactory Definition
// ---------------------------------------------
class PointSetWriterFactory {
   public:
    static auto Create(bool in_memory, const std::string& data_type, const std::filesystem::path& file_path,
                       unsigned int num_dimensions) -> std::unique_ptr<PointSetWriter>;
};

// ---------------------------------------------
// PointSetReader Definition
// ---------------------------------------------

class PointSetReader {
   public:
    virtual ~PointSetReader() = default;
    virtual auto GetPoint(unsigned int index) -> PointVariant = 0;
    virtual auto CalculateDistance(const PointVariant& query) -> double = 0;
};

// ---------------------------------------------
// InMemoryPointSetReader Definition
// ---------------------------------------------

template <typename T>
class InMemoryPointSetReader : public PointSetReader {
   public:
    InMemoryPointSetReader(const std::filesystem::path& file_path, unsigned int num_points,
                           unsigned int num_dimensions);

    auto GetPoint(unsigned int index) -> PointVariant override;
    auto CalculateDistance(const PointVariant& query) -> double override;

   private:
    unsigned int num_points_;
    unsigned int num_dimensions_;
    std::vector<Point<T>> points_;
};

// ---------------------------------------------
// DiskPointSetReader Definition
// ---------------------------------------------

template <typename T>
class DiskPointSetReader : public PointSetReader {
   public:
    DiskPointSetReader(const std::filesystem::path& file_path, unsigned int num_points, unsigned int num_dimensions);
    ~DiskPointSetReader();

    DiskPointSetReader(const DiskPointSetReader&) = delete;
    auto operator=(const DiskPointSetReader&) -> DiskPointSetReader& = delete;
    DiskPointSetReader(DiskPointSetReader&&) noexcept = default;
    auto operator=(DiskPointSetReader&&) noexcept -> DiskPointSetReader& = default;

    auto GetPoint(unsigned int index) -> PointVariant override;
    auto CalculateDistance(const PointVariant& query) -> double override;

   private:
    std::ifstream ifs_;
    unsigned int num_points_;
    unsigned int num_dimensions_;
};

// ---------------------------------------------
// PointSetReaderFactory Definition
// ---------------------------------------------
class PointSetReaderFactory {
   public:
    static auto Create(bool in_memory, const std::string& data_type, const std::filesystem::path& file_path,
                       unsigned int num_points, unsigned int num_dimensions) -> std::unique_ptr<PointSetReader>;
};

// ---------------------------------------------
// InMemoryPointSetWriter Implementation
// ---------------------------------------------

template <typename T>
InMemoryPointSetWriter<T>::InMemoryPointSetWriter(std::filesystem::path file_path, unsigned int num_dimensions)
    : file_path_(std::move(file_path)), num_dimensions_(num_dimensions) {}

template <typename T>
auto InMemoryPointSetWriter<T>::AddPoint(const PointVariant& point) -> void {
    const auto& concrete_point = std::get<Point<T>>(point);
    points_.emplace_back(concrete_point);
}

template <typename T>
auto InMemoryPointSetWriter<T>::Flush() -> void {
    std::ofstream ofs(file_path_, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error(std::format("Failed to open file: {}", file_path_.string()));
    }
    for (const auto& point : points_) {
        ofs.write(reinterpret_cast<const char*>(point.data()), static_cast<std::streamsize>(sizeof(T) * point.size()));
    }
}

// ---------------------------------------------
// DiskPointSetWriter Implementation
// ---------------------------------------------

template <typename T>
DiskPointSetWriter<T>::DiskPointSetWriter(const std::filesystem::path& file_path, unsigned int num_dimensions)
    : num_dimensions_(num_dimensions) {
    ofs_.open(file_path, std::ios::binary | std::ios::trunc);
    if (!ofs_.is_open()) {
        throw std::runtime_error(std::format("Failed to open file: {}", file_path.string()));
    }
}

template <typename T>
DiskPointSetWriter<T>::~DiskPointSetWriter() {
    if (ofs_.is_open()) {
        ofs_.close();
    }
}

template <typename T>
auto DiskPointSetWriter<T>::AddPoint(const PointVariant& point) -> void {
    const auto& concrete_point = std::get<Point<T>>(point);

    if (concrete_point.size() != num_dimensions_) {
        throw std::invalid_argument(std::format("Point dimensions do not match the set dimensions: expected {}, got {}",
                                                num_dimensions_, concrete_point.size()));
    }

    ofs_.seekp(0, std::ios::end);
    ofs_.write(reinterpret_cast<const char*>(concrete_point.data()),
               static_cast<std::streamsize>(sizeof(T) * concrete_point.size()));
}

template <typename T>
auto DiskPointSetWriter<T>::Flush() -> void {
    // Do nothing here, as we write directly to the file on each AddPoint call.
}

// ---------------------------------------------
// InMemoryPointSetReader Implementation
// ---------------------------------------------

template <typename T>
InMemoryPointSetReader<T>::InMemoryPointSetReader(const std::filesystem::path& file_path, unsigned int num_points,
                                                  unsigned int num_dimensions)
    : num_points_(num_points), num_dimensions_(num_dimensions) {
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error(std::format("Failed to open file: {}", file_path.string()));
    }

    points_.resize(num_points_);
    for (unsigned int i = 0; i < num_points_; ++i) {
        points_[i].resize(num_dimensions_);
        ifs.read(reinterpret_cast<char*>(points_[i].data()), sizeof(T) * num_dimensions_);
    }
}

template <typename T>
auto InMemoryPointSetReader<T>::GetPoint(unsigned int index) -> PointVariant {
    return points_.at(index);
}

template <typename T>
auto InMemoryPointSetReader<T>::CalculateDistance(const PointVariant& query) -> double {
    double distance = std::numeric_limits<double>::max();
    const auto& concrete_query = std::get<Point<T>>(query);
    for (const auto& point : points_) {
        double curr_distance = Utils::CalculateL1Distance(point, concrete_query);
        distance = std::min(curr_distance, distance);
    }
    return distance;
}

// ---------------------------------------------
// DiskPointSetReader Implementation
// ---------------------------------------------

template <typename T>
DiskPointSetReader<T>::DiskPointSetReader(const std::filesystem::path& file_path, unsigned int num_points,
                                          unsigned int num_dimensions)
    : num_points_(num_points), num_dimensions_(num_dimensions) {
    ifs_.open(file_path, std::ios::binary);
    if (!ifs_.is_open()) {
        throw std::runtime_error(std::format("Failed to open file: {}", file_path.string()));
    }
}

template <typename T>
DiskPointSetReader<T>::~DiskPointSetReader() {
    if (ifs_.is_open()) {
        ifs_.close();
    }
}

template <typename T>
auto DiskPointSetReader<T>::GetPoint(unsigned int index) -> PointVariant {
    ifs_.seekg(index * sizeof(T) * num_dimensions_, std::ios::beg);
    Point<T> point(num_dimensions_);
    ifs_.read(reinterpret_cast<char*>(point.data()), sizeof(T) * num_dimensions_);
    return point;
}

template <typename T>
auto DiskPointSetReader<T>::CalculateDistance(const PointVariant& query) -> double {
    const auto& concrete_query = std::get<Point<T>>(query);

    if (concrete_query.size() != num_dimensions_) {
        throw std::invalid_argument("Query point dimensions do not match the set dimensions.");
    }

    double distance = std::numeric_limits<double>::max();
    Point<T> point(num_dimensions_);

    ifs_.seekg(0, std::ios::beg);
    for (unsigned int i = 0; i < num_points_; i++) {
        ifs_.read(reinterpret_cast<char*>(point.data()), sizeof(T) * num_dimensions_);
        double curr_distance = Utils::CalculateL1Distance(point, concrete_query);
        distance = std::min(curr_distance, distance);
    }

    return distance;
}

#endif