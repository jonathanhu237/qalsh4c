#ifndef POINT_SET_H_
#define POINT_SET_H_

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "utils.h"

template <typename T>
using Point = std::vector<T>;

using PointVariant = std::variant<Point<uint8_t>, Point<int>, Point<double>>;

// ---------------------------------------------
// Definition of PointSet-related classes
// ---------------------------------------------

class IPointSetWriter {
   public:
    virtual ~IPointSetWriter() = default;
    virtual auto AddPoint(const PointVariant& point) -> void = 0;
};

template <typename T>
class PointSetWriter : public IPointSetWriter {
   public:
    explicit PointSetWriter(const std::filesystem::path& file_path, unsigned int num_dimensions);
    ~PointSetWriter();

    PointSetWriter(const PointSetWriter&) = delete;
    auto operator=(const PointSetWriter&) -> PointSetWriter& = delete;
    PointSetWriter(PointSetWriter&&) noexcept = default;
    auto operator=(PointSetWriter&&) noexcept -> PointSetWriter& = default;

    auto AddPoint(const PointVariant& point) -> void override;

   private:
    std::ofstream ofs_;
    unsigned int num_dimensions_;
};

class PointSetWriterFactory {
   public:
    static auto Create(const std::string& data_type, const std::filesystem::path& file_path,
                       unsigned int num_dimensions) -> std::unique_ptr<IPointSetWriter> {
        if (data_type == "uint8") {
            return std::unique_ptr<IPointSetWriter>(new PointSetWriter<uint8_t>(file_path, num_dimensions));
        }
        if (data_type == "int") {
            return std::unique_ptr<IPointSetWriter>(new PointSetWriter<int>(file_path, num_dimensions));
        }
        if (data_type == "double") {
            return std::unique_ptr<IPointSetWriter>(new PointSetWriter<double>(file_path, num_dimensions));
        }

        throw std::invalid_argument(std::format("Unsupported data type: {}", data_type));
    }
};

class IPointSetReader {
   public:
    virtual ~IPointSetReader() = default;
    virtual auto GetPoint(unsigned int index) -> PointVariant = 0;
    virtual auto CalculateDistance(const PointVariant& query) -> double = 0;
};

template <typename T>
class PointSetReader : public IPointSetReader {
   public:
    explicit PointSetReader(const std::filesystem::path& file_path, unsigned int num_points,
                            unsigned int num_dimensions);
    ~PointSetReader();

    PointSetReader(const PointSetReader&) = delete;
    auto operator=(const PointSetReader&) -> PointSetReader& = delete;
    PointSetReader(PointSetReader&&) noexcept = default;
    auto operator=(PointSetReader&&) noexcept -> PointSetReader& = default;

    auto GetPoint(unsigned int index) -> PointVariant override;
    auto CalculateDistance(const PointVariant& query) -> double override;

   private:
    std::ifstream ifs_;
    unsigned int num_points_;
    unsigned int num_dimensions_;
};

class PointSetReaderFactory {
   public:
    static auto Create(const std::string& data_type, const std::filesystem::path& file_path, unsigned int num_points,
                       unsigned int num_dimensions) -> std::unique_ptr<IPointSetReader> {
        if (data_type == "uint8") {
            return std::unique_ptr<IPointSetReader>(new PointSetReader<uint8_t>(file_path, num_points, num_dimensions));
        }
        if (data_type == "int") {
            return std::unique_ptr<IPointSetReader>(new PointSetReader<int>(file_path, num_points, num_dimensions));
        }
        if (data_type == "double") {
            return std::unique_ptr<IPointSetReader>(new PointSetReader<double>(file_path, num_points, num_dimensions));
        }

        throw std::invalid_argument(std::format("Unsupported data type: {}", data_type));
    }
};

// ---------------------------------------------
// PointSetWriter Implementation
// ---------------------------------------------

template <typename T>
PointSetWriter<T>::PointSetWriter(const std::filesystem::path& file_path, unsigned int num_dimensions)
    : num_dimensions_(num_dimensions) {
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
auto PointSetWriter<T>::AddPoint(const PointVariant& point) -> void {
    const auto& concrete_point = std::get<Point<T>>(point);

    if (concrete_point.size() != num_dimensions_) {
        throw std::invalid_argument(std::format("Point dimensions do not match the set dimensions: expected {}, got {}",
                                                num_dimensions_, concrete_point.size()));
    }

    ofs_.seekp(0, std::ios::end);
    ofs_.write(reinterpret_cast<const char*>(concrete_point.data()),
               static_cast<std::streamsize>(sizeof(T) * concrete_point.size()));
}

// ---------------------------------------------
// PointSetReader Implementation
// ---------------------------------------------

template <typename T>
PointSetReader<T>::PointSetReader(const std::filesystem::path& file_path, unsigned int num_points,
                                  unsigned int num_dimensions)
    : num_points_(num_points), num_dimensions_(num_dimensions) {
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
auto PointSetReader<T>::GetPoint(unsigned int index) -> PointVariant {
    ifs_.seekg(index * sizeof(T) * num_dimensions_, std::ios::beg);
    Point<T> point(num_dimensions_);
    ifs_.read(reinterpret_cast<char*>(point.data()), sizeof(T) * num_dimensions_);
    return point;
}

template <typename T>
auto PointSetReader<T>::CalculateDistance(const PointVariant& query) -> double {
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