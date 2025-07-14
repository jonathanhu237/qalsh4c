#ifndef TYPES_H_
#define TYPES_H_

#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

template <typename T>
using Point = std::vector<T>;

using PointVariant = std::variant<Point<uint8_t>, Point<int>, Point<double>>;

struct DatasetMetadata {
    std::string data_type_;
    unsigned int base_num_points_{0};
    unsigned int query_num_points_{0};
    unsigned int num_dimensions_{0};
    double chamfer_distance_{0.0};

    auto Save(const std::filesystem::path& file_path) const -> void;
    auto Load(const std::filesystem::path& file_path) -> void;
};

#endif