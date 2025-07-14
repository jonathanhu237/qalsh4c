#ifndef UTILS_H_
#define UTILS_H_

#include <toml++/toml.h>

#include <Eigen/Eigen>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string_view>
#include <vector>

class Utils {
   public:
    template <typename T>
    static constexpr auto to_string() -> std::string_view;

    template <typename T>
    static auto CalculateL1Distance(const std::vector<T> &vec1, const std::vector<T> &vec2) -> double;

    template <typename T>
    static auto GetValueFromTomlTable(const toml::table &tbl, std::string_view key) -> T;

    template <typename T1, typename T2>
    static auto DotProduct(const std::vector<T1> &vec1, const std::vector<T2> &vec2) -> double;

    template <typename T>
    auto static WriteToBuffer(std::vector<char> &buffer, size_t &offset, const T &data) -> void;

    template <typename T>
    auto static ReadFromBuffer(const std::vector<char> &buffer, size_t &offset) -> T;
};

template <typename T>
constexpr auto Utils::to_string() -> std::string_view {
    if constexpr (std::is_same_v<T, uint8_t>) {
        return "uint8";
    } else if constexpr (std::is_same_v<T, int>) {
        return "int";
    } else if constexpr (std::is_same_v<T, double>) {
        return "double";
    } else {
        static_assert(sizeof(T) == 0, "Unsupported type for Utils::to_string");
    }
}

template <typename T>
auto Utils::CalculateL1Distance(const std::vector<T> &vec1, const std::vector<T> &vec2) -> double {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }

    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> eigen_vec1(vec1.data(), vec1.size());
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> eigen_vec2(vec2.data(), vec2.size());

    return (eigen_vec1 - eigen_vec2).template lpNorm<1>();
}

template <typename T>
auto Utils::GetValueFromTomlTable(const toml::table &tbl, std::string_view key) -> T {
    auto value = tbl.get(key)->value<T>();
    if (!value) {
        throw std::runtime_error(std::format("Key '{}' not found in TOML table", key));
    }
    return *value;
}
template <typename T1, typename T2>
auto Utils::DotProduct(const std::vector<T1> &vec1, const std::vector<T2> &vec2) -> double {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must be of the same size for dot product.");
    }

    Eigen::Map<const Eigen::Matrix<T1, Eigen::Dynamic, 1>> eigen_vec1(vec1.data(), vec1.size());
    Eigen::Map<const Eigen::Matrix<T2, Eigen::Dynamic, 1>> eigen_vec2(vec2.data(), vec2.size());

    return eigen_vec1.dot(eigen_vec2);
}

template <typename T>
auto Utils::WriteToBuffer(std::vector<char> &buffer, size_t &offset, const T &data) -> void {
    if (offset + sizeof(T) > buffer.size()) {
        throw std::out_of_range("Not enough space in buffer to write.");
    }
    std::memcpy(&buffer[offset], &data, sizeof(T));
    offset += sizeof(T);
}

template <typename T>
auto Utils::ReadFromBuffer(const std::vector<char> &buffer, size_t &offset) -> T {
    if (offset + sizeof(T) > buffer.size()) {
        throw std::out_of_range("Not enough data in buffer to read.");
    }

    T data;
    std::memcpy(&data, &buffer[offset], sizeof(T));
    offset += sizeof(T);

    return data;
}

#endif