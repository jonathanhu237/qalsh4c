#ifndef UTILS_H_
#define UTILS_H_

#include <Eigen/Eigen>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

class Utils {
   public:
    template <typename T>
    static constexpr auto to_string() -> std::string_view;

    template <typename T>
    static auto CalculateL1Distance(const std::vector<T> &vector1, const std::vector<T> &vector2) -> double;
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
auto Utils::CalculateL1Distance(const std::vector<T> &vector1, const std::vector<T> &vector2) -> double {
    if (vector1.size() != vector2.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }

    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> eigen_vec1(vector1.data(), vector1.size());
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> eigen_vec2(vector2.data(), vector2.size());

    return (eigen_vec1 - eigen_vec2).template lpNorm<1>();
}

#endif