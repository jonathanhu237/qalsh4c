#ifndef UTILS_H_
#define UTILS_H_

#include <cstdint>
#include <string_view>

class Utils {
   public:
    template <typename T>
    static constexpr auto to_string() -> std::string_view;
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

#endif