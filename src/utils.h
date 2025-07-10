#ifndef UTILS_H_
#define UTILS_H_

#include <cstdint>
#include <string>

class Utils {
   public:
    template <typename T>
    static auto to_string() -> std::string {
        if constexpr (std::is_same_v<T, uint8_t>) {
            return "uint8";
        } else if constexpr (std::is_same_v<T, int>) {
            return "int";
        } else if constexpr (std::is_same_v<T, double>) {
            return "double";
        } else {
            return "unknown";
        }
    }
};

#endif