#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <cstdint>
#include <string_view>
#include <type_traits>

namespace qalsh_chamfer {

enum class DataType : uint8_t { kInt8, kInt16, kInt32, kInt64, kFloat, kDouble };

template <typename T>
constexpr auto GetDataType() -> DataType {
    if constexpr (std::is_same_v<T, int8_t>) {
        return DataType::kInt8;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return DataType::kInt16;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DataType::kInt32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return DataType::kInt64;
    } else if constexpr (std::is_same_v<T, float>) {
        return DataType::kFloat;
    } else if constexpr (std::is_same_v<T, double>) {
        return DataType::kDouble;
    } else {
        static_assert(sizeof(T) == 0, "GetDataType called with an unsupported type.");
    }
}

constexpr auto DataTypeToString(DataType data_type) -> std::string_view {
    switch (data_type) {
        case DataType::kInt8:
            return "int8";
        case DataType::kInt16:
            return "int16";
        case DataType::kInt32:
            return "int32";
        case DataType::kInt64:
            return "int64";
        case DataType::kFloat:
            return "float";
        case DataType::kDouble:
            return "double";
        default:
            return "unknown";
    }
}

}  // namespace qalsh_chamfer

#endif