#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <cstdint>
#include <map>
#include <string>
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

const std::map<std::string, DataType> DataTypeMap = {
    {"int8", DataType::kInt8},   {"int16", DataType::kInt16}, {"int32", DataType::kInt32},
    {"int64", DataType::kInt64}, {"float", DataType::kFloat}, {"double", DataType::kDouble},
};

}  // namespace qalsh_chamfer

#endif