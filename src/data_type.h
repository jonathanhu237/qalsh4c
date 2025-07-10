#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <cstdint>
#include <map>
#include <string>
#include <string_view>

enum class DataType : uint8_t { kUnknown, kUint8, kInt, kDouble };

constexpr auto to_string(DataType data_type) -> std::string_view {
    switch (data_type) {
        case DataType::kUnknown:
            return "unknown";
        case DataType::kUint8:
            return "uint8";
        case DataType::kInt:
            return "int";
        case DataType::kDouble:
            return "double";
    }
}

const std::map<std::string, DataType> data_type_map = {
    {"unknown", DataType::kUnknown},
    {"uint8", DataType::kUint8},
    {"int", DataType::kInt},
    {"double", DataType::kDouble},
};

#endif
