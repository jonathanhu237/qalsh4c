#ifndef DATASET_METADATA_H_
#define DATASET_METADATA_H_

#include <filesystem>
#include <string_view>

struct DatasetMetadata {
    unsigned int base_num_points_{0};
    unsigned int query_num_points_{0};
    unsigned int num_dimensions_{0};
    std::string_view data_type_;

    auto Save(const std::filesystem::path& file_path) const -> void;
    auto Load(const std::filesystem::path& file_path) -> void;
};

#endif