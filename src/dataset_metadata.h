#ifndef DATASET_METADATA_H_
#define DATASET_METADATA_H_

#include <filesystem>
#include <string_view>

struct DatasetMetadata {
    std::string_view data_type_;
    unsigned int base_num_points_{0};
    unsigned int query_num_points_{0};
    unsigned int num_dimensions_{0};
    double chamfer_distance_{0.0};

    auto Save(const std::filesystem::path& file_path) const -> void;
    auto Load(const std::filesystem::path& file_path) -> void;
};

#endif