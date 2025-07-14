#ifndef TYPES_H_
#define TYPES_H_

#include <filesystem>
#include <string>

struct DatasetMetadata {
    std::string data_type_;
    unsigned int base_num_points_{0};
    unsigned int query_num_points_{0};
    unsigned int num_dimensions_{0};
    double chamfer_distance_{0.0};

    auto Save(const std::filesystem::path& file_path) const -> void;
    auto Load(const std::filesystem::path& file_path) -> void;
};

struct QalshConfiguration {
    double approximation_ratio_{0.0};
    double bucket_width_{0.0};
    double beta_{0.0};
    double error_probability_{0.0};
    unsigned int num_hash_tables_{0};
    unsigned int collision_threshold_{0};
    unsigned int page_size_{0};

    auto Save(const std::filesystem::path& file_path) const -> void;
    auto Load(const std::filesystem::path& file_path) -> void;
};

#endif