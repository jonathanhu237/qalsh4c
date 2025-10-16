#ifndef QALSH_CONFIG_H_
#define QALSH_CONFIG_H_

#include <filesystem>

struct DatasetMetadata {
    unsigned int num_points_a{0};
    unsigned int num_points_b{0};
    unsigned int num_dimensions{0};
    double chamfer_distance_l1{0.0};
    double chamfer_distance_l2{0.0};

    void Load(const std::filesystem::path &file_path);
};

struct QalshConfig {
    double approximation_ratio{0.0};
    double bucket_width{0.0};
    double error_probability{0.0};
    unsigned int num_hash_tables{0};
    unsigned int collision_threshold{0};
    unsigned int page_size{0};

    void Save(const std::filesystem::path &file_path);
    void Load(const std::filesystem::path &file_path);
    void Regularize(unsigned int num_points, double norm_order);
};

#endif