#ifndef TYPES_H_
#define TYPES_H_

#include <filesystem>
#include <string>

struct DatasetMetadata {
    std::string data_type;
    unsigned int base_num_points{0};
    unsigned int query_num_points{0};
    unsigned int num_dimensions{0};
    double chamfer_distance{0.0};

    void Save(const std::filesystem::path& file_path) const;
    void Load(const std::filesystem::path& file_path);
    [[nodiscard]] std::string Details() const;
};

struct QalshConfiguration {
    double approximation_ratio{0.0};
    double bucket_width{0.0};
    double beta{0.0};
    double error_probability{0.0};
    unsigned int num_hash_tables{0};
    unsigned int collision_threshold{0};
    unsigned int page_size{0};

    void Save(const std::filesystem::path& file_path) const;
    void Load(const std::filesystem::path& file_path);
    void Regularize(unsigned int num_points);
    [[nodiscard]] std::string Details() const;
};

#endif