#ifndef DATASET_METADATA_H_
#define DATASET_METADATA_H_

#include <filesystem>
#include <string>

class DatasetMetadata {
   public:
    DatasetMetadata(unsigned int base_num_points, unsigned int query_num_points, unsigned int num_dimensions,
                    std::string data_type);
    DatasetMetadata(const std::filesystem::path& file_path);
    auto Save(const std::filesystem::path& file_path) const -> void;

    unsigned int base_num_points_{0};
    unsigned int query_num_points_{0};
    unsigned int num_dimensions_{0};
    std::string data_type_;
};

#endif