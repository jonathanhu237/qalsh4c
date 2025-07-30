#ifndef DATASET_METADATA
#define DATASET_METADATA

#include <filesystem>

struct DatasetMetadata {
    unsigned int num_points_a{0};
    unsigned int num_points_b{0};
    unsigned int num_dimensions{0};
    double chamfer_distance{0.0};

    void Load(const std::filesystem::path& file_path);
};

#endif