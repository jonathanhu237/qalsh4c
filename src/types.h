#ifndef TYPES_H_
#define TYPES_H_

#include <filesystem>

struct PointSetMetadata {
    std::filesystem::path file_path;
    unsigned int num_points{0};
    unsigned int num_dimensions{0};
};

#endif