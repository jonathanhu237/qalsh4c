#ifndef TYPES_H_
#define TYPES_H_

#include <filesystem>
#include <vector>

struct KeyPageNumPair {
    double key;
    unsigned int page_num;
};

struct DotProductPointIdPair {
    double dot_product{0.0};
    unsigned int point_id{0};
};

struct AnnResult {
    double distance{0.0};
    unsigned int point_id{0};
};

struct CompareAnnResult {
    bool operator()(const AnnResult& a, const AnnResult& b) const { return a.distance > b.distance; }
};

struct DatasetMetadata {
    unsigned int num_points_a{0};
    unsigned int num_points_b{0};
    unsigned int num_dimensions{0};
    double chamfer_distance_l1{0.0};
    double chamfer_distance_l2{0.0};
};

using Coordinate = double;
using Point = std::vector<Coordinate>;

struct PointSetMetadata {
    std::filesystem::path file_path;
    unsigned int num_points{0};
    unsigned int num_dimensions{0};
};

struct QalshConfig {
    double approximation_ratio{0.0};
    double bucket_width{0.0};
    double error_probability{0.0};
    unsigned int num_hash_tables{0};
    unsigned int collision_threshold{0};
    unsigned int page_size{0};
};

#endif