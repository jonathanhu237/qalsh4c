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

using Coordinate = double;
using Point = std::vector<Coordinate>;

struct PointSetMetadata {
    std::filesystem::path file_path;
    unsigned int num_points{0};
    unsigned int num_dimensions{0};
};

#endif