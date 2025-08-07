#ifndef TYPES_H_
#define TYPES_H_

struct KeyPageNumPair {
    double key;
    unsigned int page_num;
};

struct DotProductPointIdPair {
    double dot_product{0.0};
    unsigned int point_id{0};
};

struct AnnResult {
    unsigned int point_id{0};
    double distance{0.0};
};

struct CompareAnnResult {
    bool operator()(const AnnResult& a, const AnnResult& b) const { return a.distance > b.distance; }
};

struct DatasetMetadata {
    unsigned int num_points_a{0};
    unsigned int num_points_b{0};
    unsigned int num_dimensions{0};
    double chamfer_distance{0.0};
};

#endif