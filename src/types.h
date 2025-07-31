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

#endif