#ifndef QALSH_CONFIG_H
#define QALSH_CONFIG_H

struct QalshConfig {
    double approximation_ratio{0.0};
    double bucket_width{0.0};
    double beta{0.0};
    double error_probability{0.0};
    unsigned int num_hash_tables{0};
    unsigned int collision_threshold{0};
    unsigned int page_size{0};

    void Regularize(unsigned int num_points);
};

#endif