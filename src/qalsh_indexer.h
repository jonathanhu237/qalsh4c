#ifndef QALSH_INDEXER_H_
#define QALSH_INDEXER_H_

#include <filesystem>

#include "indexer.h"

class QalshIndexer : public Indexer {
   public:
    QalshIndexer(std::filesystem::path dataset_directory, double approximation_ratio, double bucket_width, double beta,
                 double error_probability, unsigned int num_hash_tables, unsigned int collision_threshold,
                 unsigned int page_size);
    auto BuildIndex() -> void override;

   private:
    auto PrintConfiguration() const -> void;

    // Dataset specific parameters
    std::filesystem::path dataset_directory_;
    unsigned int base_num_points_{0};
    unsigned int query_num_points_{0};
    unsigned int num_dimensions_{0};
    std::string data_type;

    // QALSH specific parameters
    double approximation_ratio_{0.0};
    double bucket_width_{0.0};
    double beta_{0.0};
    double error_probability_{0.0};
    unsigned int num_hash_tables_{0};
    unsigned int collision_threshold_{0};
    unsigned int page_size_{0};
    bool verbose_{false};
};

#endif