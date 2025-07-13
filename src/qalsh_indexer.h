#ifndef QALSH_INDEXER_H_
#define QALSH_INDEXER_H_

#include <filesystem>

#include "dataset_metadata.h"
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
    DatasetMetadata dataset_metadata_;

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