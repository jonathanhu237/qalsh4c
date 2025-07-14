#ifndef INDEXER_H_
#define INDEXER_H_

#include <random>

#include "filesystem"
#include "point_set.h"
#include "types.h"

class Indexer {
   public:
    virtual ~Indexer() = default;
    virtual auto BuildIndex() -> void = 0;
};

class QalshIndexer : public Indexer {
   public:
    QalshIndexer(std::filesystem::path dataset_directory, double approximation_ratio, double bucket_width, double beta,
                 double error_probability, unsigned int num_hash_tables, unsigned int collision_threshold,
                 unsigned int page_size, bool in_memory);
    auto BuildIndex() -> void override;

   private:
    auto PrintConfiguration() const -> void;

    // Dataset specific parameters
    std::filesystem::path dataset_directory_;
    DatasetMetadata dataset_metadata_;
    std::unique_ptr<IPointSetReader> base_reader_;

    // QALSH specific parameters
    QalshConfiguration qalsh_config_;

    bool in_memory_;
    std::mt19937 gen_;
};

#endif