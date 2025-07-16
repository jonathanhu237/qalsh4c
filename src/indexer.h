#ifndef INDEXER_H_
#define INDEXER_H_

#include <filesystem>
#include <memory>
#include <random>

#include "point_set.h"
#include "types.h"

class Indexer {
   public:
    virtual ~Indexer() = default;
    virtual void BuildIndex(const std::filesystem::path& dataset_directory) = 0;
};

class QalshIndexer : public Indexer {
   public:
    QalshIndexer(QalshConfiguration qalsh_config, bool in_memory);
    void BuildIndex(const std::filesystem::path& dataset_directory) override;

   private:
    [[nodiscard]] std::string Details() const;

    // Dataset specific parameters
    std::filesystem::path dataset_directory_;
    DatasetMetadata dataset_metadata_;
    std::unique_ptr<PointSetReader> base_reader_;

    // QALSH specific parameters
    QalshConfiguration qalsh_config_;
    bool in_memory_;

    std::mt19937 gen_;
};

#endif