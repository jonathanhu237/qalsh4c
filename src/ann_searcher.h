#ifndef ANN_SEARCHER_H_
#define ANN_SEARCHER_H_

#include <filesystem>
#include <memory>
#include <vector>

#include "hash_table_searcher.h"
#include "point_set.h"
#include "types.h"

class BPlusTreeSearcher;

class AnnSearcher {
   public:
    virtual ~AnnSearcher() = default;

    virtual void Init(const std::filesystem::path& dataset_directory) = 0;
    virtual AnnResult Search(const PointVariant& query_point) = 0;
};

class LinearScanAnnSearcher : public AnnSearcher {
   public:
    LinearScanAnnSearcher() = default;

    void Init(const std::filesystem::path& dataset_directory) override;
    AnnResult Search(const PointVariant& query_point) override;

   private:
    std::unique_ptr<PointSetReader> base_reader_;
};

class QalshAnnSearcher : public AnnSearcher {
   public:
    QalshAnnSearcher() = default;

    void Init(const std::filesystem::path& dataset_directory) override;
    AnnResult Search(const PointVariant& query_point) override;

   private:
    std::unique_ptr<PointSetReader> base_reader_;

    std::filesystem::path index_directory_;
    QalshConfiguration qalsh_config_;
    std::vector<std::vector<double>> dot_vectors_;
    std::vector<std::unique_ptr<HashTableSearcher>> qalsh_searchers_;
};

#endif