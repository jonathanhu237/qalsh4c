#ifndef ANN_SEARCHER_H_
#define ANN_SEARCHER_H_

#include <filesystem>
#include <memory>
#include <vector>

#include "point_set.h"
#include "qalsh_searcher.h"
#include "types.h"

class BPlusTreeSearcher;

class AnnSearcher {
   public:
    virtual ~AnnSearcher() = default;
    virtual AnnResult Search(const PointVariant& query_point) = 0;
};

class LinearScanAnnSearcher : public AnnSearcher {
   public:
    LinearScanAnnSearcher(PointSetReader* base_reader);

    AnnResult Search(const PointVariant& query_point) override;

   private:
    PointSetReader* base_reader_;
};

class QalshAnnSearcher : public AnnSearcher {
   public:
    QalshAnnSearcher(PointSetReader* base_reader, std::filesystem::path index_directory);

    AnnResult Search(const PointVariant& query_point) override;

   private:
    PointSetReader* base_reader_;
    std::filesystem::path index_directory_;
    QalshConfiguration qalsh_config_;
    std::vector<std::vector<double>> dot_vectors_;
    std::vector<std::unique_ptr<QalshSearcher>> qalsh_searchers_;
};

#endif