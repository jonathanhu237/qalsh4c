#ifndef ANN_SEARCHER_H_
#define ANN_SEARCHER_H_

#include <filesystem>
#include <vector>

#include "b_plus_tree.h"
#include "point_set.h"
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
    std::vector<BPlusTreeSearcher> b_plus_tree_searchers_;
};

// Searcher for the QALSH.
class BPlusTreeSearcher {
   public:
    BPlusTreeSearcher(const std::filesystem::path& file_path, unsigned int page_size);

    void Init(double key);
    std::vector<unsigned int> IncrementalSearch(double bound);

   private:
    LeafNode LocateLeafMayContainKey();
    LeafNode LocateLeafByPageNum(unsigned int page_num);

    std::vector<char> ReadPage(unsigned int page_num);

    std::ifstream ifs_;
    unsigned int page_size_{0};
    double key_{0.0};
    std::optional<SearchLocation> left_search_location_;
    std::optional<SearchLocation> right_search_location_;

    // Header
    unsigned int root_page_num_{0};
    unsigned int level_{0};
    unsigned int internal_node_order_{0};
    unsigned int leaf_node_order_{0};

    // Performance optimizations
    std::unordered_map<unsigned int, std::vector<char>> page_cache_;
};

#endif