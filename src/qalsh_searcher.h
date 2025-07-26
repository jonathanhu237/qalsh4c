#ifndef QALSH_SEARCHER_H_
#define QALSH_SEARCHER_H_

#include <filesystem>
#include <optional>
#include <unordered_map>
#include <vector>

#include "b_plus_tree.h"
#include "point_set.h"

using SearchLocation = std::pair<LeafNode, size_t>;

class QalshSearcher {
   public:
    virtual ~QalshSearcher() = default;

    virtual void Init(double key) = 0;
    virtual std::vector<unsigned int> IncrementalSearch(double bound) = 0;
};

class BPlusTreeSearcher : public QalshSearcher {
   public:
    BPlusTreeSearcher(const std::filesystem::path& file_path, unsigned int page_size);

    void Init(double key) override;
    std::vector<unsigned int> IncrementalSearch(double bound) override;

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

class InMemorySearcher : public QalshSearcher {
   public:
    InMemorySearcher(PointSetReader* base_reader, std::vector<double> dot_vector);

    void Init(double key) override;
    std::vector<unsigned int> IncrementalSearch(double bound) override;

   private:
    double key_{0.0};
    std::vector<KeyValuePair> data_;

    std::optional<unsigned int> left_index_;
    std::optional<unsigned int> right_index_;
};

#endif