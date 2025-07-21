#ifndef B_PLUS_TREE_H_
#define B_PLUS_TREE_H_

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

class LeafNode;

using SearchLocation = std::pair<LeafNode, size_t>;
using KeyValuePair = std::pair<double, unsigned int>;

class InternalNode {
   public:
    friend class BPlusTree;
    friend class BPlusTreeBulkLoader;
    friend class BPlusTreeSearcher;

   private:
    InternalNode(unsigned int order);
    InternalNode(const std::vector<char>& buffer);

    static size_t GetHeaderSize();
    void Serialize(std::vector<char>& buffer) const;

    // Header
    unsigned int num_children_{0};

    // Data
    std::vector<double> keys_;
    std::vector<unsigned int> pointers_;
};

class LeafNode {
   public:
    friend class BPlusTree;
    friend class BPlusTreeBulkLoader;
    friend class BPlusTreeSearcher;

   private:
    LeafNode(unsigned int order);
    LeafNode(const std::vector<char>& buffer);

    static size_t GetHeaderSize();
    void Serialize(std::vector<char>& buffer) const;

    // Header
    unsigned int num_entries_{0};
    unsigned int prev_leaf_page_num_{0};
    unsigned int next_leaf_page_num_{0};

    // Data
    std::vector<double> keys_;
    std::vector<unsigned int> values_;
};

class BPlusTreeBulkLoader {
   public:
    BPlusTreeBulkLoader(const std::filesystem::path& file_path, unsigned int page_size);

    void Build(const std::vector<KeyValuePair>& data);

   private:
    unsigned int AllocatePage();
    void WritePage(unsigned int page_num, const std::vector<char>& buffer);

    std::ofstream ofs_;
    unsigned int page_size_{0};
    unsigned int num_page_{0};
    unsigned int next_page_num_{0};

    // Header
    unsigned int root_page_num_{0};
    unsigned int level_{0};
    unsigned int internal_node_order_{0};
    unsigned int leaf_node_order_{0};
};

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