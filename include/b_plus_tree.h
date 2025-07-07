#ifndef B_PLUS_TREE_H_
#define B_PLUS_TREE_H_

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "pager.h"

namespace qalsh_chamfer {

class InternalNode {
   public:
    friend class BPlusTree;

   private:
    InternalNode();
    InternalNode(unsigned int order);
    InternalNode(const std::vector<char>& buffer);

    auto static GetHeaderSize() -> size_t;
    auto Serialize(std::vector<char>& buffer) const -> void;

    // Header
    unsigned int num_children_;

    // Data
    std::vector<double> keys_;
    std::vector<unsigned int> pointers_;
};

class LeafNode {
   public:
    friend class BPlusTree;

   private:
    LeafNode();
    LeafNode(unsigned int order);
    LeafNode(const std::vector<char>& buffer);

    auto static GetHeaderSize() -> size_t;
    auto Serialize(std::vector<char>& buffer) const -> void;

    // Header
    unsigned int num_entries_;
    unsigned int prev_leaf_page_num_;
    unsigned int next_leaf_page_num_;

    // Data
    std::vector<double> keys_;
    std::vector<unsigned int> values_;
};

class BPlusTree {
   public:
    using KeyValuePair = std::pair<double, unsigned int>;
    using SearchLocation = std::pair<LeafNode, size_t>;

    BPlusTree(Pager&& pager);

    auto BulkLoad(std::vector<KeyValuePair>& data) -> void;

    auto InitIncrementalSearch(double key) -> void;
    auto IncrementalSearch(double key, double bound) -> std::vector<unsigned int>;

   private:
    auto LocateLeafMayContainKey(double key) -> LeafNode;
    auto LocateLeafByPageNum(unsigned int page_num) -> LeafNode;

    Pager pager_;

    // Header
    unsigned int root_page_num_;
    unsigned int level_;
    unsigned int internal_node_order_;
    unsigned int leaf_node_order_;

    // Variables for incremental search
    std::optional<SearchLocation> left_search_location_;
    std::optional<SearchLocation> right_search_location_;
};

}  // namespace qalsh_chamfer

#endif