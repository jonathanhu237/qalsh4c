#ifndef B_PLUS_TREE_H_
#define B_PLUS_TREE_H_

#include <cstddef>
#include <utility>
#include <vector>

#include "constants.h"
#include "pager.h"

namespace qalsh_chamfer {

class InternalNode {
   public:
    friend class BPlusTree;

   private:
    InternalNode(unsigned int order);
    auto static GetHeaderSize() -> size_t;
    auto Serialize(std::vector<char>& buffer) const -> void;

    // Header
    NodeType node_type_;
    unsigned int num_children_;

    // Data
    std::vector<double> keys_;
    std::vector<unsigned int> pointers_;
};

class LeafNode {
   public:
    friend class BPlusTree;

   private:
    LeafNode(unsigned int order);
    auto static GetHeaderSize() -> size_t;
    auto Serialize(std::vector<char>& buffer) const -> void;

    // Header
    NodeType node_type_;
    unsigned int num_entries_;
    unsigned int prev_leaf_page_num_;
    unsigned int next_leaf_page_num_;

    // Data
    std::vector<double> keys_;
    std::vector<unsigned int> values_;
};

class BPlusTree {
   public:
    BPlusTree(Pager&& pager);
    auto BulkLoad(std::vector<std::pair<double, unsigned int>>& data) -> void;

   private:
    Pager pager_;

    // Header
    unsigned int root_page_num_;
    unsigned int level_;
    unsigned int internal_node_order_;
    unsigned int leaf_node_order_;
};

}  // namespace qalsh_chamfer

#endif