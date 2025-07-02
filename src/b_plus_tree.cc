#include "b_plus_tree.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "constants.h"
#include "pager.h"
#include "utils.hpp"

namespace qalsh_chamfer {

// ----- InternalNode Implementation -----
InternalNode::InternalNode(unsigned int order) : node_type_(NodeType::kInternalNode), num_children_(0) {
    keys_.resize(order - 1);
    pointers_.resize(order);
};

auto InternalNode::GetHeaderSize() -> size_t { return sizeof(node_type_) + sizeof(num_children_); }

auto InternalNode::Serialize(std::vector<char>& buffer) const -> void {
    buffer.clear();

    Utils::AppendToBuffer(buffer, node_type_);
    Utils::AppendToBuffer(buffer, num_children_);

    for (auto key : keys_) {
        Utils::AppendToBuffer(buffer, key);
    }
    for (auto pointer : pointers_) {
        Utils::AppendToBuffer(buffer, pointer);
    }
}

// ----- LeafNode Implementation -----
LeafNode::LeafNode(unsigned int order)
    : node_type_(NodeType::kLeafNode), num_entries_(0), prev_leaf_page_num_(0), next_leaf_page_num_(0) {
    keys_.resize(order);
    values_.resize(order);
};

auto LeafNode::GetHeaderSize() -> size_t {
    return sizeof(node_type_) + sizeof(num_entries_) + sizeof(prev_leaf_page_num_) + sizeof(next_leaf_page_num_);
}

auto LeafNode::Serialize(std::vector<char>& buffer) const -> void {
    buffer.clear();

    Utils::AppendToBuffer(buffer, node_type_);
    Utils::AppendToBuffer(buffer, num_entries_);
    Utils::AppendToBuffer(buffer, prev_leaf_page_num_);
    Utils::AppendToBuffer(buffer, next_leaf_page_num_);

    for (auto key : keys_) {
        Utils::AppendToBuffer(buffer, key);
    }
    for (auto value : values_) {
        Utils::AppendToBuffer(buffer, value);
    }
}

// ----- BPlusTree Implementation -----
BPlusTree::BPlusTree(Pager&& pager, const std::vector<double>& dot_vector)
    : pager_(std::move(pager)),
      root_page_num_(0),
      level_(0),
      internal_node_order_(0),
      leaf_node_order_(0),
      dot_vector_(dot_vector) {
    // Calculate the order of InternalNode and LeafNode
    unsigned int page_size = pager_.get_page_size();
    internal_node_order_ = static_cast<unsigned>(
        (((page_size - InternalNode::GetHeaderSize() + sizeof(double))) / sizeof(double)) + sizeof(unsigned int));
    leaf_node_order_ =
        static_cast<unsigned>((((page_size - LeafNode::GetHeaderSize())) / sizeof(double)) + sizeof(unsigned int));
}

auto BPlusTree::GetHeaderBasicInfoSize() -> size_t {
    return sizeof(root_page_num_) + sizeof(level_) + sizeof(internal_node_order_) + sizeof(leaf_node_order_);
}

auto BPlusTree::BulkLoad(std::vector<std::pair<double, unsigned int>>& data) -> void {
    std::vector<std::pair<double, unsigned int>> parent_level_entries;

    // Reserve page 0 for the file header
    pager_.Allocate();

    // Build the leaf nodes
    unsigned int prev_leaf_page_num = 0;
    size_t data_idx = 0;

    while (data_idx < data.size()) {
        LeafNode new_leaf_node(leaf_node_order_);
        new_leaf_node.prev_leaf_page_num_ = prev_leaf_page_num;

        size_t chunk_end = std::min(data_idx + leaf_node_order_, data.size());
        for (size_t i = data_idx; i < chunk_end; i++) {
            new_leaf_node.keys_.push_back(data[i].first);
            new_leaf_node.values_.push_back(data[i].second);
            new_leaf_node.num_entries_++;
        }
        data_idx = chunk_end;

        new_leaf_node.next_leaf_page_num_ = (data_idx < data.size()) ? pager_.get_num_page() : 0;

        // Serialize the leaf node and write it to the file
        std::vector<char> buffer(pager_.get_page_size(), 0);
        new_leaf_node.Serialize(buffer);
        unsigned int new_leaf_page_num = pager_.Allocate();
        pager_.WritePage(new_leaf_page_num, buffer);

        // Add entry to the parent level
        parent_level_entries.emplace_back(new_leaf_node.keys_.front(), new_leaf_page_num);

        // Update the previous leaf page number
        prev_leaf_page_num = new_leaf_page_num;
    }

    // Build the internal nodes
    while (parent_level_entries.size() > 1) {
        level_++;

        std::vector<std::pair<double, unsigned int>> next_parent_level_entries;
        size_t entry_idx = 0;

        while (entry_idx < parent_level_entries.size()) {
            double separator_key_for_next_level = parent_level_entries[entry_idx].first;

            InternalNode new_internal_node(internal_node_order_);

            // First pointer in the node has no preceding key
            new_internal_node.pointers_.push_back(parent_level_entries[entry_idx].second);
            new_internal_node.num_children_++;

            size_t chunk_end = std::min(entry_idx + internal_node_order_, parent_level_entries.size());

            for (size_t i = entry_idx + 1; i < chunk_end; i++) {
                new_internal_node.keys_.push_back(parent_level_entries[i].first);
                new_internal_node.pointers_.push_back(parent_level_entries[i].second);
                new_internal_node.num_children_++;
            }

            // Serialize the internal node and write it to the file
            std::vector<char> buffer(pager_.get_page_size(), 0);
            new_internal_node.Serialize(buffer);
            unsigned int new_internal_page_num = pager_.Allocate();
            pager_.WritePage(new_internal_page_num, buffer);

            // Add entry to the next parent level
            next_parent_level_entries.emplace_back(separator_key_for_next_level, new_internal_page_num);

            // Update the parent level entries
            parent_level_entries = next_parent_level_entries;
        }
    }

    // write the header
    std::vector<char> buffer(pager_.get_page_size(), 0);
    Utils::AppendToBuffer(buffer, root_page_num_);
    Utils::AppendToBuffer(buffer, level_);
    Utils::AppendToBuffer(buffer, internal_node_order_);
    Utils::AppendToBuffer(buffer, leaf_node_order_);

    pager_.WritePage(0, buffer);
};

}  // namespace qalsh_chamfer