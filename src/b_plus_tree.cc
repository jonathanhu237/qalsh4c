#include "b_plus_tree.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <utility>
#include <vector>

#include "pager.h"
#include "utils.h"

namespace qalsh_chamfer {

// ----- InternalNode Implementation -----
InternalNode::InternalNode(unsigned int order) : num_children_(0) {
    keys_.reserve(order - 1);
    pointers_.reserve(order);
};

auto InternalNode::GetHeaderSize() -> size_t { return sizeof(num_children_); }

auto InternalNode::Serialize(std::vector<char>& buffer) const -> void {
    buffer.clear();

    Utils::AppendToBuffer(buffer, num_children_);

    for (auto key : keys_) {
        Utils::AppendToBuffer(buffer, key);
    }
    for (auto pointer : pointers_) {
        Utils::AppendToBuffer(buffer, pointer);
    }
}

auto InternalNode::Deserialize(const std::vector<char>& buffer) -> void {
    size_t offset = 0;
    keys_.clear();
    pointers_.clear();

    num_children_ = Utils::ReadFromBuffer<unsigned int>(buffer, offset);

    for (unsigned int i = 0; i < num_children_ - 1; ++i) {
        keys_.push_back(Utils::ReadFromBuffer<double>(buffer, offset));
    }
    for (unsigned int i = 0; i < num_children_; ++i) {
        pointers_.push_back(Utils::ReadFromBuffer<unsigned int>(buffer, offset));
    }
}

// ----- LeafNode Implementation -----
LeafNode::LeafNode(unsigned int order) : num_entries_(0), prev_leaf_page_num_(0), next_leaf_page_num_(0) {
    keys_.reserve(order);
    values_.reserve(order);
};

auto LeafNode::GetHeaderSize() -> size_t {
    return sizeof(num_entries_) + sizeof(prev_leaf_page_num_) + sizeof(next_leaf_page_num_);
}

auto LeafNode::Serialize(std::vector<char>& buffer) const -> void {
    buffer.clear();

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

auto LeafNode::Deserialize(const std::vector<char>& buffer) -> void {
    size_t offset = 0;
    keys_.clear();
    values_.clear();

    num_entries_ = Utils::ReadFromBuffer<unsigned int>(buffer, offset);
    prev_leaf_page_num_ = Utils::ReadFromBuffer<unsigned int>(buffer, offset);
    next_leaf_page_num_ = Utils::ReadFromBuffer<unsigned int>(buffer, offset);

    for (unsigned int i = 0; i < num_entries_; ++i) {
        keys_.push_back(Utils::ReadFromBuffer<double>(buffer, offset));
        values_.push_back(Utils::ReadFromBuffer<unsigned int>(buffer, offset));
    }
}

// ----- BPlusTree Implementation -----
BPlusTree::BPlusTree(Pager&& pager)
    : pager_(std::move(pager)), root_page_num_(0), level_(0), internal_node_order_(0), leaf_node_order_(0) {
    if (pager_.get_mode() == Pager::PagerMode::kRead) {
        // Read the header from page 0
        std::vector<char> header_buffer = pager_.ReadPage(0);
        size_t offset = 0;
        root_page_num_ = Utils::ReadFromBuffer<unsigned int>(header_buffer, offset);
        level_ = Utils::ReadFromBuffer<unsigned int>(header_buffer, offset);
        internal_node_order_ = Utils::ReadFromBuffer<unsigned int>(header_buffer, offset);
        leaf_node_order_ = Utils::ReadFromBuffer<unsigned int>(header_buffer, offset);
    } else {
        unsigned int page_size = pager_.get_page_size();
        internal_node_order_ = static_cast<unsigned>(
            (((page_size - InternalNode::GetHeaderSize() + sizeof(double))) / sizeof(double)) + sizeof(unsigned int));
        leaf_node_order_ =
            static_cast<unsigned>((((page_size - LeafNode::GetHeaderSize())) / sizeof(double)) + sizeof(unsigned int));
    }
}

auto BPlusTree::BulkLoad(std::vector<KeyValuePair>& data) -> void {
    std::vector<KeyValuePair> parent_level_entries;

    // Reserve page 0 for the file header
    pager_.Allocate();

    // Build the leaf nodes
    unsigned int prev_leaf_page_num = 0;
    unsigned int new_leaf_page_num = 0;
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
        new_leaf_page_num = pager_.Allocate();
        pager_.WritePage(new_leaf_page_num, buffer);

        // Add entry to the parent level
        parent_level_entries.emplace_back(new_leaf_node.keys_.front(), new_leaf_page_num);

        // Update the previous leaf page number
        prev_leaf_page_num = new_leaf_page_num;
    }

    root_page_num_ = new_leaf_page_num;

    unsigned int new_internal_page_num = 0;

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
            entry_idx = chunk_end;

            // Serialize the internal node and write it to the file
            std::vector<char> buffer(pager_.get_page_size(), 0);
            new_internal_node.Serialize(buffer);
            new_internal_page_num = pager_.Allocate();
            pager_.WritePage(new_internal_page_num, buffer);

            // Add entry to the next parent level
            next_parent_level_entries.emplace_back(separator_key_for_next_level, new_internal_page_num);

            // Update the parent level entries
            parent_level_entries = next_parent_level_entries;
        }
    }

    root_page_num_ = new_internal_page_num == 0 ? root_page_num_ : new_internal_page_num;

    // write the header
    std::vector<char> buffer(pager_.get_page_size(), 0);
    Utils::AppendToBuffer(buffer, root_page_num_);
    Utils::AppendToBuffer(buffer, level_);
    Utils::AppendToBuffer(buffer, internal_node_order_);
    Utils::AppendToBuffer(buffer, leaf_node_order_);

    pager_.WritePage(0, buffer);
};

auto BPlusTree::Locate(double key) -> LocateResult {
    unsigned int current_level = level_;
    unsigned int next_page_num = root_page_num_;

    while (current_level != 0) {
        const std::vector<char> buffer = pager_.ReadPage(next_page_num);
        InternalNode internal_node(internal_node_order_);
        internal_node.Deserialize(buffer);

        auto it = std::ranges::upper_bound(internal_node.keys_, key);
        auto index = static_cast<size_t>(std::distance(internal_node.keys_.begin(), it));
        next_page_num = internal_node.pointers_.at(index);

        current_level--;
    }

    return Locate(next_page_num);
}

auto BPlusTree::Locate(unsigned int page_num) -> LocateResult {
    LeafNode leaf_node(leaf_node_order_);
    const std::vector<char> buffer = pager_.ReadPage(page_num);
    leaf_node.Deserialize(buffer);

    LocateResult result;
    result.left_page_num = leaf_node.prev_leaf_page_num_;
    result.right_page_num = leaf_node.next_leaf_page_num_;
    result.data.reserve(leaf_node.num_entries_);
    for (size_t i = 0; i < leaf_node.num_entries_; ++i) {
        result.data.emplace_back(leaf_node.keys_[i], leaf_node.values_[i]);
    }
    return result;
}

}  // namespace qalsh_chamfer