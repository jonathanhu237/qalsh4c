#include "b_plus_tree.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include "utils.h"

// ---------- InternalNode Implementation ----------
InternalNode::InternalNode(unsigned int order) {
    keys_.reserve(order - 1);
    pointers_.reserve(order);
};

InternalNode::InternalNode(const std::vector<char>& buffer) {
    size_t offset = 0;

    num_children_ = Utils::ReadFromBuffer<unsigned int>(buffer, offset);

    keys_ = Utils::ReadVectorFromBuffer<double>(buffer, offset, num_children_ - 1);
    pointers_ = Utils::ReadVectorFromBuffer<unsigned int>(buffer, offset, num_children_);
};

size_t InternalNode::GetHeaderSize() { return sizeof(num_children_); }

void InternalNode::Serialize(std::vector<char>& buffer) const {
    size_t offset = 0;
    Utils::WriteToBuffer(buffer, offset, num_children_);

    for (auto key : keys_) {
        Utils::WriteToBuffer(buffer, offset, key);
    }
    for (auto pointer : pointers_) {
        Utils::WriteToBuffer(buffer, offset, pointer);
    }
}

// ---------- LeafNode Implementation ----------
LeafNode::LeafNode(unsigned int order) {
    keys_.reserve(order);
    values_.reserve(order);
};

LeafNode::LeafNode(const std::vector<char>& buffer) {
    size_t offset = 0;

    num_entries_ = Utils::ReadFromBuffer<unsigned int>(buffer, offset);
    prev_leaf_page_num_ = Utils::ReadFromBuffer<unsigned int>(buffer, offset);
    next_leaf_page_num_ = Utils::ReadFromBuffer<unsigned int>(buffer, offset);

    keys_ = Utils::ReadVectorFromBuffer<double>(buffer, offset, num_entries_);
    values_ = Utils::ReadVectorFromBuffer<unsigned int>(buffer, offset, num_entries_);
}

size_t LeafNode::GetHeaderSize() {
    return sizeof(num_entries_) + sizeof(prev_leaf_page_num_) + sizeof(next_leaf_page_num_);
}

void LeafNode::Serialize(std::vector<char>& buffer) const {
    size_t offset = 0;
    Utils::WriteToBuffer(buffer, offset, num_entries_);
    Utils::WriteToBuffer(buffer, offset, prev_leaf_page_num_);
    Utils::WriteToBuffer(buffer, offset, next_leaf_page_num_);

    for (auto key : keys_) {
        Utils::WriteToBuffer(buffer, offset, key);
    }
    for (auto value : values_) {
        Utils::WriteToBuffer(buffer, offset, value);
    }
}

// ---------- BPlusTreeBulkLoader Implementation ----------
BPlusTreeBulkLoader::BPlusTreeBulkLoader(const std::filesystem::path& file_path, unsigned int page_size)
    : page_size_(page_size) {
    ofs_.open(file_path, std::ios::binary | std::ios::trunc);
    if (!ofs_) {
        spdlog::error("Failed to open file: {}", file_path.string());
    }

    internal_node_order_ = static_cast<unsigned int>((page_size - InternalNode::GetHeaderSize() + sizeof(double)) /
                                                     (sizeof(double) + sizeof(unsigned int)));
    leaf_node_order_ =
        static_cast<unsigned int>((page_size - LeafNode::GetHeaderSize()) / (sizeof(double) + sizeof(unsigned int)));
    buffer_.resize(page_size_, 0);
}

void BPlusTreeBulkLoader::Build(const std::vector<DotProductPointIdPair>& data) {
    std::vector<KeyPageNumPair> parent_level_entries;

    // Reserve page 0 for the file header
    AllocatePage();

    // Build the leaf nodes
    unsigned int prev_leaf_page_num = 0;
    unsigned int new_leaf_page_num = 0;
    size_t data_idx = 0;

    while (data_idx < data.size()) {
        LeafNode new_leaf_node(leaf_node_order_);
        new_leaf_node.prev_leaf_page_num_ = prev_leaf_page_num;

        size_t chunk_end = std::min(data_idx + leaf_node_order_, data.size());
        for (size_t i = data_idx; i < chunk_end; i++) {
            new_leaf_node.keys_.emplace_back(data[i].dot_product);
            new_leaf_node.values_.emplace_back(data[i].point_id);
            new_leaf_node.num_entries_++;
        }
        data_idx = chunk_end;

        // Serialize the leaf node and write it to the file
        new_leaf_page_num = AllocatePage();
        new_leaf_node.next_leaf_page_num_ = (data_idx < data.size()) ? next_page_num_ : 0;
        new_leaf_node.Serialize(buffer_);
        WritePage(new_leaf_page_num);

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

        std::vector<KeyPageNumPair> next_parent_level_entries;
        size_t entry_idx = 0;

        while (entry_idx < parent_level_entries.size()) {
            double separator_key_for_next_level = parent_level_entries[entry_idx].key;

            InternalNode new_internal_node(internal_node_order_);

            // First pointer in the node has no preceding key
            new_internal_node.pointers_.emplace_back(parent_level_entries[entry_idx].page_num);
            new_internal_node.num_children_++;

            size_t chunk_end = std::min(entry_idx + internal_node_order_, parent_level_entries.size());

            for (size_t i = entry_idx + 1; i < chunk_end; i++) {
                new_internal_node.keys_.emplace_back(parent_level_entries[i].key);
                new_internal_node.pointers_.emplace_back(parent_level_entries[i].page_num);
                new_internal_node.num_children_++;
            }
            entry_idx = chunk_end;

            // Serialize the internal node and write it to the file
            new_internal_page_num = AllocatePage();
            new_internal_node.Serialize(buffer_);
            WritePage(new_internal_page_num);

            // Add entry to the next parent level
            next_parent_level_entries.emplace_back(separator_key_for_next_level, new_internal_page_num);

            // Update the parent level entries
            parent_level_entries = next_parent_level_entries;
        }
    }

    root_page_num_ = new_internal_page_num == 0 ? root_page_num_ : new_internal_page_num;

    // write the header
    size_t offset = 0;

    Utils::WriteToBuffer(buffer_, offset, root_page_num_);
    Utils::WriteToBuffer(buffer_, offset, level_);

    WritePage(0);
}

unsigned int BPlusTreeBulkLoader::AllocatePage() {
    unsigned int new_page_num = next_page_num_++;

    ofs_.seekp(static_cast<std::streamoff>(new_page_num * page_size_));
    std::vector<char> empty_page(page_size_, 0);
    ofs_.write(empty_page.data(), static_cast<std::streamsize>(empty_page.size()));
    num_page_++;

    return new_page_num;
}

void BPlusTreeBulkLoader::WritePage(unsigned int page_num) {
    ofs_.seekp(static_cast<std::streamoff>(page_num * page_size_));
    ofs_.write(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
}
