#include "b_plus_tree.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include "log.h"

// ---------- InternalNode Implementation ----------
InternalNode::InternalNode(unsigned int order) {
    keys_.reserve(order - 1);
    pointers_.reserve(order);
};

InternalNode::InternalNode(const Buffer& buffer) {
    size_t offset = 0;

    num_children_ = buffer.Read<unsigned int>(offset);
    keys_ = buffer.ReadVector<double>(offset, num_children_ - 1);
    pointers_ = buffer.ReadVector<unsigned int>(offset, num_children_);
};

size_t InternalNode::GetHeaderSize() { return sizeof(num_children_); }

void InternalNode::Serialize(Buffer& buffer) const {
    size_t offset = 0;
    buffer.Write(offset, num_children_);

    for (auto key : keys_) {
        buffer.Write(offset, key);
    }
    for (auto pointer : pointers_) {
        buffer.Write(offset, pointer);
    }
}

// ---------- LeafNode Implementation ----------
LeafNode::LeafNode(unsigned int order) {
    keys_.reserve(order);
    values_.reserve(order);
};

LeafNode::LeafNode(const Buffer& buffer) {
    size_t offset = 0;

    num_entries_ = buffer.Read<unsigned int>(offset);
    prev_leaf_page_num_ = buffer.Read<unsigned int>(offset);
    next_leaf_page_num_ = buffer.Read<unsigned int>(offset);

    keys_ = buffer.ReadVector<double>(offset, num_entries_);
    values_ = buffer.ReadVector<unsigned int>(offset, num_entries_);
}

size_t LeafNode::GetHeaderSize() {
    return sizeof(num_entries_) + sizeof(prev_leaf_page_num_) + sizeof(next_leaf_page_num_);
}

void LeafNode::Serialize(Buffer& buffer) const {
    size_t offset = 0;
    buffer.Write(offset, num_entries_);
    buffer.Write(offset, prev_leaf_page_num_);
    buffer.Write(offset, next_leaf_page_num_);

    for (auto key : keys_) {
        buffer.Write(offset, key);
    }
    for (auto value : values_) {
        buffer.Write(offset, value);
    }
}

// ---------- BPlusTreeBulkLoader Implementation ----------
BPlusTreeBulkLoader::BPlusTreeBulkLoader(const std::filesystem::path& file_path, unsigned int page_size)
    : page_size_(page_size) {
    ofs_.open(file_path, std::ios::binary | std::ios::trunc);
    if (!ofs_) {
        Log::error("Failed to open file: {}", file_path.string());
    }

    internal_node_order_ = static_cast<unsigned int>((page_size - InternalNode::GetHeaderSize() + sizeof(double)) /
                                                     (sizeof(double) + sizeof(unsigned int)));
    leaf_node_order_ =
        static_cast<unsigned int>((page_size - LeafNode::GetHeaderSize()) / (sizeof(double) + sizeof(unsigned int)));
    buffer_.Init(page_size_);
}

void BPlusTreeBulkLoader::Build(const std::vector<DotProductPointIdPair>& data) {
    size_t index = 0;
    BuildFromStream(data.size(), [&data, &index]() { return data.at(index++); });
}

void BPlusTreeBulkLoader::BuildFromStream(size_t num_entries, std::function<DotProductPointIdPair()> next) {
    std::vector<KeyPageNumPair> parent_level_entries;

    // Reserve page 0 for the file header
    AllocatePage();

    // Build the leaf nodes
    unsigned int prev_leaf_page_num = 0;
    unsigned int new_leaf_page_num = 0;
    size_t processed_entries = 0;

    while (processed_entries < num_entries) {
        LeafNode new_leaf_node(leaf_node_order_);
        new_leaf_node.prev_leaf_page_num_ = prev_leaf_page_num;

        size_t chunk_end = std::min(processed_entries + leaf_node_order_, num_entries);
        for (size_t i = processed_entries; i < chunk_end; i++) {
            DotProductPointIdPair pair = next();
            new_leaf_node.keys_.emplace_back(pair.dot_product);
            new_leaf_node.values_.emplace_back(pair.point_id);
            new_leaf_node.num_entries_++;
        }
        processed_entries = chunk_end;

        // Serialize the leaf node and write it to the file
        new_leaf_page_num = AllocatePage();
        new_leaf_node.next_leaf_page_num_ = (processed_entries < num_entries) ? next_page_num_ : 0;
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
        }

        // Move up one level after finishing the current level
        parent_level_entries = std::move(next_parent_level_entries);
    }

    root_page_num_ = new_internal_page_num == 0 ? root_page_num_ : new_internal_page_num;

    // write the header
    size_t offset = 0;

    buffer_.Write(offset, root_page_num_);
    buffer_.Write(offset, level_);

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
    buffer_.Save(ofs_);
}

// ---------- BPlusTreeSearcher Implementation ----------
BPlusTreeSearcher::BPlusTreeSearcher(const std::filesystem::path& file_path, unsigned int page_size)
    : page_size_(page_size) {
    // Open the B+ tree file
    ifs_.open(file_path, std::ios::binary);
    if (!ifs_) {
        Log::error("Failed to open file: {}", file_path.string());
        return;
    }

    // Initialize the buffer
    buffer_.Init(page_size_);

    ReadPage(0);
    size_t offset = 0;
    root_page_num_ = buffer_.Read<unsigned int>(offset);
    level_ = buffer_.Read<unsigned int>(offset);
}

void BPlusTreeSearcher::Init(double query_key) {
    query_key_ = query_key;

    // Locate the leaf node that may contain the key.
    std::shared_ptr<LeafNode> leaf_node = LocateLeafMayContainKey();
    auto it = std::ranges::lower_bound(leaf_node->keys_, query_key_);
    auto index = static_cast<size_t>(std::distance(leaf_node->keys_.begin(), it));

    // Determine the left search location
    if (index == 0) {
        if (leaf_node->prev_leaf_page_num_ != 0) {
            std::shared_ptr<LeafNode> prev_leaf_node = LocateLeafByPageNum(leaf_node->prev_leaf_page_num_);
            left_ = SearchRecord{.leaf_node = prev_leaf_node, .index = prev_leaf_node->num_entries_ - 1};
        } else {
            left_ = std::nullopt;
        }
    } else {
        left_ = SearchRecord{.leaf_node = leaf_node, .index = static_cast<unsigned int>(index - 1)};
    }

    // Determine the right search location
    if (index == leaf_node->keys_.size()) {
        if (leaf_node->next_leaf_page_num_ != 0) {
            std::shared_ptr<LeafNode> next_leaf_node = LocateLeafByPageNum(leaf_node->next_leaf_page_num_);
            right_ = SearchRecord{.leaf_node = next_leaf_node, .index = 0};
        } else {
            right_ = std::nullopt;
        }
    } else {
        right_ = SearchRecord{.leaf_node = leaf_node, .index = static_cast<unsigned int>(index)};
    }
}

void BPlusTreeSearcher::ReadPage(unsigned int page_num) {
    ifs_.seekg(static_cast<std::streamoff>(page_num) * page_size_, std::ios::beg);
    buffer_.Load(ifs_);
}

std::shared_ptr<LeafNode> BPlusTreeSearcher::LocateLeafMayContainKey() {
    unsigned int current_level = level_;
    unsigned int next_page_num = root_page_num_;
    while (current_level != 0) {
        auto node_it = internal_node_cache_.find(next_page_num);
        if (node_it == internal_node_cache_.end()) {
            ReadPage(next_page_num);
            node_it = internal_node_cache_.emplace(next_page_num, InternalNode(buffer_)).first;
        }
        const InternalNode& internal_node = node_it->second;

        auto it = std::ranges::upper_bound(internal_node.keys_, query_key_);
        auto index = static_cast<size_t>(std::distance(internal_node.keys_.begin(), it));
        next_page_num = internal_node.pointers_.at(index);

        current_level--;
    }
    return LocateLeafByPageNum(next_page_num);
}

std::shared_ptr<LeafNode> BPlusTreeSearcher::LocateLeafByPageNum(unsigned int page_num) {
    ReadPage(page_num);
    auto new_node_ptr = std::make_shared<LeafNode>(buffer_);
    return new_node_ptr;
}
