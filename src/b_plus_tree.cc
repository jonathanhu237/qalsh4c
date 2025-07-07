#include "b_plus_tree.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <format>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utils.h"

namespace qalsh_chamfer {

// ---------- InternalNode Implementation ----------
InternalNode::InternalNode() : num_children_(0) {}

InternalNode::InternalNode(unsigned int order) : num_children_(0) {
    keys_.reserve(order - 1);
    pointers_.reserve(order);
};

InternalNode::InternalNode(const std::vector<char>& buffer) : num_children_(0) {
    size_t offset = 0;

    num_children_ = Utils::ReadFromBuffer<unsigned int>(buffer, offset);
    keys_.reserve(num_children_ - 1);
    pointers_.reserve(num_children_);

    for (unsigned int i = 0; i < num_children_ - 1; ++i) {
        keys_.push_back(Utils::ReadFromBuffer<double>(buffer, offset));
    }
    for (unsigned int i = 0; i < num_children_; ++i) {
        pointers_.push_back(Utils::ReadFromBuffer<unsigned int>(buffer, offset));
    }
};

auto InternalNode::GetHeaderSize() -> size_t { return sizeof(num_children_); }

auto InternalNode::Serialize(std::vector<char>& buffer) const -> void {
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
LeafNode::LeafNode() : num_entries_(0), prev_leaf_page_num_(0), next_leaf_page_num_(0) {};

LeafNode::LeafNode(unsigned int order) : num_entries_(0), prev_leaf_page_num_(0), next_leaf_page_num_(0) {
    keys_.reserve(order);
    values_.reserve(order);
};

LeafNode::LeafNode(const std::vector<char>& buffer) : num_entries_(0), prev_leaf_page_num_(0), next_leaf_page_num_(0) {
    size_t offset = 0;

    num_entries_ = Utils::ReadFromBuffer<unsigned int>(buffer, offset);
    keys_.reserve(num_entries_);
    values_.reserve(num_entries_);
    prev_leaf_page_num_ = Utils::ReadFromBuffer<unsigned int>(buffer, offset);
    next_leaf_page_num_ = Utils::ReadFromBuffer<unsigned int>(buffer, offset);

    for (unsigned int i = 0; i < num_entries_; ++i) {
        keys_.push_back(Utils::ReadFromBuffer<double>(buffer, offset));
    }
    for (unsigned int i = 0; i < num_entries_; ++i) {
        values_.push_back(Utils::ReadFromBuffer<unsigned int>(buffer, offset));
    }
};

auto LeafNode::GetHeaderSize() -> size_t {
    return sizeof(num_entries_) + sizeof(prev_leaf_page_num_) + sizeof(next_leaf_page_num_);
}

auto LeafNode::Serialize(std::vector<char>& buffer) const -> void {
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
BPlusTreeBulkLoader::BPlusTreeBulkLoader(const fs::path& file_path, unsigned int page_size)
    : page_size_(page_size),
      num_page_(0),
      next_page_num_(0),
      root_page_num_(0),
      level_(0),
      internal_node_order_(0),
      leaf_node_order_(0) {
    ofs_.open(file_path, std::ios::binary | std::ios::trunc);
    if (!ofs_) {
        throw std::runtime_error(std::format("Failed to open file: {}", file_path.string()));
    }

    internal_node_order_ = static_cast<unsigned int>((page_size - InternalNode::GetHeaderSize() + sizeof(double)) /
                                                     (sizeof(double) + sizeof(unsigned int)));
    leaf_node_order_ =
        static_cast<unsigned int>((page_size - LeafNode::GetHeaderSize()) / (sizeof(double) + sizeof(unsigned int)));
}

auto BPlusTreeBulkLoader::Build(const std::vector<KeyValuePair>& data) -> void {
    std::vector<KeyValuePair> parent_level_entries;

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
            new_leaf_node.keys_.push_back(data[i].first);
            new_leaf_node.values_.push_back(data[i].second);
            new_leaf_node.num_entries_++;
        }
        data_idx = chunk_end;

        // Serialize the leaf node and write it to the file
        new_leaf_page_num = AllocatePage();
        new_leaf_node.next_leaf_page_num_ = (data_idx < data.size()) ? next_page_num_ : 0;
        std::vector<char> buffer(page_size_, 0);
        new_leaf_node.Serialize(buffer);
        WritePage(new_leaf_page_num, buffer);

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
            new_internal_page_num = AllocatePage();
            std::vector<char> buffer(page_size_, 0);
            new_internal_node.Serialize(buffer);
            WritePage(new_internal_page_num, buffer);

            // Add entry to the next parent level
            next_parent_level_entries.emplace_back(separator_key_for_next_level, new_internal_page_num);

            // Update the parent level entries
            parent_level_entries = next_parent_level_entries;
        }
    }

    root_page_num_ = new_internal_page_num == 0 ? root_page_num_ : new_internal_page_num;

    // write the header
    std::vector<char> buffer(page_size_, 0);
    size_t offset = 0;

    Utils::WriteToBuffer(buffer, offset, root_page_num_);
    Utils::WriteToBuffer(buffer, offset, level_);
    Utils::WriteToBuffer(buffer, offset, internal_node_order_);
    Utils::WriteToBuffer(buffer, offset, leaf_node_order_);

    WritePage(0, buffer);
}

auto BPlusTreeBulkLoader::AllocatePage() -> unsigned int {
    unsigned int new_page_num = next_page_num_++;

    ofs_.seekp(static_cast<std::streamoff>(new_page_num * page_size_));
    std::vector<char> empty_page(page_size_, 0);
    ofs_.write(empty_page.data(), static_cast<std::streamsize>(empty_page.size()));
    num_page_++;

    return new_page_num;
}

auto BPlusTreeBulkLoader::WritePage(unsigned int page_num, const std::vector<char>& buffer) -> void {
    ofs_.seekp(static_cast<std::streamoff>(page_num * page_size_));
    ofs_.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
}

// ---------- BPlusTreeSearcher Implementation ----------
BPlusTreeSearcher::BPlusTreeSearcher(const fs::path& file_path, unsigned int page_size, double key)
    : page_size_(page_size), key_(key), root_page_num_(0), level_(0), internal_node_order_(0), leaf_node_order_(0) {
    // Open the B+ tree file
    ifs_.open(file_path, std::ios::binary);
    if (!ifs_.is_open()) {
        throw std::runtime_error(std::format("Failed to open file: {}", file_path.string()));
    }

    // Read the header from page 0
    std::vector<char> header_buffer = ReadPage(0);
    size_t offset = 0;
    root_page_num_ = Utils::ReadFromBuffer<unsigned int>(header_buffer, offset);
    level_ = Utils::ReadFromBuffer<unsigned int>(header_buffer, offset);
    internal_node_order_ = Utils::ReadFromBuffer<unsigned int>(header_buffer, offset);
    leaf_node_order_ = Utils::ReadFromBuffer<unsigned int>(header_buffer, offset);

    // Locate the first key k satisfying k >= key in the B+ tree
    LeafNode leaf_node = LocateLeafMayContainKey();
    auto it = std::ranges::lower_bound(leaf_node.keys_, key);
    auto index = static_cast<size_t>(std::distance(leaf_node.keys_.begin(), it));

    // Determine the left search location
    if (index == 0) {
        if (leaf_node.prev_leaf_page_num_ != 0) {
            LeafNode prev_leaf_node = LocateLeafByPageNum(leaf_node.prev_leaf_page_num_);
            left_search_location_ = std::make_pair(prev_leaf_node, prev_leaf_node.num_entries_ - 1);
        }
    } else {
        left_search_location_ = std::make_pair(leaf_node, index - 1);
    }

    // Determine the right search location
    if (index == leaf_node.keys_.size()) {
        if (leaf_node.next_leaf_page_num_ != 0) {
            LeafNode next_leaf_node = LocateLeafByPageNum(leaf_node.next_leaf_page_num_);
            right_search_location_ = std::make_pair(next_leaf_node, 0);
        }
    } else {
        right_search_location_ = std::make_pair(leaf_node, index);
    }
}

auto BPlusTreeSearcher::IncrementalSearch(double bound) -> std::vector<unsigned int> {
    std::vector<unsigned int> result;

    // Search to the left
    while (left_search_location_) {
        auto& [leaf_node, index] = *left_search_location_;

        if (std::fabs(leaf_node.keys_.at(index) - key_) > bound) {
            break;
        }

        result.push_back(leaf_node.values_.at(index));

        if (index > 0) {
            index--;
        } else {
            if (leaf_node.prev_leaf_page_num_ == 0) {
                left_search_location_.reset();
            } else {
                leaf_node = LocateLeafByPageNum(leaf_node.prev_leaf_page_num_);
                index = leaf_node.num_entries_ - 1;
            }
        }
    }

    // Search to the right
    while (right_search_location_) {
        auto& [leaf_node, index] = *right_search_location_;

        if (std::fabs(leaf_node.keys_.at(index) - key_) > bound) {
            break;
        }

        result.push_back(leaf_node.values_.at(index));

        if (index < leaf_node.num_entries_ - 1) {
            index++;
        } else {
            if (leaf_node.next_leaf_page_num_ == 0) {
                right_search_location_.reset();
            } else {
                leaf_node = LocateLeafByPageNum(leaf_node.next_leaf_page_num_);
                index = 0;
            }
        }
    }

    return result;
}

auto BPlusTreeSearcher::LocateLeafMayContainKey() -> LeafNode {
    unsigned int current_level = level_;
    unsigned int next_page_num = root_page_num_;

    while (current_level != 0) {
        const std::vector<char> buffer = ReadPage(next_page_num);
        InternalNode internal_node(buffer);

        auto it = std::ranges::upper_bound(internal_node.keys_, key_);
        auto index = static_cast<size_t>(std::distance(internal_node.keys_.begin(), it));
        next_page_num = internal_node.pointers_.at(index);

        current_level--;
    }

    return LocateLeafByPageNum(next_page_num);
}

auto BPlusTreeSearcher::LocateLeafByPageNum(unsigned int page_num) -> LeafNode {
    const std::vector<char> buffer = ReadPage(page_num);
    LeafNode leaf_node(buffer);
    return leaf_node;
}

auto BPlusTreeSearcher::ReadPage(unsigned int page_num) -> std::vector<char> {
    std::vector<char> buffer(page_size_, 0);
    ifs_.seekg(static_cast<std::streamoff>(page_num * page_size_));
    ifs_.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return buffer;
}

}  // namespace qalsh_chamfer