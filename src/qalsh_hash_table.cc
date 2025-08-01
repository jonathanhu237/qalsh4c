#include "qalsh_hash_table.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <optional>

#include "utils.h"

// ---------------------------------------------
// InMemoryQalshHashTable Implementation
// ---------------------------------------------
InMemoryQalshHashTable::InMemoryQalshHashTable(const std::vector<DotProductPointIdPair>& data) : data_(data) {}

void InMemoryQalshHashTable::Init(double key) {
    key_ = key;
    left_.reset();
    right_.reset();
    while (!pq_.empty()) {
        pq_.pop();
    }

    // Locate the first key k satisfying k >= key.
    auto it =
        std::ranges::lower_bound(data_, key, {}, [](const DotProductPointIdPair& pair) { return pair.dot_product; });
    auto index = static_cast<size_t>(std::distance(data_.begin(), it));

    // Determine the left index and right index.
    left_ = (index == 0) ? std::nullopt : std::make_optional(index - 1);
    if (left_.has_value()) {
        pq_.emplace(SearchRecord{
            .is_left = true,
            .dist = key_ - data_[left_.value()].dot_product,
        });
    }

    right_ = (index == data_.size()) ? std::nullopt : std::make_optional(index);
    if (right_.has_value()) {
        pq_.emplace(SearchRecord{
            .is_left = false,
            .dist = data_[right_.value()].dot_product - key_,
        });
    }
}

std::optional<unsigned int> InMemoryQalshHashTable::FindNext(double bound) {
    if (pq_.empty() || pq_.top().dist > bound) {
        return std::nullopt;
    }

    if (pq_.top().is_left) {
        unsigned int point_id = data_[left_.value()].point_id;
        pq_.pop();
        if (left_.value() == 0) {
            left_.reset();
        } else {
            left_.value()--;
            pq_.emplace(SearchRecord{
                .is_left = true,
                .dist = key_ - data_[left_.value()].dot_product,
            });
        }

        return point_id;
    }

    unsigned int point_id = data_[right_.value()].point_id;
    pq_.pop();
    if (right_.value() == data_.size() - 1) {
        right_.reset();
    } else {
        right_.value()++;
        pq_.emplace(SearchRecord{
            .is_left = false,
            .dist = data_[right_.value()].dot_product - key_,
        });
    }

    return point_id;
}

std::optional<unsigned int> InMemoryQalshHashTable::LeftFindNext(double bound) {
    if (!left_.has_value() || key_ - data_[left_.value()].dot_product > bound) {
        return std::nullopt;
    }

    unsigned int point_id = data_[left_.value()].point_id;
    if (left_.value() == 0) {
        left_.reset();
    } else {
        left_.value()--;
    }

    return point_id;
}

std::optional<unsigned int> InMemoryQalshHashTable::RightFindNext(double bound) {
    if (!right_.has_value() || data_[right_.value()].dot_product - key_ > bound) {
        return std::nullopt;
    }

    unsigned int point_id = data_[right_.value()].point_id;
    if (right_.value() == data_.size() - 1) {
        right_.reset();
    } else {
        right_.value()++;
    }

    return point_id;
}

// ---------------------------------------------
// DiskQalshHashTable Implementation
// ---------------------------------------------
DiskQalshHashTable::DiskQalshHashTable(const std::filesystem::path& file_path, unsigned int page_size)
    : page_size_(page_size) {
    // Open the B+ tree file
    ifs_.open(file_path, std::ios::binary);
    if (!ifs_.is_open()) {
        spdlog::error("Failed to open file: {}", file_path.string());
    }

    // Read the header from page 0
    std::vector<char> header_buffer = ReadPage(0);
    size_t offset = 0;
    root_page_num_ = Utils::ReadFromBuffer<unsigned int>(header_buffer, offset);
    level_ = Utils::ReadFromBuffer<unsigned int>(header_buffer, offset);
    internal_node_order_ = Utils::ReadFromBuffer<unsigned int>(header_buffer, offset);
    leaf_node_order_ = Utils::ReadFromBuffer<unsigned int>(header_buffer, offset);
}

void DiskQalshHashTable::Init(double key) {
    key_ = key;
    left_.reset();
    right_.reset();
    while (!pq_.empty()) {
        pq_.pop();
    }
    page_cache_.clear();

    // Locate the first key k satisfying k >= key in the B+ tree
    LeafNode leaf_node = LocateLeafMayContainKey();
    auto it = std::ranges::lower_bound(leaf_node.keys_, key);
    auto index = static_cast<size_t>(std::distance(leaf_node.keys_.begin(), it));

    // Determine the left search location
    if (index == 0) {
        if (leaf_node.prev_leaf_page_num_ != 0) {
            LeafNode prev_leaf_node = LocateLeafByPageNum(leaf_node.prev_leaf_page_num_);
            left_ = std::make_optional(
                SearchLocation{.leaf_node = prev_leaf_node, .index = prev_leaf_node.num_entries_ - 1});
        }
    } else {
        left_ = std::make_optional(SearchLocation{.leaf_node = leaf_node, .index = index - 1});
    }
    if (left_.has_value()) {
        pq_.emplace(SearchRecord{
            .is_left = true,
            .dist = key_ - left_->leaf_node.keys_[left_->index],
        });
    }

    // Determine the right search location
    if (index == leaf_node.keys_.size()) {
        if (leaf_node.next_leaf_page_num_ != 0) {
            LeafNode next_leaf_node = LocateLeafByPageNum(leaf_node.next_leaf_page_num_);
            right_ = std::make_optional(SearchLocation{.leaf_node = next_leaf_node, .index = 0});
        }
    } else {
        right_ = std::make_optional(SearchLocation{.leaf_node = leaf_node, .index = index});
    }
    if (right_.has_value()) {
        pq_.emplace(SearchRecord{
            .is_left = false,
            .dist = right_->leaf_node.keys_[right_->index] - key_,
        });
    }
}

std::optional<unsigned int> DiskQalshHashTable::FindNext(double bound) {
    if (pq_.empty() || pq_.top().dist > bound) {
        return std::nullopt;
    }

    if (pq_.top().is_left) {
        auto& [leaf_node, index] = left_.value();
        unsigned int point_id = leaf_node.values_[index];
        pq_.pop();
        if (index > 0) {
            index--;
        } else {
            if (leaf_node.prev_leaf_page_num_ == 0) {
                left_.reset();
            } else {
                leaf_node = LocateLeafByPageNum(leaf_node.prev_leaf_page_num_);
                index = leaf_node.num_entries_ - 1;
            }
        }
        if (left_.has_value()) {
            pq_.emplace(SearchRecord{
                .is_left = true,
                .dist = key_ - leaf_node.keys_.at(index),
            });
        }

        return point_id;
    }

    auto& [leaf_node, index] = right_.value();
    unsigned int point_id = leaf_node.values_[index];
    pq_.pop();
    if (index < leaf_node.num_entries_ - 1) {
        index++;
    } else {
        if (leaf_node.next_leaf_page_num_ == 0) {
            right_.reset();
        } else {
            leaf_node = LocateLeafByPageNum(leaf_node.next_leaf_page_num_);
            index = 0;
        }
    }
    if (right_.has_value()) {
        pq_.emplace(SearchRecord{
            .is_left = false,
            .dist = leaf_node.keys_.at(index) - key_,
        });
    }

    return point_id;
}

std::optional<unsigned int> DiskQalshHashTable::LeftFindNext(double bound) {
    if (!left_.has_value()) {
        return std::nullopt;
    }

    auto& [leaf_node, index] = left_.value();
    if (key_ - leaf_node.keys_.at(index) > bound) {
        return std::nullopt;
    }

    unsigned int point_id = leaf_node.values_[index];
    if (index > 0) {
        index--;
    } else {
        if (leaf_node.prev_leaf_page_num_ == 0) {
            left_.reset();
        } else {
            leaf_node = LocateLeafByPageNum(leaf_node.prev_leaf_page_num_);
            index = leaf_node.num_entries_ - 1;
        }
    }

    return point_id;
}

std::optional<unsigned int> DiskQalshHashTable::RightFindNext(double bound) {
    if (!right_.has_value()) {
        return std::nullopt;
    }

    auto& [leaf_node, index] = right_.value();
    if (leaf_node.keys_.at(index) - key_ > bound) {
        return std::nullopt;
    }

    unsigned int point_id = leaf_node.values_[index];
    if (index < leaf_node.num_entries_ - 1) {
        index++;
    } else {
        if (leaf_node.next_leaf_page_num_ == 0) {
            right_.reset();
        } else {
            leaf_node = LocateLeafByPageNum(leaf_node.next_leaf_page_num_);
            index = 0;
        }
    }

    return point_id;
}

LeafNode DiskQalshHashTable::LocateLeafMayContainKey() {
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

LeafNode DiskQalshHashTable::LocateLeafByPageNum(unsigned int page_num) {
    const std::vector<char> buffer = ReadPage(page_num);
    LeafNode leaf_node(buffer);
    return leaf_node;
}

std::vector<char> DiskQalshHashTable::ReadPage(unsigned int page_num) {
    // Check cache first
    auto cache_it = page_cache_.find(page_num);
    if (cache_it != page_cache_.end()) {
        return cache_it->second;
    }

    // Read from disk and cache
    page_cache_[page_num].resize(page_size_);
    ifs_.seekg(static_cast<std::streamoff>(page_num * page_size_));
    ifs_.read(page_cache_[page_num].data(), static_cast<std::streamsize>(page_size_));

    return page_cache_[page_num];
}