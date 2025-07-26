#include "qalsh_searcher.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "b_plus_tree.h"
#include "point_set.h"
#include "utils.h"

// ---------------------------------------------
// BPlusTreeSearcher Implementation
// ---------------------------------------------
BPlusTreeSearcher::BPlusTreeSearcher(const std::filesystem::path& file_path, unsigned int page_size)
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

void BPlusTreeSearcher::Init(double key) {
    key_ = key;
    left_search_location_.reset();
    right_search_location_.reset();

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

std::vector<unsigned int> BPlusTreeSearcher::IncrementalSearch(double bound) {
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

LeafNode BPlusTreeSearcher::LocateLeafMayContainKey() {
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

LeafNode BPlusTreeSearcher::LocateLeafByPageNum(unsigned int page_num) {
    const std::vector<char> buffer = ReadPage(page_num);
    LeafNode leaf_node(buffer);
    return leaf_node;
}

std::vector<char> BPlusTreeSearcher::ReadPage(unsigned int page_num) {
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

// ---------------------------------------------
// InMemorySearcher Implementation
// ---------------------------------------------
InMemorySearcher::InMemorySearcher(PointSetReader* base_reader, std::vector<double> dot_vector) {
    data_.reserve(base_reader->get_num_points());
    for (unsigned int i = 0; i < base_reader->get_num_points(); i++) {
        PointVariant point = base_reader->GetPoint(i);
        double dot_product = 0.0;
        std::visit([&](const auto& concrete_point) { dot_product = Utils::DotProduct(concrete_point, dot_vector); },
                   point);
        data_.emplace_back(dot_product, i);
    }
    std::ranges::sort(data_);
}

void InMemorySearcher::Init(double key) {
    key_ = key;
    left_index_.reset();
    right_index_.reset();

    // Locate the first key k satisfying k >= key.
    auto it = std::ranges::lower_bound(data_, key, {}, [](const KeyValuePair& kvp) { return kvp.first; });
    auto index = static_cast<size_t>(std::distance(data_.begin(), it));

    // Determine the left index and right index.
    left_index_ = (index == 0) ? std::nullopt : std::make_optional(index - 1);
    right_index_ = (index == data_.size()) ? std::nullopt : std::make_optional(index);
}

std::vector<unsigned int> InMemorySearcher::IncrementalSearch(double bound) {
    std::vector<unsigned int> results;

    // Search to the left.
    while (left_index_.has_value()) {
        if (std::fabs(data_[left_index_.value()].first - key_) > bound) {
            break;
        }

        results.push_back(data_[left_index_.value()].second);
        if (left_index_.value() == 0) {
            left_index_.reset();
        } else {
            --(left_index_.value());
        }
    }

    // Search to the right.
    while (right_index_.has_value()) {
        if (std::fabs(data_[right_index_.value()].first - key_) > bound) {
            break;
        }

        results.push_back(data_[right_index_.value()].second);
        if (right_index_.value() == data_.size() - 1) {
            right_index_.reset();
        } else {
            ++(right_index_.value());
        }
    }

    return results;
}