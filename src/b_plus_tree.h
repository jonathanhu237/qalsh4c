#pragma once

#include <concepts>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "buffer.h"
#include "global.h"

struct KeyPageNumPair {
    double key;
    unsigned int page_num;
};

struct DotProductPointIdPair {
    double dot_product{0.0};
    unsigned int point_id{0};
};

struct DotProductPointIdPairLess {
    bool operator()(const DotProductPointIdPair& lhs, const DotProductPointIdPair& rhs) const {
        if (lhs.dot_product != rhs.dot_product) {
            return lhs.dot_product < rhs.dot_product;
        }
        return lhs.point_id < rhs.point_id;
    }
};

class InternalNode {
   public:
    friend class BPlusTreeBulkLoader;
    friend class BPlusTreeSearcher;
    friend class QalshAnnSearcher;
    InternalNode(unsigned int order);
    InternalNode(const Buffer& buffer);

   private:
    static size_t GetHeaderSize();
    void Serialize(Buffer& buffer) const;

    // Header
    unsigned int num_children_{0};

    // Data
    std::vector<double> keys_;
    std::vector<unsigned int> pointers_;
};

class LeafNode {
   public:
    friend class BPlusTreeBulkLoader;
    friend class BPlusTreeSearcher;
    friend class QalshAnnSearcher;
    LeafNode(unsigned int order);
    LeafNode(const Buffer& buffer);

   private:
    static size_t GetHeaderSize();
    void Serialize(Buffer& buffer) const;

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

    void Build(const std::vector<DotProductPointIdPair>& data);
    void BuildFromStream(size_t num_entries, std::function<DotProductPointIdPair()> next);

   private:
    unsigned int AllocatePage();
    void WritePage(unsigned int page_num);

    std::ofstream ofs_;
    unsigned int page_size_{0};
    unsigned int num_page_{0};
    unsigned int next_page_num_{0};

    // Header
    unsigned int root_page_num_{0};
    unsigned int level_{0};
    unsigned int internal_node_order_{0};
    unsigned int leaf_node_order_{0};

    // utils
    Buffer buffer_;
};

class BPlusTreeSearcher {
   public:
    BPlusTreeSearcher(const std::filesystem::path& file_path, unsigned int page_size);
    void Init(double query_key);

    template <std::invocable<unsigned int, double> Handler>
    bool IncrementalSearch(double bound, Handler&& handler);

    double get_query_key() const { return query_key_; }

   private:
    template <std::invocable<unsigned int, double> Handler>
    bool IncrementalSearchLeft(double bound, Handler&& handler);

    template <std::invocable<unsigned int, double> Handler>
    bool IncrementalSearchRight(double bound, Handler&& handler);

    void ReadPage(unsigned int page_num);
    std::shared_ptr<LeafNode> LocateLeafMayContainKey();
    std::shared_ptr<LeafNode> LocateLeafByPageNum(unsigned int page_num);

    std::ifstream ifs_;
    unsigned int page_size_{0};
    Buffer buffer_;
    double query_key_{0.0};

    struct SearchRecord {
        std::shared_ptr<LeafNode> leaf_node;
        unsigned int index{0};
    };

    std::optional<SearchRecord> left_;
    std::optional<SearchRecord> right_;
};

// --------------------------------------------------------------------------------
// Template Implementation
// --------------------------------------------------------------------------------
template <std::invocable<unsigned int, double> Handler>
bool BPlusTreeSearcher::IncrementalSearchLeft(double bound, Handler&& handler) {
    if (!left_.has_value()) {
        return true;
    }

    for (unsigned int i = 0; i < Global::kScanSize; i++) {
        auto& [leaf_node, index] = left_.value();
        const double dot_product = leaf_node->keys_[index];
        const unsigned int point_id = leaf_node->values_[index];

        // Check if we've exceeded the search bound (left direction)
        if (query_key_ - dot_product > bound) {
            return true;
        }

        std::forward<Handler>(handler)(point_id, dot_product);

        // Move to next position (left direction)
        if (index > 0) {
            index--;
        } else {
            if (leaf_node->prev_leaf_page_num_ == 0) {
                left_ = std::nullopt;
                return true;
            }
            leaf_node = LocateLeafByPageNum(leaf_node->prev_leaf_page_num_);
            index = leaf_node->num_entries_ - 1;
        }
    }

    return false;
}

template <std::invocable<unsigned int, double> Handler>
bool BPlusTreeSearcher::IncrementalSearchRight(double bound, Handler&& handler) {
    if (!right_.has_value()) {
        return true;
    }

    for (unsigned int i = 0; i < Global::kScanSize; i++) {
        auto& [leaf_node, index] = right_.value();
        const double dot_product = leaf_node->keys_[index];
        const unsigned int point_id = leaf_node->values_[index];

        // Check if we've exceeded the search bound (right direction)
        if (dot_product - query_key_ > bound) {
            return true;
        }

        std::forward<Handler>(handler)(point_id, dot_product);

        // Move to next position (right direction)
        if (index < leaf_node->num_entries_ - 1) {
            index++;
        } else {
            if (leaf_node->next_leaf_page_num_ == 0) {
                right_ = std::nullopt;
                return true;
            }
            leaf_node = LocateLeafByPageNum(leaf_node->next_leaf_page_num_);
            index = 0;
        }
    }

    return false;
}

template <std::invocable<unsigned int, double> Handler>
bool BPlusTreeSearcher::IncrementalSearch(double bound, Handler&& handler) {
    bool left_finished = IncrementalSearchLeft(bound, std::forward<Handler>(handler));
    bool right_finished = IncrementalSearchRight(bound, std::forward<Handler>(handler));
    return left_finished && right_finished;
}
