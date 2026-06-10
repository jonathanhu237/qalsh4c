#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "global.h"

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

class InMemoryQalshTable {
   public:
    InMemoryQalshTable() = default;

    explicit InMemoryQalshTable(std::vector<DotProductPointIdPair> entries) { Build(std::move(entries)); }

    void Build(std::vector<DotProductPointIdPair> entries) {
        entries_ = std::move(entries);
        std::ranges::sort(entries_, DotProductPointIdPairLess{});
    }

    void Init(double query_key) {
        query_key_ = query_key;

        auto it = std::lower_bound(entries_.begin(), entries_.end(), query_key_,
                                   [](const DotProductPointIdPair& entry, double key) {
                                       return entry.dot_product < key;
                                   });
        const size_t index = static_cast<size_t>(std::distance(entries_.begin(), it));

        left_ = index == 0 ? std::nullopt : std::optional<size_t>{index - 1};
        right_ = index == entries_.size() ? std::nullopt : std::optional<size_t>{index};
    }

    template <std::invocable<unsigned int, double> Handler>
    bool IncrementalSearch(double bound, Handler&& handler);

   private:
    template <std::invocable<unsigned int, double> Handler>
    bool IncrementalSearchLeft(double bound, Handler&& handler);

    template <std::invocable<unsigned int, double> Handler>
    bool IncrementalSearchRight(double bound, Handler&& handler);

    std::vector<DotProductPointIdPair> entries_;
    double query_key_{0.0};
    std::optional<size_t> left_;
    std::optional<size_t> right_;
};

template <std::invocable<unsigned int, double> Handler>
bool InMemoryQalshTable::IncrementalSearchLeft(double bound, Handler&& handler) {
    if (!left_.has_value()) {
        return true;
    }

    for (unsigned int scan_count = 0; scan_count < Global::kScanSize; ++scan_count) {
        const size_t index = *left_;
        const DotProductPointIdPair& entry = entries_[index];
        if (query_key_ - entry.dot_product > bound) {
            return true;
        }

        std::forward<Handler>(handler)(entry.point_id, entry.dot_product);

        if (index == 0) {
            left_ = std::nullopt;
            return true;
        }
        left_ = index - 1;
    }

    return false;
}

template <std::invocable<unsigned int, double> Handler>
bool InMemoryQalshTable::IncrementalSearchRight(double bound, Handler&& handler) {
    if (!right_.has_value()) {
        return true;
    }

    for (unsigned int scan_count = 0; scan_count < Global::kScanSize; ++scan_count) {
        const size_t index = *right_;
        const DotProductPointIdPair& entry = entries_[index];
        if (entry.dot_product - query_key_ > bound) {
            return true;
        }

        std::forward<Handler>(handler)(entry.point_id, entry.dot_product);

        if (index + 1 >= entries_.size()) {
            right_ = std::nullopt;
            return true;
        }
        right_ = index + 1;
    }

    return false;
}

template <std::invocable<unsigned int, double> Handler>
bool InMemoryQalshTable::IncrementalSearch(double bound, Handler&& handler) {
    bool left_finished = IncrementalSearchLeft(bound, std::forward<Handler>(handler));
    bool right_finished = IncrementalSearchRight(bound, std::forward<Handler>(handler));
    return left_finished && right_finished;
}
