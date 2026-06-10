#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
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

struct InMemoryQalshTableCursor {
    double query_key{0.0};
    size_t left_index{0};
    size_t right_index{0};
    bool has_left{false};
    bool has_right{false};
};

class InMemoryQalshTable {
   public:
    InMemoryQalshTable() = default;

    explicit InMemoryQalshTable(std::vector<DotProductPointIdPair> entries) { Build(std::move(entries)); }

    void Build(std::vector<DotProductPointIdPair> entries) {
        entries_ = std::move(entries);
        std::ranges::sort(entries_, DotProductPointIdPairLess{});
    }

    [[nodiscard]] InMemoryQalshTableCursor InitCursor(double query_key) const {
        auto it = std::lower_bound(entries_.begin(), entries_.end(), query_key,
                                   [](const DotProductPointIdPair& entry, double key) {
                                       return entry.dot_product < key;
                                   });
        const size_t index = static_cast<size_t>(std::distance(entries_.begin(), it));

        return InMemoryQalshTableCursor{
            .query_key = query_key,
            .left_index = index == 0 ? 0 : index - 1,
            .right_index = index,
            .has_left = index > 0,
            .has_right = index < entries_.size(),
        };
    }

    template <std::invocable<unsigned int> Handler>
    bool IncrementalSearch(InMemoryQalshTableCursor& cursor, double bound, Handler&& handler) const;

   private:
    template <std::invocable<unsigned int> Handler>
    bool IncrementalSearchLeft(InMemoryQalshTableCursor& cursor, double bound, Handler&& handler) const;

    template <std::invocable<unsigned int> Handler>
    bool IncrementalSearchRight(InMemoryQalshTableCursor& cursor, double bound, Handler&& handler) const;

    std::vector<DotProductPointIdPair> entries_;
};

template <std::invocable<unsigned int> Handler>
bool InMemoryQalshTable::IncrementalSearchLeft(InMemoryQalshTableCursor& cursor, double bound,
                                               Handler&& handler) const {
    if (!cursor.has_left) {
        return true;
    }

    for (unsigned int scan_count = 0; scan_count < Global::kScanSize; ++scan_count) {
        const size_t index = cursor.left_index;
        const DotProductPointIdPair& entry = entries_[index];
        if (cursor.query_key - entry.dot_product > bound) {
            return true;
        }

        std::forward<Handler>(handler)(entry.point_id);

        if (index == 0) {
            cursor.has_left = false;
            return true;
        }
        cursor.left_index = index - 1;
    }

    return false;
}

template <std::invocable<unsigned int> Handler>
bool InMemoryQalshTable::IncrementalSearchRight(InMemoryQalshTableCursor& cursor, double bound,
                                                Handler&& handler) const {
    if (!cursor.has_right) {
        return true;
    }

    for (unsigned int scan_count = 0; scan_count < Global::kScanSize; ++scan_count) {
        const size_t index = cursor.right_index;
        const DotProductPointIdPair& entry = entries_[index];
        if (entry.dot_product - cursor.query_key > bound) {
            return true;
        }

        std::forward<Handler>(handler)(entry.point_id);

        if (index + 1 >= entries_.size()) {
            cursor.has_right = false;
            return true;
        }
        cursor.right_index = index + 1;
    }

    return false;
}

template <std::invocable<unsigned int> Handler>
bool InMemoryQalshTable::IncrementalSearch(InMemoryQalshTableCursor& cursor, double bound, Handler&& handler) const {
    bool left_finished = IncrementalSearchLeft(cursor, bound, std::forward<Handler>(handler));
    bool right_finished = IncrementalSearchRight(cursor, bound, std::forward<Handler>(handler));
    return left_finished && right_finished;
}
