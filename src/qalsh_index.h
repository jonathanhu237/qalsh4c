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

struct InMemoryQalshTableCursor {
    double query_key{0.0};
    std::optional<size_t> left;
    std::optional<size_t> right;
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
            .left = index == 0 ? std::nullopt : std::optional<size_t>{index - 1},
            .right = index == entries_.size() ? std::nullopt : std::optional<size_t>{index},
        };
    }

    template <std::invocable<unsigned int, double> Handler>
    bool IncrementalSearch(InMemoryQalshTableCursor& cursor, double bound, Handler&& handler) const;

   private:
    template <std::invocable<unsigned int, double> Handler>
    bool IncrementalSearchLeft(InMemoryQalshTableCursor& cursor, double bound, Handler&& handler) const;

    template <std::invocable<unsigned int, double> Handler>
    bool IncrementalSearchRight(InMemoryQalshTableCursor& cursor, double bound, Handler&& handler) const;

    std::vector<DotProductPointIdPair> entries_;
};

template <std::invocable<unsigned int, double> Handler>
bool InMemoryQalshTable::IncrementalSearchLeft(InMemoryQalshTableCursor& cursor, double bound,
                                               Handler&& handler) const {
    if (!cursor.left.has_value()) {
        return true;
    }

    const DotProductPointIdPair* entries = entries_.data();
    for (unsigned int scan_count = 0; scan_count < Global::kScanSize; ++scan_count) {
        const size_t index = *cursor.left;
        const DotProductPointIdPair& entry = entries[index];
        if (cursor.query_key - entry.dot_product > bound) {
            return true;
        }

        std::forward<Handler>(handler)(entry.point_id, entry.dot_product);

        if (index == 0) {
            cursor.left = std::nullopt;
            return true;
        }
        cursor.left = index - 1;
    }

    return false;
}

template <std::invocable<unsigned int, double> Handler>
bool InMemoryQalshTable::IncrementalSearchRight(InMemoryQalshTableCursor& cursor, double bound,
                                                Handler&& handler) const {
    if (!cursor.right.has_value()) {
        return true;
    }

    const DotProductPointIdPair* entries = entries_.data();
    const size_t num_entries = entries_.size();
    for (unsigned int scan_count = 0; scan_count < Global::kScanSize; ++scan_count) {
        const size_t index = *cursor.right;
        const DotProductPointIdPair& entry = entries[index];
        if (entry.dot_product - cursor.query_key > bound) {
            return true;
        }

        std::forward<Handler>(handler)(entry.point_id, entry.dot_product);

        if (index + 1 >= num_entries) {
            cursor.right = std::nullopt;
            return true;
        }
        cursor.right = index + 1;
    }

    return false;
}

template <std::invocable<unsigned int, double> Handler>
bool InMemoryQalshTable::IncrementalSearch(InMemoryQalshTableCursor& cursor, double bound, Handler&& handler) const {
    bool left_finished = IncrementalSearchLeft(cursor, bound, std::forward<Handler>(handler));
    bool right_finished = IncrementalSearchRight(cursor, bound, std::forward<Handler>(handler));
    return left_finished && right_finished;
}
