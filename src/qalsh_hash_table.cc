#include "qalsh_hash_table.h"

#include <algorithm>
#include <cmath>
#include <optional>

InMemoryQalshHashTable::InMemoryQalshHashTable(const std::vector<DotProductPointIdPair>& data) : data_(data) {}

void InMemoryQalshHashTable::Init(double key) {
    key_ = key;
    left_.reset();
    right_.reset();

    // Locate the first key k satisfying k >= key.
    auto it =
        std::ranges::lower_bound(data_, key, {}, [](const DotProductPointIdPair& pair) { return pair.dot_product; });
    auto index = static_cast<size_t>(std::distance(data_.begin(), it));

    // Determine the left index and right index.
    left_ = (index == 0) ? std::nullopt : std::make_optional(index - 1);
    right_ = (index == data_.size()) ? std::nullopt : std::make_optional(index);
}

std::optional<unsigned int> InMemoryQalshHashTable::LeftFindNext(double bound) {
    if (!left_.has_value() || key_ - data_[left_.value()].dot_product > bound) {
        return std::nullopt;
    }

    unsigned int point_id = data_[left_.value()].point_id;
    if (left_.value() == 0) {
        left_.reset();
    }
    left_.value()--;

    return point_id;
}

std::optional<unsigned int> InMemoryQalshHashTable::RightFindNext(double bound) {
    if (!right_.has_value() || data_[right_.value()].dot_product - key_ > bound) {
        return std::nullopt;
    }

    unsigned int point_id = data_[right_.value()].point_id;
    if (right_.value() == data_.size() - 1) {
        right_.reset();
    }
    right_.value()--;

    return point_id;
}
