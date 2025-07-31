#include "qalsh_hash_table.h"

#include <algorithm>
#include <cmath>
#include <optional>

InmemoryQalshHashTable::InmemoryQalshHashTable(const std::vector<DotProductPointIdPair>& data) : data_(data) {}

void InmemoryQalshHashTable::Init(double key) {
    key_ = key;
    left_.reset();
    right_.reset();

    // Locate the first key k satisfying k >= key.
    auto it =
        std::ranges::lower_bound(data_, key, {}, [](const DotProductPointIdPair& pair) { return pair.dot_product; });
    auto index = static_cast<size_t>(std::distance(data_.begin(), it));

    // Determine the left index and right index.
    left_ = (index == 0) ? std::nullopt
                         : std::make_optional(SearchRecord{
                               .distance = key_ - data_[index - 1].dot_product,
                               .index = index - 1,
                           });
    right_ = (index == data_.size()) ? std::nullopt
                                     : std::make_optional(SearchRecord{
                                           .distance = data_[index].dot_product - key_,
                                           .index = index,
                                       });
}

std::optional<unsigned int> InmemoryQalshHashTable::FindNext(double bound) {
    if (!left_.has_value() && !right_.has_value()) {
        return std::nullopt;
    }
    if (!left_.has_value()) {
        return FindNextRight(bound);
    }
    if (!right_.has_value()) {
        return FindNextLeft(bound);
    }

    if (left_->distance <= right_->distance) {
        return FindNextLeft(bound);
    }
    return FindNextRight(bound);
}

std::optional<unsigned int> InmemoryQalshHashTable::FindNextLeft(double bound) {
    if (!left_.has_value() || left_->distance > bound) {
        return std::nullopt;
    }

    unsigned int point_id = data_[left_->index].point_id;
    if (left_->index == 0) {
        left_.reset();
    }
    left_->index--;

    return point_id;
}

std::optional<unsigned int> InmemoryQalshHashTable::FindNextRight(double bound) {
    if (!right_.has_value() || right_->distance > bound) {
        return std::nullopt;
    }

    unsigned int point_id = data_[right_->index].point_id;
    if (right_->index == data_.size() - 1) {
        right_.reset();
    }
    right_->index++;

    return point_id;
}