#include "ann_searcher.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <vector>

#include "b_plus_tree.h"
#include "global.h"
#include "types.h"
#include "utils.h"

// ---------------------------------------------
// LinearScanAnnSearcher Implementation
// ---------------------------------------------
void LinearScanAnnSearcher::Init(const PointSetMetadata& base_metadata, double norm_order) {
    if (base_file_.is_open()) {
        base_file_.close();
    }
    base_file_.open(base_metadata.file_path, std::ios::binary);
    if (!base_file_.is_open()) {
        spdlog::error("Failed to open base file: {}", base_metadata.file_path.string());
        return;
    }
    num_points_ = base_metadata.num_points;
    num_dimensions_ = base_metadata.num_dimensions;
    norm_order_ = norm_order;
}

AnnResult LinearScanAnnSearcher::Search(const Point& query_point) {
    AnnResult result{.distance = std::numeric_limits<double>::max(), .point_id = 0};

    for (unsigned int i = 0; i < num_points_; i++) {
        double distance = Calc::LpDistance(Utils::ReadPoint(base_file_, num_dimensions_, i), query_point, norm_order_);
        if (distance < result.distance) {
            result.point_id = i;
            result.distance = distance;
        }
    }

    return result;
}

// ---------------------------------------------
// QalshAnnSearcher Implementation
// ---------------------------------------------
void QalshAnnSearcher::Init(const PointSetMetadata& base_metadata, double norm_order) {
    // Open the base file.
    if (base_file_.is_open()) {
        base_file_.close();
    }
    base_file_.open(base_metadata.file_path, std::ios::binary);
    if (!base_file_.is_open()) {
        spdlog::error("Failed to open base file: {}", base_metadata.file_path.string());
        return;
    }
    num_points_ = base_metadata.num_points;
    num_dimensions_ = base_metadata.num_dimensions;
    norm_order_ = norm_order;

    // Load QALSH configuration.
    std::string stem = base_metadata.file_path.stem();
    std::filesystem::path index_directory =
        base_metadata.file_path.parent_path() / "index" / std::format("l{}", norm_order_) / stem;
    qalsh_config_ = Utils::LoadQalshConfig(index_directory / "config.json");

    // Print the QalshConfig parameters.
    spdlog::info(
        "QALSH Configuration:\n"
        "\tApproximation Ratio: {}\n"
        "\tBucket Width: {}\n"
        "\tError Probability: {}\n"
        "\tNumber of Hash Tables: {}\n"
        "\tCollision Threshold: {}\n"
        "\tPage Size: {}",
        qalsh_config_.approximation_ratio, qalsh_config_.bucket_width, qalsh_config_.error_probability,
        qalsh_config_.num_hash_tables, qalsh_config_.collision_threshold, qalsh_config_.page_size);

    // Initialize the buffer.
    buffer_.clear();
    buffer_.resize(qalsh_config_.page_size);

    // Open the hash tables.
    hash_tables_.clear();
    hash_tables_.reserve(qalsh_config_.num_hash_tables);
    std::filesystem::path b_plus_tree_directory = index_directory / "b_plus_trees";
    for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
        std::ifstream ifs(b_plus_tree_directory / std::format("{}.bin", i), std::ios::binary);
        if (!ifs.is_open()) {
            spdlog::error("Failed to open hash table file: {}",
                          (b_plus_tree_directory / std::format("{}.bin", i)).string());
            return;
        }
        hash_tables_.emplace_back(std::move(ifs));
    }

    // Load the dot vectors.
    std::ifstream dot_vector_file(index_directory / "dot_vectors.bin", std::ios::binary);
    if (!dot_vector_file.is_open()) {
        spdlog::error("Failed to open dot vectors file: {}", (index_directory / "dot_vectors.bin").string());
        return;
    }
    dot_vectors_.clear();
    dot_vectors_.resize(qalsh_config_.num_hash_tables);
    for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
        dot_vectors_[i].resize(num_dimensions_);
        dot_vector_file.read(reinterpret_cast<char*>(dot_vectors_[i].data()),
                             static_cast<std::streamsize>(num_dimensions_ * sizeof(Coordinate)));
    }
}

// NOLINTBEGIN(readability-function-cognitive-complexity)
AnnResult QalshAnnSearcher::Search(const Point& query_point) {
    std::vector<unsigned int> collision_count(num_points_, 0);
    std::vector<bool> visited(num_points_, false);
    std::priority_queue<AnnResult, std::vector<AnnResult>, CompareAnnResult> candidates;

    unsigned int num_hash_tables = qalsh_config_.num_hash_tables;
    unsigned int collision_threshold = qalsh_config_.collision_threshold;
    double bucket_width = qalsh_config_.bucket_width;
    double approximation_ratio = qalsh_config_.approximation_ratio;

    std::vector<double> keys;
    keys.reserve(num_hash_tables);
    std::vector<std::optional<SearchRecord>> lefts;
    lefts.reserve(num_hash_tables);
    std::vector<std::optional<SearchRecord>> rights;
    rights.reserve(num_hash_tables);

    // Initialize the keys, lefts and rights.
    for (unsigned int i = 0; i < num_hash_tables; i++) {
        double table_key = Calc::DotProduct(query_point, dot_vectors_[i]);
        keys.emplace_back(table_key);

        // Locate the leaf node that may contain the key.
        std::shared_ptr<LeafNode> leaf_node = LocateLeafMayContainKey(hash_tables_[i], table_key);
        auto it = std::ranges::lower_bound(leaf_node->keys_, table_key);
        auto index = static_cast<size_t>(std::distance(leaf_node->keys_.begin(), it));

        // Determine the left search location
        if (index == 0) {
            if (leaf_node->prev_leaf_page_num_ != 0) {
                std::shared_ptr<LeafNode> prev_leaf_node =
                    LocateLeafByPageNum(hash_tables_[i], leaf_node->prev_leaf_page_num_);
                lefts.emplace_back(
                    SearchRecord{.leaf_node = prev_leaf_node, .index = prev_leaf_node->num_entries_ - 1});
            } else {
                lefts.emplace_back(std::nullopt);
            }
        } else {
            lefts.emplace_back(SearchRecord{.leaf_node = leaf_node, .index = static_cast<unsigned int>(index - 1)});
        }

        // Determine the right search location
        if (index == leaf_node->keys_.size()) {
            if (leaf_node->next_leaf_page_num_ != 0) {
                std::shared_ptr<LeafNode> next_leaf_node =
                    LocateLeafByPageNum(hash_tables_[i], leaf_node->next_leaf_page_num_);
                rights.emplace_back(SearchRecord{.leaf_node = next_leaf_node, .index = 0});
            } else {
                rights.emplace_back(std::nullopt);
            }
        } else {
            rights.emplace_back(SearchRecord{.leaf_node = leaf_node, .index = static_cast<unsigned int>(index)});
        }
    }

    // c-ANN search
    double radius = 1.0;
    double width = bucket_width * radius / 2.0;  // NOLINT(readability-magic-numbers)

    while (true) {
        unsigned int num_finished = 0;
        std::vector<bool> finish(num_hash_tables, false);
        while (num_finished < num_hash_tables) {
            for (unsigned int i = 0; i < num_hash_tables; i++) {
                if (finish[i]) {
                    continue;
                }
                double table_key = keys[i];

                // Scan the left side of hash table.
                bool left_finished = !lefts[i].has_value();
                for (unsigned int j = 0; j < Global::kScanSize; j++) {
                    if (!lefts[i].has_value()) {
                        left_finished = true;
                        break;
                    }
                    auto& [leaf_node, index] = lefts[i].value();
                    if (!lefts[i].has_value()) {
                        left_finished = true;
                        break;
                    }

                    double dot_product = leaf_node->keys_[index];
                    unsigned int point_id = leaf_node->values_[index];

                    if (table_key - dot_product > width) {
                        left_finished = true;
                        break;
                    }
                    if (!visited[point_id] && ++collision_count[point_id] >= collision_threshold) {
                        visited[point_id] = true;
                        candidates.emplace(AnnResult{
                            .distance = Calc::LpDistance(Utils::ReadPoint(base_file_, num_dimensions_, point_id),
                                                         query_point, norm_order_),
                            .point_id = point_id});
                        if (candidates.size() >= Global::kNumCandidates) {
                            break;
                        }
                    }
                    if (index > 0) {
                        index--;
                    } else {
                        if (leaf_node->prev_leaf_page_num_ == 0) {
                            lefts[i] = std::nullopt;
                            left_finished = true;
                            break;
                        }
                        leaf_node = LocateLeafByPageNum(hash_tables_[i], leaf_node->prev_leaf_page_num_);
                        index = leaf_node->num_entries_ - 1;
                    }
                }
                if (candidates.size() >= Global::kNumCandidates) {
                    break;
                }

                // Scan the right side of hash table.
                bool right_finish = !rights[i].has_value();
                for (unsigned int j = 0; j < Global::kScanSize; j++) {
                    if (!rights[i].has_value()) {
                        right_finish = true;
                        break;
                    }
                    auto& [leaf_node, index] = rights[i].value();
                    if (!rights[i].has_value()) {
                        right_finish = true;
                        break;
                    }

                    double dot_product = leaf_node->keys_[index];
                    unsigned int point_id = leaf_node->values_[index];

                    if (dot_product - table_key > width) {
                        right_finish = true;
                        break;
                    }
                    if (!visited[point_id] && ++collision_count[point_id] >= collision_threshold) {
                        visited[point_id] = true;
                        candidates.emplace(AnnResult{
                            .distance = Calc::LpDistance(Utils::ReadPoint(base_file_, num_dimensions_, point_id),
                                                         query_point, norm_order_),
                            .point_id = point_id});
                        if (candidates.size() >= Global::kNumCandidates) {
                            break;
                        }
                    }
                    if (index < leaf_node->num_entries_ - 1) {
                        index++;
                    } else {
                        if (leaf_node->next_leaf_page_num_ == 0) {
                            rights[i] = std::nullopt;
                            right_finish = true;
                            break;
                        }
                        leaf_node = LocateLeafByPageNum(hash_tables_[i], leaf_node->next_leaf_page_num_);
                        index = 0;
                    }
                }
                if (candidates.size() >= Global::kNumCandidates) {
                    break;
                }

                if (left_finished && right_finish) {
                    finish[i] = true;
                    if (++num_finished == num_hash_tables) {
                        break;
                    }
                }
            }
            if (num_finished == num_hash_tables || candidates.size() >= Global::kNumCandidates) {
                break;
            }
        }
        if (!candidates.empty() && (candidates.top().distance <= approximation_ratio * radius ||
                                    candidates.size() >= Global::kNumCandidates)) {
            break;
        }

        radius *= approximation_ratio;
        width = bucket_width * radius / 2.0;  // NOLINT(readability-magic-numbers)
    }

    return candidates.empty() ? AnnResult{.distance = std::numeric_limits<double>::max(), .point_id = 0}
                              : candidates.top();
}
// NOLINTEND(readability-function-cognitive-complexity)

std::shared_ptr<LeafNode> QalshAnnSearcher::LocateLeafMayContainKey(std::ifstream& ifs, double key) {
    ReadPage(ifs, 0);
    size_t offset = 0;
    auto root_page_num = Utils::ReadFromBuffer<unsigned int>(buffer_, offset);
    auto level = Utils::ReadFromBuffer<unsigned int>(buffer_, offset);

    unsigned int current_level = level;
    unsigned int next_page_num = root_page_num;
    while (current_level != 0) {
        ReadPage(ifs, next_page_num);
        InternalNode internal_node(buffer_);

        auto it = std::ranges::upper_bound(internal_node.keys_, key);
        auto index = static_cast<size_t>(std::distance(internal_node.keys_.begin(), it));
        next_page_num = internal_node.pointers_.at(index);

        current_level--;
    }
    return LocateLeafByPageNum(ifs, next_page_num);
}

std::shared_ptr<LeafNode> QalshAnnSearcher::LocateLeafByPageNum(std::ifstream& ifs, unsigned int page_num) {
    ReadPage(ifs, page_num);
    auto new_node_ptr = std::make_shared<LeafNode>(buffer_);
    return new_node_ptr;
}

void QalshAnnSearcher::ReadPage(std::ifstream& ifs, unsigned int page_num) {
    ifs.seekg(static_cast<std::streamoff>(page_num) * qalsh_config_.page_size, std::ios::beg);
    ifs.read(buffer_.data(), static_cast<std::streamsize>(qalsh_config_.page_size));
}