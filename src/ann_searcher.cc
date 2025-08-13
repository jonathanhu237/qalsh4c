#include "ann_searcher.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <vector>

#include "b_plus_tree.h"
#include "global.h"
#include "types.h"
#include "utils.h"

// ---------------------------------------------
// InMemoryLinearScanAnnSearcher Definition
// ---------------------------------------------
void InMemoryLinearScanAnnSearcher::Init(const PointSetMetadata& base_metadata) {
    base_points_ =
        Utils::LoadPointsFromFile(base_metadata.file_path, base_metadata.num_points, base_metadata.num_dimensions);
}

AnnResult InMemoryLinearScanAnnSearcher::Search(const Point& query_point) {
    AnnResult result{.distance = std::numeric_limits<double>::max(), .point_id = 0};

    for (unsigned int i = 0; i < base_points_.size(); i++) {
        double distance = Utils::L1Distance(base_points_[i], query_point);
        if (distance < result.distance) {
            result.point_id = i;
            result.distance = distance;
        }
    }

    return result;
}

// ---------------------------------------------
// DiskLinearScanAnnSearcher Definition
// ---------------------------------------------
void DiskLinearScanAnnSearcher::Init(const PointSetMetadata& base_metadata) {
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
}

AnnResult DiskLinearScanAnnSearcher::Search(const Point& query_point) {
    AnnResult result{.distance = std::numeric_limits<double>::max(), .point_id = 0};

    for (unsigned int i = 0; i < num_points_; i++) {
        double distance = Utils::L1Distance(Utils::ReadPoint(base_file_, num_dimensions_, i), query_point);
        if (distance < result.distance) {
            result.point_id = i;
            result.distance = distance;
        }
    }

    return result;
}

// ---------------------------------------------
// InMemoryQalshAnnSearcher Implementation
// ---------------------------------------------
InMemoryQalshAnnSearcher::InMemoryQalshAnnSearcher(double approximation_ratio) : gen_(Utils::CreateSeededGenerator()) {
    qalsh_config_.approximation_ratio = approximation_ratio;
}

void InMemoryQalshAnnSearcher::Init(const PointSetMetadata& base_metadata) {
    // Load the base points from the file.
    base_points_ =
        Utils::LoadPointsFromFile(base_metadata.file_path, base_metadata.num_points, base_metadata.num_dimensions);

    // Regularize the QalshConfig parameters based on the number of points.
    Utils::RegularizeQalshConfig(qalsh_config_, base_metadata.num_points);

    // Print the QalshConfig parameters.
    spdlog::info(
        "QALSH Configuration:\n"
        "\tApproximation Ratio: {}\n"
        "\tBucket Width: {}\n"
        "\tError Probability: {}\n"
        "\tNumber of Hash Tables: {}\n"
        "\tCollision Threshold: {}",
        qalsh_config_.approximation_ratio, qalsh_config_.bucket_width, qalsh_config_.error_probability,
        qalsh_config_.num_hash_tables, qalsh_config_.collision_threshold);

    // Generate dot vectors.
    dot_vectors_.clear();
    dot_vectors_.resize(qalsh_config_.num_hash_tables);
    std::cauchy_distribution<double> standard_cauchy_dist(0.0, 1.0);
    auto num_dimensions = static_cast<unsigned int>(base_points_[0].size());

    for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
        dot_vectors_[i].reserve(num_dimensions);
        std::ranges::generate_n(std::back_inserter(dot_vectors_[i]), num_dimensions,
                                [&]() { return standard_cauchy_dist(gen_); });
    }

    // Initialize QALSH hash tables.
    hash_tables_.clear();
    hash_tables_.resize(qalsh_config_.num_hash_tables);
    for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
        hash_tables_[i].reserve(base_metadata.num_points);
        for (unsigned int j = 0; j < base_metadata.num_points; j++) {
            hash_tables_[i].emplace_back(DotProductPointIdPair{
                .dot_product = Utils::DotProduct(base_points_[j], dot_vectors_[i]), .point_id = j});
        }
        std::ranges::sort(hash_tables_[i], {}, &DotProductPointIdPair::dot_product);
    }
}

// NOLINTBEGIN(readability-function-cognitive-complexity)
AnnResult InMemoryQalshAnnSearcher::Search(const Point& query_point) {
    std::vector<unsigned int> collision_count(base_points_.size(), 0);
    std::vector<bool> visited(base_points_.size(), false);
    std::priority_queue<AnnResult, std::vector<AnnResult>, CompareAnnResult> candidates;

    unsigned int num_hash_tables = qalsh_config_.num_hash_tables;
    unsigned int collision_threshold = qalsh_config_.collision_threshold;
    double bucket_width = qalsh_config_.bucket_width;
    double approximation_ratio = qalsh_config_.approximation_ratio;

    std::vector<double> keys;
    keys.reserve(num_hash_tables);
    std::vector<std::optional<unsigned int>> lefts;
    lefts.reserve(num_hash_tables);
    std::vector<std::optional<unsigned int>> rights;
    rights.reserve(num_hash_tables);

    // Initialize the keys, lefts and rights.
    for (unsigned int i = 0; i < num_hash_tables; i++) {
        double table_key = Utils::DotProduct(query_point, dot_vectors_[i]);
        keys.emplace_back(table_key);
        auto it = std::ranges::lower_bound(hash_tables_[i], table_key, {}, &DotProductPointIdPair::dot_product);
        auto index = static_cast<size_t>(std::distance(hash_tables_[i].begin(), it));

        lefts.emplace_back(index == 0 ? std::nullopt : std::make_optional(index - 1));
        rights.emplace_back(index == hash_tables_[i].size() ? std::nullopt : std::make_optional(index));
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
                    auto& [dot_product, point_id] = hash_tables_[i][lefts[i].value()];
                    if (table_key - dot_product > width) {
                        left_finished = true;
                        break;
                    }
                    if (!visited[point_id] && ++collision_count[point_id] >= collision_threshold) {
                        visited[point_id] = true;
                        candidates.emplace(AnnResult{.distance = Utils::L1Distance(base_points_[point_id], query_point),
                                                     .point_id = point_id});
                        if (candidates.size() >= Global::kNumCandidates) {
                            break;
                        }
                    }
                    if (lefts[i].value() > 0) {
                        lefts[i].value()--;
                    } else {
                        lefts[i] = std::nullopt;
                        left_finished = true;
                        break;
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
                    auto& [dot_product, point_id] = hash_tables_[i][rights[i].value()];
                    if (dot_product - table_key > width) {
                        right_finish = true;
                        break;
                    }
                    if (!visited[point_id] && ++collision_count[point_id] >= collision_threshold) {
                        visited[point_id] = true;
                        candidates.emplace(AnnResult{.distance = Utils::L1Distance(base_points_[point_id], query_point),
                                                     .point_id = point_id});
                        if (candidates.size() >= Global::kNumCandidates) {
                            break;
                        }
                    }
                    if (rights[i].value() < hash_tables_[i].size() - 1) {
                        rights[i].value()++;
                    } else {
                        rights[i] = std::nullopt;
                        right_finish = true;
                        break;
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

// ---------------------------------------------
// DiskQalshAnnSearcher Implementation
// ---------------------------------------------
void DiskQalshAnnSearcher::Init(const PointSetMetadata& base_metadata) {
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

    // Load QALSH configuration.
    std::string stem = base_metadata.file_path.stem();
    std::filesystem::path index_directory = base_metadata.file_path.parent_path() / "index" / stem;
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

    // Clean the leaf nodes cache.
    leaf_nodes_cache_.clear();
}

// NOLINTBEGIN(readability-function-cognitive-complexity)
AnnResult DiskQalshAnnSearcher::Search(const Point& query_point) {
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
        double table_key = Utils::DotProduct(query_point, dot_vectors_[i]);
        keys.emplace_back(table_key);

        // Locate the leaf node that may contain the key.
        std::shared_ptr<LeafNode> leaf_node = LocateLeafMayContainKey(hash_tables_[i], i, table_key);
        auto it = std::ranges::lower_bound(leaf_node->keys_, table_key);
        auto index = static_cast<size_t>(std::distance(leaf_node->keys_.begin(), it));

        // Determine the left search location
        if (index == 0) {
            if (leaf_node->prev_leaf_page_num_ != 0) {
                std::shared_ptr<LeafNode> prev_leaf_node =
                    LocateLeafByPageNum(hash_tables_[i], i, leaf_node->prev_leaf_page_num_);
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
                    LocateLeafByPageNum(hash_tables_[i], i, leaf_node->next_leaf_page_num_);
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
                            .distance =
                                Utils::L1Distance(Utils::ReadPoint(base_file_, num_dimensions_, point_id), query_point),
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
                        leaf_node = LocateLeafByPageNum(hash_tables_[i], i, leaf_node->prev_leaf_page_num_);
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
                            .distance =
                                Utils::L1Distance(Utils::ReadPoint(base_file_, num_dimensions_, point_id), query_point),
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
                        leaf_node = LocateLeafByPageNum(hash_tables_[i], i, leaf_node->next_leaf_page_num_);
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

std::shared_ptr<LeafNode> DiskQalshAnnSearcher::LocateLeafMayContainKey(std::ifstream& ifs, unsigned int table_idx,
                                                                        double key) {
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
    return LocateLeafByPageNum(ifs, table_idx, next_page_num);
}

std::shared_ptr<LeafNode> DiskQalshAnnSearcher::LocateLeafByPageNum(std::ifstream& ifs, unsigned int table_idx,
                                                                    unsigned int page_num) {
    uint64_t composite_key = (static_cast<uint64_t>(table_idx) << 32) | page_num;  // NOLINT(readability-magic-numbers)
    auto it = leaf_nodes_cache_.find(composite_key);
    if (it != leaf_nodes_cache_.end()) {
        return it->second;
    }

    ReadPage(ifs, page_num);
    auto new_node_ptr = std::make_shared<LeafNode>(buffer_);
    leaf_nodes_cache_[composite_key] = new_node_ptr;
    return new_node_ptr;
}

void DiskQalshAnnSearcher::ReadPage(std::ifstream& ifs, unsigned int page_num) {
    ifs.seekg(static_cast<std::streamoff>(page_num) * qalsh_config_.page_size, std::ios::beg);
    ifs.read(buffer_.data(), static_cast<std::streamsize>(qalsh_config_.page_size));
}