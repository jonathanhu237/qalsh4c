#include "ann_searcher.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <ios>
#include <queue>
#include <variant>

#include "global.h"
#include "point_set.h"
#include "timer.h"
#include "types.h"
#include "utils.h"

// ---------------------------------------------
// LinearScanAnnSearcher Implementation
// ---------------------------------------------

LinearScanAnnSearcher::LinearScanAnnSearcher(PointSetReader* base_reader) : base_reader_(base_reader) {}

AnnResult LinearScanAnnSearcher::Search(const PointVariant& query_point) {
    AnnResult result{.point_id = 0, .distance = std::numeric_limits<double>::max()};

    {
        Timer timer(&Global::linear_scan_search_time_ms);

        unsigned int base_num_points = base_reader_->get_num_points();
        for (unsigned int i = 0; i < base_num_points; i++) {
            PointVariant base_point = base_reader_->GetPoint(i);
            double distance = 0.0;

            std::visit(
                [&](const auto& concrete_base_point, const auto& concrete_query_point) {
                    distance = Utils::CalculateL1Distance(concrete_base_point, concrete_query_point);
                },
                base_point, query_point);

            if (distance < result.distance) {
                result.point_id = i;
                result.distance = distance;
            }
        }
    }

    return result;
}

// ---------------------------------------------
// QalshAnnSearcher Implementation
// ---------------------------------------------

QalshAnnSearcher::QalshAnnSearcher(PointSetReader* base_reader, std::filesystem::path index_directory)
    : base_reader_(base_reader), index_directory_(std::move(index_directory)) {
    std::filesystem::path config_path = index_directory_ / "config.toml";
    spdlog::info("Loading Qalsh configuration from {}", config_path.string());
    qalsh_config_.Load(config_path);

    std::filesystem::path dot_vectors_path = index_directory_ / "dot_vectors.bin";
    spdlog::info("Loading dot vectors from {}", dot_vectors_path.string());
    std::ifstream ifs(dot_vectors_path, std::ios::binary);
    if (!ifs) {
        spdlog::error("Failed to open dot_vectors.bin for reading");
        return;
    }

    dot_vectors_.resize(qalsh_config_.num_hash_tables);
    unsigned int num_dimensions = base_reader_->get_num_dimensions();
    for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
        dot_vectors_[i].resize(num_dimensions);
        ifs.read(reinterpret_cast<char*>(dot_vectors_[i].data()),
                 static_cast<std::streamoff>(num_dimensions * sizeof(double)));
    }

    // Initialize B+ trees
    b_plus_tree_searchers_.reserve(qalsh_config_.num_hash_tables);
    for (unsigned int j = 0; j < qalsh_config_.num_hash_tables; j++) {
        const std::filesystem::path index_file_path = index_directory_ / std::format("base_idx_{}.bin", j);
        b_plus_tree_searchers_.emplace_back(index_file_path, qalsh_config_.page_size);
    }
}

AnnResult QalshAnnSearcher::Search(const PointVariant& query_point) {
    std::priority_queue<AnnResult, std::vector<AnnResult>> candidates;
    std::vector<char> visited(base_reader_->get_num_points(), 0);
    std::unordered_map<unsigned int, unsigned int> collision_count;
    std::vector<double> keys(qalsh_config_.num_hash_tables);
    double search_radius = 1.0;

    // Initialize the keys
    for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
        keys[i] = 0.0;

        std::visit(
            [&](const auto& concrete_query_point) {
                keys[i] = Utils::DotProduct(concrete_query_point, dot_vectors_[i]);
            },
            query_point);

        b_plus_tree_searchers_[i].Init(keys[i]);
    }

    // c-ANN search
    while (candidates.size() < static_cast<size_t>(std::ceil(qalsh_config_.beta * base_reader_->get_num_points()))) {
        for (unsigned int j = 0; j < qalsh_config_.num_hash_tables; j++) {
            std::vector<unsigned int> point_ids =
                b_plus_tree_searchers_[j].IncrementalSearch(qalsh_config_.bucket_width * search_radius / 2.0);

            for (auto point_id : point_ids) {
                if (visited[point_id] != 0) {
                    continue;
                }

                collision_count[point_id]++;
                if (collision_count.at(point_id) >= qalsh_config_.collision_threshold) {
                    PointVariant point = base_reader_->GetPoint(point_id);

                    double distance = 0.0;
                    std::visit(
                        [&](const auto& concrete_point, const auto& concrete_query_point) {
                            distance = Utils::CalculateL1Distance(concrete_point, concrete_query_point);
                        },
                        point, query_point);
                    candidates.emplace(distance, point_id);
                    visited[point_id] = 1;  // Changed from true
                }
            }
        }

        if (!candidates.empty() && candidates.top().distance <= qalsh_config_.approximation_ratio * search_radius) {
            break;
        }

        search_radius *= qalsh_config_.approximation_ratio;
    }

    if (!candidates.empty()) {
        return candidates.top();
    }

    // Defense programming
    return {.point_id = static_cast<unsigned int>(std::numeric_limits<double>::max()),
            .distance = std::numeric_limits<unsigned int>::max()};
}

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
