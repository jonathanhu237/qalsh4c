#include "ann_searcher.h"

#include <filesystem>
#include <fstream>
#include <ios>
#include <queue>
#include <variant>

#include "point_set.h"
#include "types.h"
#include "utils.h"

// ---------------------------------------------
// LinearScanAnnSearcher Implementation
// ---------------------------------------------

LinearScanAnnSearcher::LinearScanAnnSearcher(PointSetReader* base_reader) : base_reader_(base_reader) {}

AnnResult LinearScanAnnSearcher::Search(const PointVariant& query_point) {
    AnnResult result{.point_id = 0, .distance = std::numeric_limits<double>::max()};

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

    return result;
}

// ---------------------------------------------
// QalshAnnSearcher Implementation
// ---------------------------------------------

QalshAnnSearcher::QalshAnnSearcher(PointSetReader* base_reader, std::filesystem::path index_directory)
    : base_reader_(base_reader), index_directory_(std::move(index_directory)) {
    std::filesystem::path config_path = index_directory_ / "config.toml";
    spdlog::debug("Loading Qalsh configuration from {}", config_path.string());
    qalsh_config_.Load(config_path);

    std::filesystem::path dot_vectors_path = index_directory_ / "dot_vectors.bin";
    spdlog::debug("Loading dot vectors from {}", dot_vectors_path.string());
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
