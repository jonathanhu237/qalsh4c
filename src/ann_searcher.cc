#include "ann_searcher.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include "global.h"
#include "point_set.h"
#include "qalsh_hash_table.h"
#include "types.h"
#include "utils.h"

// ---------------------------------------------
// LinearScanAnnSearcher Implementation
// ---------------------------------------------
void LinearScanAnnSearcher::Init(PointSetMetadata point_set_metadata, bool in_memory) {
    base_set_.reset();
    if (in_memory) {
        base_set_ = std::make_unique<InMemoryPointSet>(point_set_metadata);
    } else {
        base_set_ = std::make_unique<DiskPointSet>(point_set_metadata);
    }
}

AnnResult LinearScanAnnSearcher::Search(const Point& query_point) {
    AnnResult result{.point_id = 0, .distance = std::numeric_limits<double>::max()};

    unsigned int base_num_points = base_set_->get_num_points();

    for (unsigned int i = 0; i < base_num_points; i++) {
        Point base_point = base_set_->GetPoint(i);
        double distance = Utils::L1Distance(base_point, query_point);
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
QalshAnnSearcher::QalshAnnSearcher(double approximation_ratio) {
    qalsh_config_.approximation_ratio = approximation_ratio;
}

void QalshAnnSearcher::Init(PointSetMetadata point_set_metadata, bool in_memory) {
    base_set_.reset();
    dot_vectors_.clear();
    hash_tables.clear();

    if (in_memory) {
        // Initialize base set.
        base_set_ = std::make_unique<InMemoryPointSet>(point_set_metadata);

        // Regularize QALSH config.
        Utils::RegularizeQalshConfig(qalsh_config_, point_set_metadata.num_points);

        // Generate dot vectors.
        dot_vectors_.resize(qalsh_config_.num_hash_tables);
        std::mt19937 gen(std::random_device{}());
        std::cauchy_distribution<double> standard_cauchy_dist(0.0, 1.0);

        for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
            dot_vectors_[i].resize(base_set_->get_num_dimensions());
            std::ranges::generate(dot_vectors_[i], [&]() { return standard_cauchy_dist(gen); });
        }

        // Initialize QALSH hash tables.
        hash_tables.reserve(qalsh_config_.num_hash_tables);
        for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
            std::vector<DotProductPointIdPair> data(point_set_metadata.num_points);
            for (unsigned int j = 0; j < point_set_metadata.num_points; j++) {
                data[j] = {.dot_product = Utils::DotProduct(base_set_->GetPoint(j), dot_vectors_[i]), .point_id = j};
            }
            std::ranges::sort(data, {}, &DotProductPointIdPair::dot_product);
            hash_tables.emplace_back(std::make_unique<InMemoryQalshHashTable>(data));
        }
    } else {
        // Initialize base set.
        base_set_ = std::make_unique<DiskPointSet>(point_set_metadata);

        // Load QALSH config.
        std::filesystem::path parent_directory = point_set_metadata.file_path.parent_path();
        std::string stem = point_set_metadata.file_path.stem();
        std::filesystem::path index_directory = parent_directory / "index" / stem;
        qalsh_config_ = Utils::LoadQalshConfig(index_directory / "config.json");

        // Load dot vectors.
        std::ifstream ifs(index_directory / "dot_vectors.bin");
        dot_vectors_.resize(qalsh_config_.num_hash_tables);
        for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
            dot_vectors_[i].resize(base_set_->get_num_dimensions());
            ifs.read(reinterpret_cast<char*>(dot_vectors_[i].data()),
                     static_cast<std::streamoff>(sizeof(Coordinate) * dot_vectors_[i].size()));
        }

        // Initialize QALSH hash tables.
        hash_tables.reserve(qalsh_config_.num_hash_tables);
        for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
            hash_tables.emplace_back(std::make_unique<DiskQalshHashTable>(
                index_directory / "b_plus_trees" / std::format("{}.bin", i), qalsh_config_.page_size));
        }
    }
}

AnnResult QalshAnnSearcher::Search(const Point& query_point) {
    std::priority_queue<AnnResult, std::vector<AnnResult>, CompareAnnResult> candidates;
    std::vector<bool> visited(base_set_->get_num_points(), false);
    std::vector<unsigned int> collision_count(base_set_->get_num_points(), 0);
    std::vector<double> keys(qalsh_config_.num_hash_tables);
    double search_radius = 1.0;

    // Initialize the keys
    for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
        keys[i] = Utils::DotProduct(query_point, dot_vectors_[i]);
        hash_tables[i]->Init(keys[i]);
    }

    // c-ANN search
    while (!shouldTerminate(candidates, search_radius)) {
        // (R,c)-NN search
        double bound = qalsh_config_.bucket_width * search_radius / 2.0;  // NOLINT: readability-magic-numbers
        for (unsigned int j = 0; !shouldTerminate(candidates, search_radius) && j < qalsh_config_.num_hash_tables;
             j++) {
            while (!shouldTerminate(candidates, search_radius)) {
                // Find the point in the left direction first.
                std::optional<unsigned int> point_id = hash_tables[j]->LeftFindNext(bound);
                if (!point_id.has_value()) {
                    // If there doesn't exist any point in the left direction, find
                    // the point in the right direction.
                    point_id = hash_tables[j]->RightFindNext(bound);
                }

                // If there doesn't exist any point in both direction, we should break.
                if (!point_id.has_value()) {
                    break;
                }

                // Check whether the point is checked.
                if (visited[point_id.value()]) {
                    continue;
                }

                if (++collision_count[point_id.value()] >= qalsh_config_.collision_threshold) {
                    Point point = base_set_->GetPoint(point_id.value());
                    double distance = Utils::L1Distance(point, query_point);
                    candidates.emplace(AnnResult{.point_id = point_id.value(), .distance = distance});
                    visited[point_id.value()] = true;
                }
            }
        }

        // Update search radius.
        search_radius *= qalsh_config_.approximation_ratio;
    }

    if (!candidates.empty()) {
        return candidates.top();
    }

    // Defense programming
    return {.point_id = static_cast<unsigned int>(std::numeric_limits<double>::max()),
            .distance = std::numeric_limits<unsigned int>::max()};
}

bool QalshAnnSearcher::shouldTerminate(
    const std::priority_queue<AnnResult, std::vector<AnnResult>, CompareAnnResult>& candidates,
    double search_radius) const {
    return (!candidates.empty() && candidates.top().distance <= qalsh_config_.approximation_ratio * search_radius) ||
           candidates.size() >= Global::kNumCandidates;
}