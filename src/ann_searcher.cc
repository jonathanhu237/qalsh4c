#include "ann_searcher.h"

#include <cmath>
#include <format>
#include <limits>
#include <queue>
#include <vector>

#include "b_plus_tree.h"
#include "global.h"
#include "log.h"
#include "point_set.h"
#include "utils.h"

// ---------------------------------------------
// LinearScanAnnSearcher Implementation
// ---------------------------------------------
std::unique_ptr<AnnSearcher> LinearScanAnnSearcher::Clone() const {
    return std::make_unique<LinearScanAnnSearcher>();
}

void LinearScanAnnSearcher::Init(const PointSetMetadata& metadata, double norm_order) {
    base_.Init(metadata);
    norm_order_ = norm_order;
}

AnnResult LinearScanAnnSearcher::Search(const Point& query_point) {
    AnnResult result{.distance = std::numeric_limits<double>::max(), .point_id = 0};

    for (unsigned int i = 0; i < base_.get_num_points(); i++) {
        double distance = Calc::LpDistance(base_.ReadPoint(i), query_point, norm_order_);
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
QalshAnnSearcher::QalshAnnSearcher(bool candidate_limit) : candidate_limit_(candidate_limit) {}

std::unique_ptr<AnnSearcher> QalshAnnSearcher::Clone() const {
    return std::make_unique<QalshAnnSearcher>(candidate_limit_);
}

void QalshAnnSearcher::Init(const PointSetMetadata& metadata, double norm_order) {
    base_.Init(metadata);
    norm_order_ = norm_order;

    // Load QALSH configuration.
    std::filesystem::path index_directory =
        metadata.file_path.parent_path() / "index" / std::format("l{}", norm_order_) / metadata.file_path.stem();
    qalsh_config_.Load(index_directory / "config.json");

    // Load the dot vectors.
    std::ifstream dot_vector_file(index_directory / "dot_vectors.bin", std::ios::binary);
    if (!dot_vector_file.is_open()) {
        Log::error("Failed to open dot vectors file: {}", (index_directory / "dot_vectors.bin").string());
        return;
    }
    dot_vectors_ = PointSet::LoadPointsFromFile(index_directory / "dot_vectors.bin", qalsh_config_.num_hash_tables,
                                                base_.get_num_dimensions());

    // Create B+ tree searchers.
    searchers_.clear();
    searchers_.reserve(qalsh_config_.num_hash_tables);
    for (unsigned int i = 0; i < qalsh_config_.num_hash_tables; i++) {
        searchers_.emplace_back(index_directory / "b_plus_trees" / std::format("{}.bin", i), qalsh_config_.page_size);
    }
}

AnnResult QalshAnnSearcher::Search(const Point& query_point) {
    std::vector<unsigned int> collision_count(base_.get_num_points(), 0);
    std::vector<bool> visited(base_.get_num_points(), false);
    std::priority_queue<AnnResult, std::vector<AnnResult>, CompareAnnResult> candidates;

    unsigned int num_hash_tables = qalsh_config_.num_hash_tables;
    unsigned int collision_threshold = qalsh_config_.collision_threshold;
    double bucket_width = qalsh_config_.bucket_width;
    double approximation_ratio = qalsh_config_.approximation_ratio;

    for (unsigned int i = 0; i < num_hash_tables; i++) {
        searchers_[i].Init(Calc::DotProduct(query_point, dot_vectors_[i]));
    }

    // c-ANN search
    double radius = 1.0;
    double bound = bucket_width * radius / 2.0;
    bool shouldTerminate = false;

    while (!shouldTerminate) {
        std::queue<unsigned int> active_tables;
        for (unsigned int i = 0; i < num_hash_tables; i++) {
            active_tables.push(i);
        }

        while (!active_tables.empty() && !shouldTerminate) {
            unsigned int i = active_tables.front();
            active_tables.pop();

            bool table_done =
                searchers_[i].IncrementalSearch(bound, [&](unsigned int point_id, [[maybe_unused]] double dot_product) {
                    if (shouldTerminate || visited[point_id]) {
                        return;
                    }

                    if (++collision_count[point_id] >= collision_threshold) {
                        visited[point_id] = true;
                        candidates.emplace(
                            AnnResult{.distance = Calc::LpDistance(base_.ReadPoint(point_id), query_point, norm_order_),
                                      .point_id = point_id});

                        // Termination condition 1
                        if (!candidates.empty() && (candidates.top().distance <= approximation_ratio * radius)) {
                            shouldTerminate = true;
                            return;
                        }

                        // Termination condition 2
                        if (candidate_limit_ && candidates.size() >= Global::kNumCandidates) {
                            shouldTerminate = true;
                            return;
                        }
                    }
                });

            if (!table_done) {
                active_tables.push(i);
            }
        }

        // Update radius and bound for next iteration
        radius *= approximation_ratio;
        bound = bucket_width * radius / 2.0;
    }

    return candidates.top();
}
