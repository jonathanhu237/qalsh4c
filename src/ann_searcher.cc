#include "ann_searcher.h"

#include <cmath>
#include <format>
#include <limits>
#include <queue>
#include <vector>

#include "b_plus_tree.h"
#include "log.h"
#include "point_set.h"
#include "utils.h"

// ---------------------------------------------
// LinearScanAnnSearcher Implementation
// ---------------------------------------------
bool LinearScanAnnSearcher::RequiresIndex() const {
    return false;
}

std::unique_ptr<AnnSearcher> LinearScanAnnSearcher::Clone() const {
    return std::make_unique<LinearScanAnnSearcher>();
}

void LinearScanAnnSearcher::Init(const PointSetMetadata& metadata, double norm_order) {
    base_.Init(metadata);
    norm_order_ = norm_order;
}

AnnResult LinearScanAnnSearcher::Search(const Point& query_point) {
    return SearchWithUpperBound(query_point, std::numeric_limits<double>::infinity());
}

AnnResult LinearScanAnnSearcher::SearchWithUpperBound(const Point& query_point, double initial_upper_bound) {
    AnnResult result{.distance = initial_upper_bound, .point_id = 0};

    base_.SeekPoint(0);
    for (unsigned int i = 0; i < base_.get_num_points(); i++) {
        if (!base_.ReadNextPointInto(scratch_point_)) {
            Log::error("Failed to read point {} during linear scan.", i);
        }

        double distance = Calc::LpDistanceBounded(scratch_point_, query_point, norm_order_, result.distance);
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
bool QalshAnnSearcher::RequiresIndex() const {
    return true;
}

std::unique_ptr<AnnSearcher> QalshAnnSearcher::Clone() const {
    return std::make_unique<QalshAnnSearcher>();
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
    bool has_candidate = false;
    AnnResult best_candidate{.distance = std::numeric_limits<double>::infinity(), .point_id = 0};

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

    auto t1_satisfied = [&]() {
        return has_candidate && best_candidate.distance <= approximation_ratio * radius;
    };

    while (!shouldTerminate) {
        if (t1_satisfied()) {
            break;
        }

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
                        base_.ReadPointInto(point_id, candidate_scratch_point_);
                        double distance = Calc::LpDistanceBounded(candidate_scratch_point_, query_point, norm_order_,
                                                                  best_candidate.distance);
                        if (distance < best_candidate.distance) {
                            has_candidate = true;
                            best_candidate = AnnResult{.distance = distance, .point_id = point_id};
                        }

                        // Termination condition 1
                        if (t1_satisfied()) {
                            shouldTerminate = true;
                            return;
                        }
                    }
                });

            if (!table_done) {
                active_tables.push(i);
            }
        }

        if (t1_satisfied()) {
            break;
        }

        // Update radius and bound for next iteration
        radius *= approximation_ratio;
        bound = bucket_width * radius / 2.0;
    }

    return best_candidate;
}
