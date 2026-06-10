#include "ann_searcher.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <vector>

#include "global.h"
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

std::vector<AnnResult> LinearScanAnnSearcher::SearchBatchWithUpperBounds(
    const std::vector<Point>& query_points, const std::vector<double>& initial_upper_bounds) {
    if (query_points.size() != initial_upper_bounds.size()) {
        Log::error("Query size and upper-bound size mismatch.");
    }

    std::vector<AnnResult> results(query_points.size());
    for (size_t query_index = 0; query_index < query_points.size(); ++query_index) {
        results[query_index] = AnnResult{.distance = initial_upper_bounds[query_index], .point_id = 0};
    }

    base_.SeekPoint(0);
    for (unsigned int point_id = 0; point_id < base_.get_num_points(); ++point_id) {
        if (!base_.ReadNextPointInto(scratch_point_)) {
            Log::error("Failed to read point {} during batched linear scan.", point_id);
        }

        for (size_t query_index = 0; query_index < query_points.size(); ++query_index) {
            const double distance = Calc::LpDistanceBounded(scratch_point_, query_points[query_index], norm_order_,
                                                            results[query_index].distance);
            if (distance < results[query_index].distance) {
                results[query_index].distance = distance;
                results[query_index].point_id = point_id;
            }
        }
    }

    return results;
}

// ---------------------------------------------
// QalshAnnSearcher Implementation
// ---------------------------------------------
QalshAnnSearcher::QalshAnnSearcher(double approximation_ratio)
    : QalshAnnSearcher(approximation_ratio, Random::SeededGenerator()()) {}

QalshAnnSearcher::QalshAnnSearcher(double approximation_ratio, std::mt19937::result_type random_seed)
    : approximation_ratio_(approximation_ratio),
      random_seed_(random_seed) {}

bool QalshAnnSearcher::RequiresIndex() const {
    return false;
}

std::unique_ptr<AnnSearcher> QalshAnnSearcher::Clone() const {
    return std::make_unique<QalshAnnSearcher>(approximation_ratio_, random_seed_);
}

void QalshAnnSearcher::Init(const PointSetMetadata& metadata, double norm_order) {
    norm_order_ = norm_order;
    num_points_ = metadata.num_points;
    num_dimensions_ = metadata.num_dimensions;

    if (num_points_ == 0 || num_dimensions_ == 0) {
        Log::error("QALSH requires a non-empty point set with non-zero dimensions.");
    }

    qalsh_config_ = QalshConfig{.approximation_ratio = approximation_ratio_};
    qalsh_config_.Regularize(num_points_, norm_order_);
    Log::info(
        "Building in-memory QALSH index with approximation ratio {}, bucket width {}, {} hash tables, collision "
        "threshold {}.",
        qalsh_config_.approximation_ratio, qalsh_config_.bucket_width, qalsh_config_.num_hash_tables,
        qalsh_config_.collision_threshold);

    base_points_ = PointSet::LoadPointsFromFile(metadata.file_path, num_points_, num_dimensions_);
    if (base_points_.size() != num_points_) {
        Log::error("Failed to load all target points for in-memory QALSH index.");
    }

    std::function<double()> coordinate_generator;
    std::mt19937 gen(random_seed_);
    if (std::abs(norm_order_ - 1.0) < Global::kEpsilon) {
        std::cauchy_distribution<double> dist(0.0, 1.0);
        coordinate_generator = [dist, &gen]() mutable { return dist(gen); };
    } else if (std::abs(norm_order_ - 2.0) < Global::kEpsilon) {
        std::normal_distribution<double> dist(0.0, 1.0);
        coordinate_generator = [dist, &gen]() mutable { return dist(gen); };
    } else {
        Log::error("Unsupported norm order: {}", norm_order_);
    }

    dot_vectors_.clear();
    dot_vectors_.reserve(qalsh_config_.num_hash_tables);
    for (unsigned int table_id = 0; table_id < qalsh_config_.num_hash_tables; ++table_id) {
        Point dot_vector;
        dot_vector.reserve(num_dimensions_);
        for (unsigned int dimension = 0; dimension < num_dimensions_; ++dimension) {
            dot_vector.push_back(coordinate_generator());
        }
        dot_vectors_.push_back(std::move(dot_vector));
    }

    searchers_.clear();
    searchers_.reserve(qalsh_config_.num_hash_tables);
    for (unsigned int table_id = 0; table_id < qalsh_config_.num_hash_tables; ++table_id) {
        std::vector<DotProductPointIdPair> entries;
        entries.reserve(num_points_);
        for (unsigned int point_id = 0; point_id < num_points_; ++point_id) {
            entries.push_back(DotProductPointIdPair{.dot_product = Calc::DotProduct(base_points_[point_id],
                                                                                    dot_vectors_[table_id]),
                                                    .point_id = point_id});
        }
        searchers_.emplace_back(std::move(entries));
    }

    collision_counts_.assign(num_points_, 0);
    collision_stamps_.assign(num_points_, 0);
    visited_stamps_.assign(num_points_, 0);
    search_epoch_ = 0;
}

unsigned int QalshAnnSearcher::NextSearchEpoch() {
    ++search_epoch_;
    if (search_epoch_ == 0) {
        std::ranges::fill(collision_stamps_, 0);
        std::ranges::fill(visited_stamps_, 0);
        search_epoch_ = 1;
    }
    return search_epoch_;
}

AnnResult QalshAnnSearcher::Search(const Point& query_point) {
    const unsigned int epoch = NextSearchEpoch();
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
                    if (shouldTerminate || visited_stamps_[point_id] == epoch) {
                        return;
                    }

                    if (collision_stamps_[point_id] != epoch) {
                        collision_stamps_[point_id] = epoch;
                        collision_counts_[point_id] = 0;
                    }

                    if (++collision_counts_[point_id] >= collision_threshold) {
                        visited_stamps_[point_id] = epoch;
                        const Point& candidate_point = base_points_[point_id];
                        double distance = Calc::LpDistanceBounded(candidate_point, query_point, norm_order_,
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
