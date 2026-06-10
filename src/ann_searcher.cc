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
    return std::make_unique<LinearScanAnnSearcher>(*this);
}

void LinearScanAnnSearcher::Init(const PointSetMetadata& metadata, double norm_order) {
    norm_order_ = norm_order;
    num_points_ = metadata.num_points;
    num_dimensions_ = metadata.num_dimensions;
    auto base_points = std::make_shared<std::vector<Point>>(
        PointSet::LoadPointsFromFile(metadata.file_path, num_points_, num_dimensions_));
    if (base_points->size() != num_points_) {
        Log::error("Failed to load all target points for in-memory linear scan.");
    }
    base_points_ = std::move(base_points);
}

AnnResult LinearScanAnnSearcher::Search(const Point& query_point) {
    return SearchWithUpperBound(query_point, std::numeric_limits<double>::infinity());
}

AnnResult LinearScanAnnSearcher::SearchWithUpperBound(const Point& query_point, double initial_upper_bound) {
    if (!base_points_) {
        Log::error("Linear scan searcher is not initialized.");
    }

    AnnResult result{.distance = initial_upper_bound, .point_id = 0};

    for (unsigned int i = 0; i < num_points_; i++) {
        double distance = Calc::LpDistanceBounded((*base_points_)[i], query_point, norm_order_, result.distance);
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
    if (!base_points_) {
        Log::error("Linear scan searcher is not initialized.");
    }

    std::vector<AnnResult> results(query_points.size());
    for (size_t query_index = 0; query_index < query_points.size(); ++query_index) {
        results[query_index] = AnnResult{.distance = initial_upper_bounds[query_index], .point_id = 0};
    }

    for (unsigned int point_id = 0; point_id < num_points_; ++point_id) {
        for (size_t query_index = 0; query_index < query_points.size(); ++query_index) {
            const double distance = Calc::LpDistanceBounded((*base_points_)[point_id], query_points[query_index],
                                                            norm_order_, results[query_index].distance);
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
    auto clone = std::make_unique<QalshAnnSearcher>(approximation_ratio_, random_seed_);
    clone->index_data_ = index_data_;
    clone->ResetSearchState();
    return clone;
}

void QalshAnnSearcher::Init(const PointSetMetadata& metadata, double norm_order) {
    auto index_data = std::make_shared<QalshAnnIndexData>();
    index_data->norm_order = norm_order;
    index_data->num_points = metadata.num_points;
    index_data->num_dimensions = metadata.num_dimensions;

    if (index_data->num_points == 0 || index_data->num_dimensions == 0) {
        Log::error("QALSH requires a non-empty point set with non-zero dimensions.");
    }

    index_data->qalsh_config = QalshConfig{.approximation_ratio = approximation_ratio_};
    index_data->qalsh_config.Regularize(index_data->num_points, norm_order);
    Log::info(
        "Building in-memory QALSH index with approximation ratio {}, bucket width {}, {} hash tables, collision "
        "threshold {}.",
        index_data->qalsh_config.approximation_ratio, index_data->qalsh_config.bucket_width,
        index_data->qalsh_config.num_hash_tables, index_data->qalsh_config.collision_threshold);

    index_data->base_points =
        PointSet::LoadPointsFromFile(metadata.file_path, index_data->num_points, index_data->num_dimensions);
    if (index_data->base_points.size() != index_data->num_points) {
        Log::error("Failed to load all target points for in-memory QALSH index.");
    }

    std::function<double()> coordinate_generator;
    std::mt19937 gen(random_seed_);
    if (std::abs(norm_order - 1.0) < Global::kEpsilon) {
        std::cauchy_distribution<double> dist(0.0, 1.0);
        coordinate_generator = [dist, &gen]() mutable { return dist(gen); };
    } else if (std::abs(norm_order - 2.0) < Global::kEpsilon) {
        std::normal_distribution<double> dist(0.0, 1.0);
        coordinate_generator = [dist, &gen]() mutable { return dist(gen); };
    } else {
        Log::error("Unsupported norm order: {}", norm_order);
    }

    index_data->dot_vectors.clear();
    index_data->dot_vectors.reserve(index_data->qalsh_config.num_hash_tables);
    for (unsigned int table_id = 0; table_id < index_data->qalsh_config.num_hash_tables; ++table_id) {
        Point dot_vector;
        dot_vector.reserve(index_data->num_dimensions);
        for (unsigned int dimension = 0; dimension < index_data->num_dimensions; ++dimension) {
            dot_vector.push_back(coordinate_generator());
        }
        index_data->dot_vectors.push_back(std::move(dot_vector));
    }

    index_data->searchers.clear();
    index_data->searchers.resize(index_data->qalsh_config.num_hash_tables);
    std::vector<std::vector<DotProductPointIdPair>> entries_by_table(index_data->qalsh_config.num_hash_tables);
    for (std::vector<DotProductPointIdPair>& entries : entries_by_table) {
        entries.resize(index_data->num_points);
    }

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long point_index = 0; point_index < static_cast<long long>(index_data->num_points); ++point_index) {
        const auto point_id = static_cast<unsigned int>(point_index);
        const Point& base_point = index_data->base_points[point_id];
        for (unsigned int table_id = 0; table_id < index_data->qalsh_config.num_hash_tables; ++table_id) {
            entries_by_table[table_id][point_id] =
                DotProductPointIdPair{.dot_product = Calc::DotProduct(base_point, index_data->dot_vectors[table_id]),
                                      .point_id = point_id};
        }
    }

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (long long table_index = 0; table_index < static_cast<long long>(index_data->qalsh_config.num_hash_tables);
         ++table_index) {
        const auto table_id = static_cast<unsigned int>(table_index);
        index_data->searchers[table_id].Build(std::move(entries_by_table[table_id]));
    }

    index_data_ = std::move(index_data);
    ResetSearchState();
}

void QalshAnnSearcher::ResetSearchState() {
    if (!index_data_) {
        search_cursors_.clear();
        collision_counts_.clear();
        collision_stamps_.clear();
        visited_stamps_.clear();
        search_epoch_ = 0;
        return;
    }

    search_cursors_.assign(index_data_->qalsh_config.num_hash_tables, InMemoryQalshTableCursor{});
    collision_counts_.assign(index_data_->num_points, 0);
    collision_stamps_.assign(index_data_->num_points, 0);
    visited_stamps_.assign(index_data_->num_points, 0);
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
    if (!index_data_) {
        Log::error("QALSH searcher is not initialized.");
    }

    const QalshAnnIndexData& index_data = *index_data_;
    const unsigned int epoch = NextSearchEpoch();
    bool has_candidate = false;
    AnnResult best_candidate{.distance = std::numeric_limits<double>::infinity(), .point_id = 0};

    unsigned int num_hash_tables = index_data.qalsh_config.num_hash_tables;
    unsigned int collision_threshold = index_data.qalsh_config.collision_threshold;
    double bucket_width = index_data.qalsh_config.bucket_width;
    double approximation_ratio = index_data.qalsh_config.approximation_ratio;

    for (unsigned int i = 0; i < num_hash_tables; i++) {
        search_cursors_[i] =
            index_data.searchers[i].InitCursor(Calc::DotProduct(query_point, index_data.dot_vectors[i]));
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

            bool table_done = index_data.searchers[i].IncrementalSearch(
                search_cursors_[i], bound, [&](unsigned int point_id, [[maybe_unused]] double dot_product) {
                    if (shouldTerminate || visited_stamps_[point_id] == epoch) {
                        return;
                    }

                    if (collision_stamps_[point_id] != epoch) {
                        collision_stamps_[point_id] = epoch;
                        collision_counts_[point_id] = 0;
                    }

                    if (++collision_counts_[point_id] >= collision_threshold) {
                        visited_stamps_[point_id] = epoch;
                        const Point& candidate_point = index_data.base_points[point_id];
                        double distance = Calc::LpDistanceBounded(candidate_point, query_point, index_data.norm_order,
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
