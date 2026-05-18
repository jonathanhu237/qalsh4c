#include "ann_searcher.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <ranges>
#include <utility>
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
namespace {

bool SameMetadata(const PointSetMetadata& lhs, const PointSetMetadata& rhs) {
    return lhs.file_path == rhs.file_path && lhs.num_points == rhs.num_points &&
           lhs.num_dimensions == rhs.num_dimensions;
}

std::function<double()> MakeProjectionGenerator(double norm_order, std::mt19937& gen) {
    if (std::abs(norm_order - 1.0) < Global::kEpsilon) {
        std::cauchy_distribution<double> dist(0.0, 1.0);
        return [dist, &gen]() mutable { return dist(gen); };
    }

    if (std::abs(norm_order - 2.0) < Global::kEpsilon) {
        std::normal_distribution<double> dist(0.0, 1.0);
        return [dist, &gen]() mutable { return dist(gen); };
    }

    Log::error("Unsupported norm order: {}", norm_order);
}

struct TableCursor {
    const std::vector<ProjectionPointIdPair>* entries{nullptr};
    double query_key{0.0};
    size_t left_index{0};
    size_t right_index{0};
    bool has_left{false};
    bool has_right{false};
};

}  // namespace

QalshAnnSearcher::QalshAnnSearcher(bool candidate_limit, double approximation_ratio)
    : candidate_limit_(candidate_limit), approximation_ratio_(approximation_ratio) {}

QalshAnnSearcher::QalshAnnSearcher(std::shared_ptr<const InMemoryQalshIndex> index, bool candidate_limit,
                                   double approximation_ratio)
    : candidate_limit_(candidate_limit), approximation_ratio_(approximation_ratio), index_(std::move(index)) {}

bool QalshAnnSearcher::RequiresIndex() const {
    return false;
}

std::unique_ptr<AnnSearcher> QalshAnnSearcher::Clone() const {
    auto clone = std::make_unique<QalshAnnSearcher>(index_, candidate_limit_, approximation_ratio_);
    clone->norm_order_ = norm_order_;
    return clone;
}

void QalshAnnSearcher::Init(const PointSetMetadata& metadata, double norm_order) {
    norm_order_ = norm_order;
    if (HasReusableIndex(metadata, norm_order)) {
        return;
    }

    index_.reset();
    index_ = BuildIndex(metadata, norm_order);
}

AnnResult QalshAnnSearcher::Search(const Point& query_point) {
    if (!index_) {
        Log::error("QALSH searcher is not initialized.");
    }

    std::vector<unsigned int> collision_count(index_->points.size(), 0);
    std::vector<bool> visited(index_->points.size(), false);
    std::priority_queue<AnnResult, std::vector<AnnResult>, CompareAnnResult> candidates;

    const unsigned int num_hash_tables = index_->config.num_hash_tables;
    const unsigned int collision_threshold = index_->config.collision_threshold;
    const double bucket_width = index_->config.bucket_width;
    const double approximation_ratio = index_->config.approximation_ratio;

    std::vector<TableCursor> cursors;
    cursors.reserve(num_hash_tables);
    for (unsigned int i = 0; i < num_hash_tables; i++) {
        const double query_key = Calc::DotProduct(query_point, index_->dot_vectors[i]);
        const auto& table = index_->tables[i];
        auto it = std::ranges::lower_bound(
            table, query_key, {}, [](const ProjectionPointIdPair& entry) { return entry.dot_product; });
        const auto table_index = static_cast<size_t>(std::distance(table.begin(), it));
        cursors.push_back(TableCursor{
            .entries = &table,
            .query_key = query_key,
            .left_index = table_index == 0 ? 0 : table_index - 1,
            .right_index = table_index,
            .has_left = table_index > 0,
            .has_right = table_index < table.size(),
        });
    }

    // c-ANN search
    double radius = 1.0;
    double bound = bucket_width * radius / 2.0;
    bool shouldTerminate = false;

    auto handle_candidate = [&](unsigned int point_id, [[maybe_unused]] double dot_product) {
        if (shouldTerminate || visited[point_id]) {
            return;
        }

        if (++collision_count[point_id] >= collision_threshold) {
            visited[point_id] = true;
            candidates.emplace(AnnResult{
                .distance = Calc::LpDistance(index_->points[point_id], query_point, norm_order_),
                .point_id = point_id,
            });

            // Termination condition 1
            if (!candidates.empty() && (candidates.top().distance <= approximation_ratio * radius)) {
                shouldTerminate = true;
                return;
            }

            // Termination condition 2
            if (candidate_limit_ && candidates.size() >= Global::kNumCandidates) {
                shouldTerminate = true;
            }
        }
    };

    auto scan_left = [&](TableCursor& cursor) {
        if (!cursor.has_left) {
            return true;
        }

        for (unsigned int scanned = 0; scanned < Global::kScanSize; ++scanned) {
            const ProjectionPointIdPair& entry = cursor.entries->at(cursor.left_index);
            if (cursor.query_key - entry.dot_product > bound) {
                return true;
            }

            handle_candidate(entry.point_id, entry.dot_product);

            if (cursor.left_index == 0) {
                cursor.has_left = false;
                return true;
            }
            --cursor.left_index;
        }

        return false;
    };

    auto scan_right = [&](TableCursor& cursor) {
        if (!cursor.has_right) {
            return true;
        }

        for (unsigned int scanned = 0; scanned < Global::kScanSize; ++scanned) {
            const ProjectionPointIdPair& entry = cursor.entries->at(cursor.right_index);
            if (entry.dot_product - cursor.query_key > bound) {
                return true;
            }

            handle_candidate(entry.point_id, entry.dot_product);

            ++cursor.right_index;
            if (cursor.right_index >= cursor.entries->size()) {
                cursor.has_right = false;
                return true;
            }
        }

        return false;
    };

    while (!shouldTerminate) {
        std::queue<unsigned int> active_tables;
        for (unsigned int i = 0; i < num_hash_tables; i++) {
            active_tables.push(i);
        }

        while (!active_tables.empty() && !shouldTerminate) {
            unsigned int i = active_tables.front();
            active_tables.pop();

            bool left_finished = scan_left(cursors[i]);
            bool right_finished = scan_right(cursors[i]);
            bool table_done = left_finished && right_finished;

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

std::optional<QalshConfig> QalshAnnSearcher::GetConfig() const {
    if (!index_) {
        return std::nullopt;
    }
    return index_->config;
}

bool QalshAnnSearcher::HasReusableIndex(const PointSetMetadata& metadata, double norm_order) const {
    return index_ && std::abs(norm_order_ - norm_order) < Global::kEpsilon && SameMetadata(index_->metadata, metadata);
}

std::shared_ptr<const InMemoryQalshIndex> QalshAnnSearcher::BuildIndex(const PointSetMetadata& metadata,
                                                                       double norm_order) const {
    auto index = std::make_shared<InMemoryQalshIndex>();
    index->metadata = metadata;
    index->config = QalshConfig{
        .approximation_ratio = approximation_ratio_,
        .page_size = Global::kDefaultPageSize,
    };
    index->config.Regularize(metadata.num_points, norm_order);

    Log::info(
        "Building in-memory QALSH index:\n"
        "\tTarget: {}\n"
        "\tApproximation Ratio: {}\n"
        "\tBucket Width: {}\n"
        "\tError Probability: {}\n"
        "\tNumber of Hash Tables: {}\n"
        "\tCollision Threshold: {}",
        metadata.file_path.stem().string(), index->config.approximation_ratio, index->config.bucket_width,
        index->config.error_probability, index->config.num_hash_tables, index->config.collision_threshold);

    index->points = PointSet::LoadPointsFromFile(metadata.file_path, metadata.num_points, metadata.num_dimensions);

    std::mt19937 gen = Random::SeededGenerator();
    std::function<double()> generator = MakeProjectionGenerator(norm_order, gen);
    index->dot_vectors.resize(index->config.num_hash_tables);
    for (Point& dot_vector : index->dot_vectors) {
        dot_vector.reserve(metadata.num_dimensions);
        std::ranges::generate_n(std::back_inserter(dot_vector), metadata.num_dimensions, generator);
    }

    index->tables.resize(index->config.num_hash_tables);
    for (std::vector<ProjectionPointIdPair>& table : index->tables) {
        table.reserve(metadata.num_points);
    }

    for (unsigned int point_id = 0; point_id < metadata.num_points; ++point_id) {
        const Point& point = index->points[point_id];
        for (unsigned int table_id = 0; table_id < index->config.num_hash_tables; ++table_id) {
            index->tables[table_id].push_back(ProjectionPointIdPair{
                .dot_product = Calc::DotProduct(point, index->dot_vectors[table_id]),
                .point_id = point_id,
            });
        }
    }

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
    for (long long table_index = 0; table_index < static_cast<long long>(index->tables.size()); ++table_index) {
        std::ranges::sort(index->tables[static_cast<size_t>(table_index)], ProjectionPointIdPairLess{});
    }
#else
    for (std::vector<ProjectionPointIdPair>& table : index->tables) {
        std::ranges::sort(table, ProjectionPointIdPairLess{});
    }
#endif

    return index;
}
