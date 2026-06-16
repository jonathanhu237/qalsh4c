#include "estimator.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ann_searcher.h"
#include "config.h"
#include "global.h"
#include "log.h"
#include "point_set.h"
#include "utils.h"
#include "weights_generator.h"

#ifdef QALSH4C_USE_OPENMP
#include <omp.h>
#endif

namespace {

constexpr size_t kMaxAutoExactScanBatchSize = 64;

unsigned int MaxExactScanThreads() {
#ifdef QALSH4C_USE_OPENMP
    return static_cast<unsigned int>(std::max(1, omp_get_max_threads()));
#else
    return 1;
#endif
}

unsigned int ResolveExactBatchSize(unsigned int requested_batch_size, size_t num_unique_samples) {
    if (requested_batch_size != 0) {
        return requested_batch_size;
    }
    if (num_unique_samples == 0) {
        return 1;
    }

    const size_t num_threads = MaxExactScanThreads();
    const size_t batch_size = (num_unique_samples + num_threads - 1) / num_threads;
    return static_cast<unsigned int>(std::clamp(batch_size, size_t{1}, kMaxAutoExactScanBatchSize));
}

std::vector<Point> LoadSampledQueryPoints(const PointSetMetadata& metadata,
                                          const std::vector<std::pair<unsigned int, unsigned int>>& sample_counts) {
    PointSet point_set;
    point_set.Init(metadata);

    std::vector<Point> query_points;
    query_points.reserve(sample_counts.size());
    for (const auto& [point_id, count] : sample_counts) {
        (void)count;
        query_points.push_back(point_set.ReadPoint(point_id));
    }

    return query_points;
}

void InitExactScanSearcher(LinearScanAnnSearcher& ann_searcher, const PointSetMetadata& metadata, double norm_order,
                           const std::shared_ptr<const std::vector<Point>>& loaded_target_points) {
    if (loaded_target_points) {
        if (loaded_target_points->size() != metadata.num_points) {
            Log::error("Loaded target point count does not match exact scan metadata.");
        }
        ann_searcher.InitLoaded(loaded_target_points, metadata.num_dimensions, norm_order);
        return;
    }
    ann_searcher.Init(metadata, norm_order);
}

double EstimateDiskBasedLinearScan(const PointSetMetadata& from, const PointSetMetadata& to, double norm_order,
                                   unsigned int disk_block_size, unsigned int disk_query_batch_size) {
    if (from.num_dimensions != to.num_dimensions) {
        Log::error("Source and target dimension mismatch: {} vs {}.", from.num_dimensions, to.num_dimensions);
    }
    if (from.num_points == 0 || to.num_points == 0 || from.num_dimensions == 0) {
        Log::error("Disk-based linear scan requires non-empty point sets with non-zero dimensions.");
    }

    std::ifstream query_input(from.file_path, std::ios::binary);
    if (!query_input.is_open()) {
        Log::error("Could not open query point set file: {}", from.file_path.string());
    }
    std::ifstream target_input(to.file_path, std::ios::binary);
    if (!target_input.is_open()) {
        Log::error("Could not open target point set file: {}", to.file_path.string());
    }

    std::vector<Coordinate> query_batch;
    std::vector<Coordinate> target_block;
    query_batch.reserve(static_cast<size_t>(disk_query_batch_size) * from.num_dimensions);
    target_block.reserve(static_cast<size_t>(disk_block_size) * to.num_dimensions);
    double estimation = 0.0;

    for (unsigned int batch_begin = 0; batch_begin < from.num_points; batch_begin += disk_query_batch_size) {
        const unsigned int query_count = std::min(disk_query_batch_size, from.num_points - batch_begin);
        const size_t query_coordinate_count = static_cast<size_t>(query_count) * from.num_dimensions;
        query_batch.resize(query_coordinate_count);
        const auto query_bytes =
            static_cast<std::streamsize>(query_coordinate_count * static_cast<size_t>(sizeof(Coordinate)));
        query_input.read(reinterpret_cast<char*>(query_batch.data()), query_bytes);
        if (query_input.gcount() != query_bytes) {
            Log::error("Failed to read query batch starting at point {} from {}.", batch_begin,
                       from.file_path.string());
        }

        std::vector<AnnResult> batch_results(query_count,
                                             AnnResult{.distance = std::numeric_limits<double>::infinity(),
                                                       .point_id = 0});

        target_input.clear();
        target_input.seekg(0, std::ios::beg);
        if (!target_input) {
            Log::error("Failed to seek target point set file: {}", to.file_path.string());
        }

        for (unsigned int block_begin = 0; block_begin < to.num_points; block_begin += disk_block_size) {
            const unsigned int target_count = std::min(disk_block_size, to.num_points - block_begin);
            const size_t target_coordinate_count = static_cast<size_t>(target_count) * to.num_dimensions;
            target_block.resize(target_coordinate_count);
            const auto target_bytes =
                static_cast<std::streamsize>(target_coordinate_count * static_cast<size_t>(sizeof(Coordinate)));
            target_input.read(reinterpret_cast<char*>(target_block.data()), target_bytes);
            if (target_input.gcount() != target_bytes) {
                Log::error("Failed to read target block starting at point {} from {}.", block_begin,
                           to.file_path.string());
            }

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (long long query_index = 0; query_index < static_cast<long long>(query_count); ++query_index) {
                const auto local_query_index = static_cast<unsigned int>(query_index);
                AnnResult& result = batch_results[local_query_index];
                const Coordinate* query_data =
                    query_batch.data() + static_cast<size_t>(local_query_index) * from.num_dimensions;
                for (unsigned int target_index = 0; target_index < target_count; ++target_index) {
                    const Coordinate* target_data =
                        target_block.data() + static_cast<size_t>(target_index) * to.num_dimensions;
                    const double distance = Calc::LpDistanceBoundedRaw(target_data, query_data, to.num_dimensions,
                                                                       norm_order, result.distance);
                    if (distance < result.distance) {
                        result.distance = distance;
                        result.point_id = block_begin + target_index;
                    }
                }
            }
        }

        for (const AnnResult& result : batch_results) {
            estimation += result.distance;
        }
    }

    return estimation;
}

double BinaryKlDivergence(double x, double y) {
    if (x < 0.0 || x > 1.0 || y <= 0.0 || y >= 1.0) {
        Log::error("Invalid KL divergence parameters: x={}, y={}", x, y);
    }

    const double left = x <= 0.0 ? 0.0 : x * std::log(x / y);
    const double right = x >= 1.0 ? 0.0 : (1.0 - x) * std::log((1.0 - x) / (1.0 - y));
    return left + right;
}

}  // namespace

// --------------------------------------------------
// AnnEstimator Implementation
// --------------------------------------------------
AnnEstimator::AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher, bool disk_based, unsigned int disk_block_size,
                           unsigned int disk_query_batch_size)
    : ann_searcher_(std::move(ann_searcher)),
      disk_based_(disk_based),
      disk_block_size_(disk_block_size),
      disk_query_batch_size_(disk_query_batch_size) {
    if (disk_block_size_ == 0) {
        Log::error("Disk block size must be positive.");
    }
    if (disk_query_batch_size_ == 0) {
        Log::error("Disk query batch size must be positive.");
    }
}

bool AnnEstimator::RequiresIndex() const {
    return ann_searcher_ && ann_searcher_->RequiresIndex();
}

double AnnEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, double norm_order) {
    // Check the ANN searcher
    if (!ann_searcher_) {
        Log::error("The ANN searcher is not set.");
    }

    if (disk_based_) {
        if (dynamic_cast<LinearScanAnnSearcher*>(ann_searcher_.get()) == nullptr) {
            Log::error("Disk-based mode is currently implemented only for ANN linear_scan.");
        }
        return EstimateDiskBasedLinearScan(from, to, norm_order, disk_block_size_, disk_query_batch_size_);
    }

    ann_searcher_->Init(to, norm_order);
    std::vector<Point> query_points = PointSet::LoadPointsFromFile(from.file_path, from.num_points, from.num_dimensions);
    std::vector<double> distances(from.num_points, 0.0);

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel
    {
        std::unique_ptr<AnnSearcher> local_searcher = ann_searcher_->Clone();

#pragma omp for schedule(dynamic)
        for (long long query_index = 0; query_index < static_cast<long long>(from.num_points); ++query_index) {
            const unsigned int point_id = static_cast<unsigned int>(query_index);
            distances[point_id] = local_searcher->Search(query_points[point_id]).distance;
        }
    }
#else
    for (unsigned int point_id = 0; point_id < from.num_points; point_id++) {
        distances[point_id] = ann_searcher_->Search(query_points[point_id]).distance;
    }
#endif

    return std::accumulate(distances.begin(), distances.end(), 0.0);
}

// --------------------------------------------------
// SamplingEstimator Implementation
// --------------------------------------------------
SamplingEstimator::SamplingEstimator(std::unique_ptr<WeightsGenerator> weights_generator,
                                     unsigned int num_samples_default, unsigned int num_samples_ab,
                                     unsigned int num_samples_ba, double epsilon, double delta, bool use_cache,
                                     unsigned int exact_batch_size, bool disk_based, unsigned int disk_block_size,
                                     unsigned int disk_query_batch_size)
    : weights_generator_(std::move(weights_generator)),
      num_samples_default_(num_samples_default),
      num_samples_ab_(num_samples_ab),
      num_samples_ba_(num_samples_ba),
      epsilon_(epsilon),
      delta_(delta),
      use_cache_(use_cache),
      exact_batch_size_(exact_batch_size),
      disk_based_(disk_based),
      disk_block_size_(disk_block_size),
      disk_query_batch_size_(disk_query_batch_size) {
    if (disk_block_size_ == 0) {
        Log::error("Disk block size must be positive.");
    }
    if (disk_query_batch_size_ == 0) {
        Log::error("Disk query batch size must be positive.");
    }
}

bool SamplingEstimator::RequiresIndex() const {
    return weights_generator_ && weights_generator_->RequiresIndex();
}

double SamplingEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to,
                                           double norm_order) {
    // Check the ANN searcher
    if (!weights_generator_) {
        Log::error("The ANN searcher is not set.");
    }
    if (disk_based_) {
        Log::error("Disk-based mode is not implemented for sampling estimators yet.");
    }

    auto* qalsh_weights_generator = dynamic_cast<QalshWeightsGenerator*>(weights_generator_.get());
    const bool weights_are_upper_bounds = qalsh_weights_generator != nullptr;

    // Determine the direction and select the appropriate sample count
    std::string from_name = from.file_path.stem().string();
    std::string to_name = to.file_path.stem().string();
    bool is_ab_direction = (from_name == "A" && to_name == "B");
    bool is_ba_direction = (from_name == "B" && to_name == "A");

    if (!is_ab_direction && !is_ba_direction) {
        Log::warn("Cannot determine direction from file names: {} -> {}, falling back to point count comparison",
                  from_name, to_name);
        is_ab_direction = (from.num_points > to.num_points);
    }

    // Select base sample count based on direction-specific parameters
    unsigned int base_num_samples = 0;
    if (is_ab_direction && num_samples_ab_ != 0) {
        base_num_samples = num_samples_ab_;
        Log::info("Using specified A->B sample count: {}", base_num_samples);
    } else if (is_ba_direction && num_samples_ba_ != 0) {
        base_num_samples = num_samples_ba_;
        Log::info("Using specified B->A sample count: {}", base_num_samples);
    } else {
        base_num_samples = num_samples_default_;
        if (base_num_samples != 0) {
            Log::info("Using default sample count: {}", base_num_samples);
        }
    }

    // Update the num_samples
    double updated_num_samples = base_num_samples;
    if (updated_num_samples == 0) {
        if (weights_are_upper_bounds) {
            const double epsilon = epsilon_;
            const double delta = delta_;
            if (epsilon <= 0.0) {
                Log::error("Epsilon must be positive for automatic QALSH sampling.");
            }
            if (delta <= 0.0 || delta >= 1.0) {
                Log::error("Delta must be in (0, 1) for automatic QALSH sampling.");
            }

            QalshConfig qalsh_config = qalsh_weights_generator->BuildConfig(to.num_points, norm_order);

            const unsigned int m = qalsh_config.num_hash_tables;
            const unsigned int l = qalsh_config.collision_threshold;
            const double c = qalsh_config.approximation_ratio;
            if (m == 0) {
                Log::error("QALSH index has zero hash tables.");
            }

            double p1 = 0.0;
            if (std::abs(norm_order - 1.0) < Global::kEpsilon) {
                p1 = Calc::L1Probability(qalsh_config.bucket_width / 2.0);
            } else if (std::abs(norm_order - 2.0) < Global::kEpsilon) {
                p1 = Calc::L2Probability(qalsh_config.bucket_width / 2.0);
            } else {
                Log::error("Unsupported norm order: {}", norm_order);
                return 0.0;
            }

            const double theta = (static_cast<double>(l) - 1.0) / static_cast<double>(m);
            if (theta < 0.0 || theta >= p1) {
                Log::error("Invalid QALSH threshold ratio for automatic sampling: theta={}, p1={}", theta, p1);
            }

            const double kl_divergence = BinaryKlDivergence(theta, p1);
            const double delta0 = std::exp(-static_cast<double>(m) * kl_divergence);
            if (c * delta0 >= 1.0) {
                Log::error("Automatic QALSH sampling requires c * delta0 < 1, got c={}, delta0={}", c, delta0);
            }

            const double kappa_exp = (c * c) * (1.0 - delta0) / (1.0 - (c * delta0));
            const double min_samples = (kappa_exp - 1.0) / (delta * epsilon * epsilon);
            updated_num_samples = static_cast<unsigned int>(std::ceil(min_samples));

            Log::info("SamplingEstimator automatically selected {} samples.", updated_num_samples);
        } else if ((dynamic_cast<UniformWeightsGenerator*>(weights_generator_.get()) != nullptr)) {
            Log::error("You must specify num_samples for UniformWeightsGenerator.");
            return 0.0;
        } else {
            Log::error("Unknown WeightsGenerator type.");
            return 0.0;
        }
    }

    // Generate weights.
    Log::info("Generating weights...");
    std::vector<double> weights = weights_generator_->Generate(from, to, norm_order, use_cache_);

    // Check the size of weights.
    if (weights.size() != from.num_points) {
        Log::error("Weights size does not match the number of query points in the dataset");
    }
    const std::shared_ptr<const std::vector<Point>> loaded_target_points =
        qalsh_weights_generator == nullptr ? nullptr : qalsh_weights_generator->TargetPoints();

    // Sample the points using the generated weights.
    WeightedSampler sampler(weights);
    const double sum = sampler.Total();
    Log::info("Total sum of weights: {}", sum);

    const unsigned int num_samples = static_cast<unsigned int>(updated_num_samples);
    std::unordered_map<unsigned int, unsigned int> sample_count_map;
    sample_count_map.reserve(num_samples);
    for (unsigned int sample_index = 0; sample_index < num_samples; ++sample_index) {
        ++sample_count_map[sampler.Sample()];
    }

    std::vector<std::pair<unsigned int, unsigned int>> sample_counts;
    sample_counts.reserve(sample_count_map.size());
    for (const auto& [point_id, count] : sample_count_map) {
        sample_counts.emplace_back(point_id, count);
    }
    std::ranges::sort(sample_counts);

    const unsigned int exact_batch_size = ResolveExactBatchSize(exact_batch_size_, sample_counts.size());
    if (exact_batch_size_ == 0) {
        Log::info("SamplingEstimator automatically selected exact batch size {} for {} unique samples and {} threads.",
                  exact_batch_size, sample_counts.size(), MaxExactScanThreads());
    } else {
        Log::info("Using specified exact batch size: {}", exact_batch_size);
    }

#ifdef QALSH4C_USE_OPENMP
    std::vector<Point> query_points = LoadSampledQueryPoints(from, sample_counts);
    LinearScanAnnSearcher shared_ann_searcher;
    InitExactScanSearcher(shared_ann_searcher, to, norm_order, loaded_target_points);
    double estimation = 0.0;

#pragma omp parallel reduction(+ : estimation)
    {
        LinearScanAnnSearcher ann_searcher = shared_ann_searcher;

        if (exact_batch_size == 1) {
#pragma omp for schedule(dynamic)
            for (long long sample_index = 0; sample_index < static_cast<long long>(sample_counts.size());
                 ++sample_index) {
                const auto& [point_id, count] = sample_counts[static_cast<size_t>(sample_index)];
                const double initial_upper_bound =
                    weights_are_upper_bounds ? weights[point_id] : std::numeric_limits<double>::infinity();
                const double nearest_distance =
                    ann_searcher.SearchWithUpperBound(query_points[static_cast<size_t>(sample_index)],
                                                      initial_upper_bound)
                        .distance;
                estimation += (static_cast<double>(count) * sum * nearest_distance / weights[point_id]);
            }
        } else {
            std::vector<Point> batch_queries;
            std::vector<double> batch_upper_bounds;
            std::vector<unsigned int> batch_point_ids;
            std::vector<unsigned int> batch_counts;
            batch_queries.reserve(exact_batch_size);
            batch_upper_bounds.reserve(exact_batch_size);
            batch_point_ids.reserve(exact_batch_size);
            batch_counts.reserve(exact_batch_size);

#pragma omp for schedule(dynamic)
            for (long long batch_begin = 0; batch_begin < static_cast<long long>(sample_counts.size());
                 batch_begin += static_cast<long long>(exact_batch_size)) {
                const size_t begin = static_cast<size_t>(batch_begin);
                const size_t end = std::min(begin + static_cast<size_t>(exact_batch_size), sample_counts.size());

                batch_queries.clear();
                batch_upper_bounds.clear();
                batch_point_ids.clear();
                batch_counts.clear();

                for (size_t sample_index = begin; sample_index < end; ++sample_index) {
                    const auto& [point_id, count] = sample_counts[sample_index];
                    batch_point_ids.push_back(point_id);
                    batch_counts.push_back(count);
                    batch_queries.push_back(query_points[sample_index]);
                    batch_upper_bounds.push_back(weights_are_upper_bounds ? weights[point_id]
                                                                          : std::numeric_limits<double>::infinity());
                }

                std::vector<AnnResult> batch_results =
                    ann_searcher.SearchBatchWithUpperBounds(batch_queries, batch_upper_bounds);
                for (size_t batch_index = 0; batch_index < batch_results.size(); ++batch_index) {
                    const unsigned int point_id = batch_point_ids[batch_index];
                    const unsigned int count = batch_counts[batch_index];
                    const double nearest_distance = batch_results[batch_index].distance;
                    estimation += (static_cast<double>(count) * sum * nearest_distance / weights[point_id]);
                }
            }
        }
    }
#else
    double estimation = 0.0;
    LinearScanAnnSearcher ann_searcher;
    InitExactScanSearcher(ann_searcher, to, norm_order, loaded_target_points);

    std::vector<Point> query_points = LoadSampledQueryPoints(from, sample_counts);

    if (exact_batch_size == 1) {
        for (size_t sample_index = 0; sample_index < sample_counts.size(); ++sample_index) {
            const auto& [point_id, count] = sample_counts[sample_index];
            const double initial_upper_bound =
                weights_are_upper_bounds ? weights[point_id] : std::numeric_limits<double>::infinity();
            const double nearest_distance =
                ann_searcher.SearchWithUpperBound(query_points[sample_index], initial_upper_bound).distance;
            estimation += (static_cast<double>(count) * sum * nearest_distance / weights[point_id]);
        }
    } else {
        std::vector<Point> batch_queries;
        std::vector<double> batch_upper_bounds;
        std::vector<unsigned int> batch_point_ids;
        std::vector<unsigned int> batch_counts;
        batch_queries.reserve(exact_batch_size);
        batch_upper_bounds.reserve(exact_batch_size);
        batch_point_ids.reserve(exact_batch_size);
        batch_counts.reserve(exact_batch_size);

        for (size_t begin = 0; begin < sample_counts.size(); begin += exact_batch_size) {
            const size_t end = std::min(begin + static_cast<size_t>(exact_batch_size), sample_counts.size());

            batch_queries.clear();
            batch_upper_bounds.clear();
            batch_point_ids.clear();
            batch_counts.clear();

            for (size_t sample_index = begin; sample_index < end; ++sample_index) {
                const auto& [point_id, count] = sample_counts[sample_index];
                batch_point_ids.push_back(point_id);
                batch_counts.push_back(count);
                batch_queries.push_back(query_points[sample_index]);
                batch_upper_bounds.push_back(weights_are_upper_bounds ? weights[point_id]
                                                                      : std::numeric_limits<double>::infinity());
            }

            std::vector<AnnResult> batch_results =
                ann_searcher.SearchBatchWithUpperBounds(batch_queries, batch_upper_bounds);
            for (size_t batch_index = 0; batch_index < batch_results.size(); ++batch_index) {
                const unsigned int point_id = batch_point_ids[batch_index];
                const unsigned int count = batch_counts[batch_index];
                const double nearest_distance = batch_results[batch_index].distance;
                estimation += (static_cast<double>(count) * sum * nearest_distance / weights[point_id]);
            }
        }
    }
#endif

    return estimation / updated_num_samples;
}
