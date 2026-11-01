#include "estimator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ann_searcher.h"
#include "config.h"
#include "exact_scan.h"
#include "global.h"
#include "log.h"
#include "point_set.h"
#include "pool_plan.h"
#include "utils.h"
#include "weights_generator.h"

#ifdef QALSH4C_USE_OPENMP
#include <omp.h>
#endif

namespace {

constexpr size_t kMaxAutoExactScanBatchSize = 64;
constexpr double kQalshSamplingTailTolerance = 1e-10;
constexpr size_t kMaxQalshSamplingSeriesIterations = 100000;

bool IsL1Norm(float norm_order) { return std::abs(norm_order - 1.0F) < Global::kEpsilon; }

bool IsL2Norm(float norm_order) { return std::abs(norm_order - 2.0F) < Global::kEpsilon; }

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

void InitExactScanSearcher(LinearScanAnnSearcher& ann_searcher, const PointSetMetadata& metadata, float norm_order,
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

std::streamoff PointFileOffset(unsigned int point_id, unsigned int num_dimensions) {
    const auto offset = static_cast<std::uint64_t>(point_id) * static_cast<std::uint64_t>(num_dimensions) *
                        static_cast<std::uint64_t>(sizeof(Coordinate));
    return static_cast<std::streamoff>(offset);
}

void ValidateDiskScanMetadata(const PointSetMetadata& from, const PointSetMetadata& to) {
    if (from.num_dimensions != to.num_dimensions) {
        Log::error("Source and target dimension mismatch: {} vs {}.", from.num_dimensions, to.num_dimensions);
    }
    if (from.num_points == 0 || to.num_points == 0 || from.num_dimensions == 0) {
        Log::error("Disk-based scan requires non-empty point sets with non-zero dimensions.");
    }
}

void OpenPointSetForRead(std::ifstream& input, const std::filesystem::path& file_path, const std::string& label) {
    input.open(file_path, std::ios::binary);
    if (!input.is_open()) {
        Log::error("Could not open {} point set file: {}", label, file_path.string());
    }
}

void ReadContiguousPointBatch(std::ifstream& input, const PointSetMetadata& metadata, unsigned int batch_begin,
                              unsigned int point_count, std::vector<Coordinate>& buffer, const std::string& label) {
    const size_t coordinate_count = static_cast<size_t>(point_count) * metadata.num_dimensions;
    buffer.resize(coordinate_count);
    const auto bytes_to_read =
        static_cast<std::streamsize>(coordinate_count * static_cast<size_t>(sizeof(Coordinate)));
    input.read(reinterpret_cast<char*>(buffer.data()), bytes_to_read);
    if (input.gcount() != bytes_to_read) {
        Log::error("Failed to read {} batch starting at point {} from {}.", label, batch_begin,
                   metadata.file_path.string());
    }
}

void ReadPointIntoBuffer(std::ifstream& input, const PointSetMetadata& metadata, unsigned int point_id,
                         Coordinate* output, const std::string& label) {
    input.clear();
    input.seekg(PointFileOffset(point_id, metadata.num_dimensions), std::ios::beg);
    if (!input) {
        Log::error("Failed to seek {} point {} in {}.", label, point_id, metadata.file_path.string());
    }
    const auto bytes_to_read =
        static_cast<std::streamsize>(static_cast<size_t>(metadata.num_dimensions) * sizeof(Coordinate));
    input.read(reinterpret_cast<char*>(output), bytes_to_read);
    if (input.gcount() != bytes_to_read) {
        Log::error("Failed to read {} point {} from {}.", label, point_id, metadata.file_path.string());
    }
}

void ScanDiskTargetBlocks(const std::vector<Coordinate>& query_batch, unsigned int query_count,
                          const PointSetMetadata& target_metadata, float norm_order, unsigned int disk_block_size,
                          std::ifstream& target_input, std::vector<Coordinate>& target_block,
                          std::vector<AnnResult>& batch_results) {
    target_input.clear();
    target_input.seekg(0, std::ios::beg);
    if (!target_input) {
        Log::error("Failed to seek target point set file: {}", target_metadata.file_path.string());
    }

    if (ExactScan::Supports(norm_order) && !Global::kUseScalarExactScan) {
        std::vector<const Coordinate*> queries(query_count);
        std::vector<Distance> upper_bounds(query_count);
        for (unsigned int query_index = 0; query_index < query_count; ++query_index) {
            queries[query_index] = query_batch.data() + static_cast<size_t>(query_index) * target_metadata.num_dimensions;
            upper_bounds[query_index] = batch_results[query_index].distance;
        }
        ExactScan scan(queries, target_metadata.num_dimensions, norm_order, upper_bounds);
        for (unsigned int block_begin = 0; block_begin < target_metadata.num_points; block_begin += disk_block_size) {
            const unsigned int target_count = std::min(disk_block_size, target_metadata.num_points - block_begin);
            ReadContiguousPointBatch(target_input, target_metadata, block_begin, target_count, target_block, "target");
            scan.ScanBlock(target_block.data(), target_count, block_begin);
        }
        batch_results = scan.Results();
        return;
    }

    for (unsigned int block_begin = 0; block_begin < target_metadata.num_points; block_begin += disk_block_size) {
        const unsigned int target_count = std::min(disk_block_size, target_metadata.num_points - block_begin);
        ReadContiguousPointBatch(target_input, target_metadata, block_begin, target_count, target_block, "target");

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long query_index = 0; query_index < static_cast<long long>(query_count); ++query_index) {
            const auto local_query_index = static_cast<unsigned int>(query_index);
            AnnResult& result = batch_results[local_query_index];
            const Coordinate* query_data =
                query_batch.data() + static_cast<size_t>(local_query_index) * target_metadata.num_dimensions;
            for (unsigned int target_index = 0; target_index < target_count; ++target_index) {
                const Coordinate* target_data =
                    target_block.data() + static_cast<size_t>(target_index) * target_metadata.num_dimensions;
                const float distance = Calc::LpDistanceBoundedRaw(target_data, query_data,
                                                                   target_metadata.num_dimensions, norm_order,
                                                                   result.distance);
                if (distance < result.distance) {
                    result.distance = distance;
                    result.point_id = block_begin + target_index;
                }
            }
        }
    }
}

std::vector<AnnResult> InitBatchResults(unsigned int query_count) {
    return std::vector<AnnResult>(query_count, AnnResult{.distance = std::numeric_limits<float>::infinity(),
                                                        .point_id = 0});
}

std::vector<AnnResult> InitBatchResultsWithUpperBounds(
    const std::vector<std::pair<unsigned int, unsigned int>>& sample_counts, size_t batch_begin,
    unsigned int query_count, const std::vector<Weight>& weights, bool use_upper_bounds) {
    std::vector<AnnResult> batch_results(query_count);
    for (unsigned int batch_index = 0; batch_index < query_count; ++batch_index) {
        const unsigned int point_id = sample_counts[batch_begin + batch_index].first;
        batch_results[batch_index] = AnnResult{
            .distance = use_upper_bounds ? weights[point_id] : std::numeric_limits<float>::infinity(),
            .point_id = 0,
        };
    }
    return batch_results;
}

float EstimateDiskBasedLinearScan(const PointSetMetadata& from, const PointSetMetadata& to, float norm_order,
                                   unsigned int disk_block_size, unsigned int disk_query_batch_size) {
    ValidateDiskScanMetadata(from, to);

    std::ifstream query_input;
    std::ifstream target_input;
    OpenPointSetForRead(query_input, from.file_path, "query");
    OpenPointSetForRead(target_input, to.file_path, "target");

    std::vector<Coordinate> query_batch;
    std::vector<Coordinate> target_block;
    query_batch.reserve(static_cast<size_t>(disk_query_batch_size) * from.num_dimensions);
    target_block.reserve(static_cast<size_t>(disk_block_size) * to.num_dimensions);
    float estimation = 0.0;

    for (unsigned int batch_begin = 0; batch_begin < from.num_points; batch_begin += disk_query_batch_size) {
        const unsigned int query_count = std::min(disk_query_batch_size, from.num_points - batch_begin);
        ReadContiguousPointBatch(query_input, from, batch_begin, query_count, query_batch, "query");

        std::vector<AnnResult> batch_results = InitBatchResults(query_count);
        ScanDiskTargetBlocks(query_batch, query_count, to, norm_order, disk_block_size, target_input, target_block,
                             batch_results);

        for (const AnnResult& result : batch_results) {
            estimation += result.distance;
        }
    }

    return estimation;
}

float EstimateDiskBasedAnnSearch(const PointSetMetadata& from, const PointSetMetadata& to, float norm_order,
                                 AnnSearcher& ann_searcher, unsigned int disk_query_batch_size) {
    ValidateDiskScanMetadata(from, to);

    auto* root_disk_qalsh_searcher = dynamic_cast<DiskBasedQalshAnnSearcher*>(&ann_searcher);
    if (root_disk_qalsh_searcher != nullptr) {
        root_disk_qalsh_searcher->ResetProfile();
    }

    ann_searcher.Init(to, norm_order);

    std::ifstream query_input;
    OpenPointSetForRead(query_input, from.file_path, "query");

    std::vector<Coordinate> query_batch;
    query_batch.reserve(static_cast<size_t>(disk_query_batch_size) * from.num_dimensions);
    std::vector<Distance> distances(disk_query_batch_size, 0.0F);
    float estimation = 0.0F;

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel
    {
        std::unique_ptr<AnnSearcher> local_searcher = ann_searcher.Clone();
        Point query_point(from.num_dimensions);

        for (unsigned int batch_begin = 0; batch_begin < from.num_points; batch_begin += disk_query_batch_size) {
            const unsigned int query_count = std::min(disk_query_batch_size, from.num_points - batch_begin);

#pragma omp single
            {
                ReadContiguousPointBatch(query_input, from, batch_begin, query_count, query_batch, "query");
                distances.assign(query_count, 0.0F);
            }

#pragma omp for schedule(dynamic)
            for (long long query_index = 0; query_index < static_cast<long long>(query_count); ++query_index) {
                const auto local_query_index = static_cast<unsigned int>(query_index);
                const Coordinate* query_data =
                    query_batch.data() + static_cast<size_t>(local_query_index) * from.num_dimensions;
                std::copy_n(query_data, from.num_dimensions, query_point.begin());
                distances[local_query_index] = local_searcher->Search(query_point).distance;
            }

#pragma omp single
            estimation += std::accumulate(distances.begin(), distances.end(), 0.0F);
        }

        if (root_disk_qalsh_searcher != nullptr) {
            auto* local_disk_qalsh_searcher = dynamic_cast<DiskBasedQalshAnnSearcher*>(local_searcher.get());
            if (local_disk_qalsh_searcher != nullptr) {
                const DiskQalshProfileSnapshot snapshot = local_disk_qalsh_searcher->ProfileSnapshot();
#pragma omp critical(disk_qalsh_profile_merge)
                root_disk_qalsh_searcher->MergeProfile(snapshot);
            }
        }
    }
#else
    for (unsigned int batch_begin = 0; batch_begin < from.num_points; batch_begin += disk_query_batch_size) {
        const unsigned int query_count = std::min(disk_query_batch_size, from.num_points - batch_begin);
        ReadContiguousPointBatch(query_input, from, batch_begin, query_count, query_batch, "query");
        distances.assign(query_count, 0.0F);
        Point query_point(from.num_dimensions);
        for (unsigned int local_query_index = 0; local_query_index < query_count; ++local_query_index) {
            const Coordinate* query_data =
                query_batch.data() + static_cast<size_t>(local_query_index) * from.num_dimensions;
            std::copy_n(query_data, from.num_dimensions, query_point.begin());
            distances[local_query_index] = ann_searcher.Search(query_point).distance;
        }

        estimation += std::accumulate(distances.begin(), distances.end(), 0.0F);
    }
#endif

    if (root_disk_qalsh_searcher != nullptr) {
        root_disk_qalsh_searcher->PrintProfile();
    }

    return estimation;
}

float EstimateDiskBasedUniformSampling(const PointSetMetadata& from, const PointSetMetadata& to, float norm_order,
                                        unsigned int num_samples, unsigned int disk_block_size,
                                        unsigned int disk_query_batch_size) {
    ValidateDiskScanMetadata(from, to);
    if (num_samples == 0) {
        Log::error("Uniform sampling requires a positive sample count.");
    }

    std::mt19937 gen = Random::SeededGenerator();
    std::uniform_real_distribution<float> distribution(0.0, static_cast<float>(from.num_points));
    std::unordered_map<unsigned int, unsigned int> sample_count_map;
    sample_count_map.reserve(num_samples);
    for (unsigned int sample_index = 0; sample_index < num_samples; ++sample_index) {
        const auto point_id = std::min(static_cast<unsigned int>(distribution(gen)), from.num_points - 1);
        ++sample_count_map[point_id];
    }

    std::vector<std::pair<unsigned int, unsigned int>> sample_counts;
    sample_counts.reserve(sample_count_map.size());
    for (const auto& [point_id, count] : sample_count_map) {
        sample_counts.emplace_back(point_id, count);
    }
    std::ranges::sort(sample_counts);

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
    float estimation = 0.0;
    const float sum = static_cast<float>(from.num_points);

    for (size_t batch_begin = 0; batch_begin < sample_counts.size(); batch_begin += disk_query_batch_size) {
        const size_t batch_end = std::min(batch_begin + static_cast<size_t>(disk_query_batch_size),
                                          sample_counts.size());
        const unsigned int query_count = static_cast<unsigned int>(batch_end - batch_begin);
        const size_t query_coordinate_count = static_cast<size_t>(query_count) * from.num_dimensions;
        query_batch.resize(query_coordinate_count);

        for (size_t sample_index = batch_begin; sample_index < batch_end; ++sample_index) {
            const unsigned int point_id = sample_counts[sample_index].first;
            Coordinate* output =
                query_batch.data() + (sample_index - batch_begin) * static_cast<size_t>(from.num_dimensions);
            ReadPointIntoBuffer(query_input, from, point_id, output, "query");
        }

        std::vector<AnnResult> batch_results = InitBatchResults(query_count);
        ScanDiskTargetBlocks(query_batch, query_count, to, norm_order, disk_block_size, target_input, target_block,
                             batch_results);

        for (size_t batch_index = 0; batch_index < batch_results.size(); ++batch_index) {
            const unsigned int count = sample_counts[batch_begin + batch_index].second;
            estimation += static_cast<float>(count) * sum * batch_results[batch_index].distance;
        }
    }

    return estimation / static_cast<float>(num_samples);
}

float EstimateDiskBasedWeightedSampling(const PointSetMetadata& from, const PointSetMetadata& to, float norm_order,
                                        unsigned int num_samples, const std::vector<Weight>& weights,
                                        bool weights_are_upper_bounds, unsigned int disk_block_size,
                                        unsigned int disk_query_batch_size) {
    ValidateDiskScanMetadata(from, to);
    if (num_samples == 0) {
        Log::error("Weighted sampling requires a positive sample count.");
    }
    if (weights.size() != from.num_points) {
        Log::error("Weights size does not match the number of query points in the dataset.");
    }

    WeightedSampler sampler(weights);
    const Weight sum = sampler.Total();
    Log::info("Total sum of weights: {}", sum);

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

    std::ifstream query_input;
    std::ifstream target_input;
    OpenPointSetForRead(query_input, from.file_path, "query");
    OpenPointSetForRead(target_input, to.file_path, "target");

    std::vector<Coordinate> query_batch;
    std::vector<Coordinate> target_block;
    query_batch.reserve(static_cast<size_t>(disk_query_batch_size) * from.num_dimensions);
    target_block.reserve(static_cast<size_t>(disk_block_size) * to.num_dimensions);
    float estimation = 0.0F;

    for (size_t batch_begin = 0; batch_begin < sample_counts.size(); batch_begin += disk_query_batch_size) {
        const size_t batch_end =
            std::min(batch_begin + static_cast<size_t>(disk_query_batch_size), sample_counts.size());
        const unsigned int query_count = static_cast<unsigned int>(batch_end - batch_begin);
        query_batch.resize(static_cast<size_t>(query_count) * from.num_dimensions);

        for (size_t sample_index = batch_begin; sample_index < batch_end; ++sample_index) {
            const unsigned int point_id = sample_counts[sample_index].first;
            Coordinate* output =
                query_batch.data() + (sample_index - batch_begin) * static_cast<size_t>(from.num_dimensions);
            ReadPointIntoBuffer(query_input, from, point_id, output, "query");
        }

        std::vector<AnnResult> batch_results =
            InitBatchResultsWithUpperBounds(sample_counts, batch_begin, query_count, weights, weights_are_upper_bounds);
        ScanDiskTargetBlocks(query_batch, query_count, to, norm_order, disk_block_size, target_input, target_block,
                             batch_results);

        for (size_t batch_index = 0; batch_index < batch_results.size(); ++batch_index) {
            const unsigned int point_id = sample_counts[batch_begin + batch_index].first;
            const unsigned int count = sample_counts[batch_begin + batch_index].second;
            if (weights[point_id] <= 0.0F) {
                Log::error("Sampled point {} has non-positive weight {}.", point_id, weights[point_id]);
            }
            estimation += static_cast<float>(count) * sum * batch_results[batch_index].distance / weights[point_id];
        }
    }

    return estimation / static_cast<float>(num_samples);
}

double LogBinomialCoefficient(unsigned int n, unsigned int k) {
    if (k > n) {
        Log::error("Invalid binomial coefficient parameters: n={}, k={}", n, k);
    }
    return std::lgamma(static_cast<double>(n) + 1.0) - std::lgamma(static_cast<double>(k) + 1.0) -
           std::lgamma(static_cast<double>(n - k) + 1.0);
}

double ExpNonNegativeLog(double log_value, const std::string& label) {
    if (log_value == -std::numeric_limits<double>::infinity()) {
        return 0.0;
    }
    if (!std::isfinite(log_value)) {
        Log::error("{} is not finite in automatic QALSH sampling: {}", label, log_value);
    }
    if (log_value > std::log(std::numeric_limits<double>::max())) {
        Log::error("{} overflows in automatic QALSH sampling: log value {}", label, log_value);
    }
    return std::exp(log_value);
}

double LogBinomialUpperTail(unsigned int n, unsigned int min_successes, double success_probability) {
    if (min_successes == 0) {
        return 0.0;
    }
    if (min_successes > n || success_probability <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    if (success_probability >= 1.0) {
        return 0.0;
    }
    if (!std::isfinite(success_probability)) {
        Log::error("Invalid binomial success probability in automatic QALSH sampling: {}", success_probability);
    }

    const double log_success = std::log(success_probability);
    const double log_failure = std::log1p(-success_probability);
    double max_log_term = -std::numeric_limits<double>::infinity();

    for (unsigned int r = min_successes; r <= n; ++r) {
        const double log_term = LogBinomialCoefficient(n, r) + static_cast<double>(r) * log_success +
                                static_cast<double>(n - r) * log_failure;
        max_log_term = std::max(max_log_term, log_term);
    }

    double scaled_sum = 0.0;
    for (unsigned int r = min_successes; r <= n; ++r) {
        const double log_term = LogBinomialCoefficient(n, r) + static_cast<double>(r) * log_success +
                                static_cast<double>(n - r) * log_failure;
        scaled_sum += std::exp(log_term - max_log_term);
    }

    const double log_tail = max_log_term + std::log(scaled_sum);
    return std::min(0.0, log_tail);
}

double QalshMissProbability(float norm_order, double bucket_width, double log_c_power) {
    if (IsL1Norm(norm_order)) {
        const double log_ratio = std::log(2.0 / bucket_width) - log_c_power;
        const double ratio =
            log_ratio < std::log(std::numeric_limits<double>::min()) ? 0.0 : std::exp(log_ratio);
        return 2.0 / std::numbers::pi_v<double> * std::atan(ratio);
    }
    if (IsL2Norm(norm_order)) {
        constexpr double kInverseTwoSqrt2 = 0.3535533905932737622;
        const double log_argument = std::log(bucket_width * kInverseTwoSqrt2) + log_c_power;
        const double argument = log_argument > std::log(std::numeric_limits<double>::max())
                                    ? std::numeric_limits<double>::infinity()
                                    : std::exp(log_argument);
        return std::erfc(argument);
    }

    Log::error("Unsupported norm order: {}", norm_order);
}

double LogL1QalshSeriesTailBound(unsigned int m, unsigned int h, double bucket_width, double approximation_ratio,
                                 size_t last_included_index) {
    const double log_c = std::log(approximation_ratio);
    const double log_denominator = std::log(-std::expm1(-static_cast<double>(h - 1) * log_c));
    return LogBinomialCoefficient(m, h) + static_cast<double>(h) * std::log(4.0 / (std::numbers::pi_v<double> * bucket_width)) -
           static_cast<double>(last_included_index + 1) * static_cast<double>(h - 1) * log_c -
           log_denominator;
}

double LogL2QalshSeriesTailBound(unsigned int m, unsigned int h, double bucket_width, double approximation_ratio,
                                 size_t last_included_index) {
    const double log_c = std::log(approximation_ratio);
    const double next_index = static_cast<double>(last_included_index + 1);
    const double log_c_power_squared = 2.0 * next_index * log_c;
    const double log_max = std::log(std::numeric_limits<double>::max());

    constexpr double kTwoSqrt2 = 2.8284271247461900976;
    const double log_a_factor = std::log(kTwoSqrt2 / (std::sqrt(std::numbers::pi_v<double>) * bucket_width));
    const double a = bucket_width * bucket_width / 8.0;
    const double c_power_squared =
        log_c_power_squared > log_max ? std::numeric_limits<double>::infinity() : std::exp(log_c_power_squared);

    double log_u = -std::numeric_limits<double>::infinity();
    double log_r = -std::numeric_limits<double>::infinity();
    if (std::isfinite(c_power_squared)) {
        log_u = LogBinomialCoefficient(m, h) + static_cast<double>(h) * log_a_factor -
                next_index * static_cast<double>(h - 1) * log_c - a * static_cast<double>(h) * c_power_squared;

        const double c_squared_minus_one = (approximation_ratio * approximation_ratio) - 1.0;
        log_r = -static_cast<double>(h - 1) * log_c -
                a * static_cast<double>(h) * c_power_squared * c_squared_minus_one;
    }

    const double denominator = -std::expm1(log_r);
    if (denominator <= 0.0 || !std::isfinite(denominator)) {
        Log::error("Invalid L2 QALSH series tail denominator in automatic sampling: {}", denominator);
    }
    return log_u - std::log(denominator);
}

double LogQalshSeriesTailBound(float norm_order, unsigned int m, unsigned int h, double bucket_width,
                               double approximation_ratio, size_t last_included_index) {
    if (IsL1Norm(norm_order)) {
        return LogL1QalshSeriesTailBound(m, h, bucket_width, approximation_ratio, last_included_index);
    }
    if (IsL2Norm(norm_order)) {
        return LogL2QalshSeriesTailBound(m, h, bucket_width, approximation_ratio, last_included_index);
    }
    Log::error("Unsupported norm order: {}", norm_order);
}

double RevisedQalshKappaExp(const QalshConfig& qalsh_config, float norm_order) {
    const unsigned int m = qalsh_config.num_hash_tables;
    const unsigned int l = qalsh_config.collision_threshold;
    const double c = static_cast<double>(qalsh_config.approximation_ratio);
    const double bucket_width = static_cast<double>(qalsh_config.bucket_width);
    if (m == 0) {
        Log::error("QALSH index has zero hash tables.");
    }
    if (l == 0 || l >= m) {
        Log::error("Automatic QALSH sampling requires collision threshold l to satisfy 1 <= l < m, got l={}, m={}",
                   l, m);
    }
    if (!std::isfinite(c) || c <= 1.0) {
        Log::error("Automatic QALSH sampling requires approximation ratio c > 1, got {}", c);
    }
    if (!std::isfinite(bucket_width) || bucket_width <= 0.0) {
        Log::error("Automatic QALSH sampling requires positive bucket width, got {}", bucket_width);
    }

    const unsigned int h = m - l + 1;
    if (h < 2 || h > m) {
        Log::error("Automatic QALSH sampling requires h = m - l + 1 in [2, m], got h={}, m={}, l={}", h, m, l);
    }

    const double log_c = std::log(c);
    const double log_tolerance = std::log(kQalshSamplingTailTolerance);
    double partial_sum = 0.0;
    double tail_bound = 0.0;
    bool converged = false;

    for (size_t j = 0; j < kMaxQalshSamplingSeriesIterations; ++j) {
        const double log_c_power = static_cast<double>(j) * log_c;
        const double miss_probability = QalshMissProbability(norm_order, bucket_width, log_c_power);
        const double log_delta = LogBinomialUpperTail(m, h, miss_probability);
        const double term = ExpNonNegativeLog(log_c_power + log_delta, "QALSH sampling series term");
        partial_sum += term;
        if (!std::isfinite(partial_sum)) {
            Log::error("QALSH sampling series partial sum is not finite.");
        }

        const double log_tail_bound = LogQalshSeriesTailBound(norm_order, m, h, bucket_width, c, j);
        if (!std::isfinite(log_tail_bound) && log_tail_bound != -std::numeric_limits<double>::infinity()) {
            Log::error("QALSH sampling series tail bound is not finite: {}", log_tail_bound);
        }
        if (log_tail_bound < log_tolerance) {
            tail_bound = ExpNonNegativeLog(log_tail_bound, "QALSH sampling series tail bound");
            converged = true;
            break;
        }
    }

    if (!converged) {
        Log::error("Automatic QALSH sampling series did not converge below {} after {} terms.",
                   kQalshSamplingTailTolerance, kMaxQalshSamplingSeriesIterations);
    }

    const double series_bound = partial_sum + tail_bound;
    const double kappa_exp = c * c * (1.0 + (c - 1.0) * series_bound);
    if (!std::isfinite(kappa_exp) || kappa_exp < 1.0) {
        Log::error("Invalid automatic QALSH sampling kappa_exp: {}", kappa_exp);
    }
    return kappa_exp;
}

unsigned int AutomaticQalshSampleCount(const QalshConfig& qalsh_config, float norm_order, double epsilon,
                                       double delta) {
    const double kappa_exp = RevisedQalshKappaExp(qalsh_config, norm_order);
    const double min_samples = (kappa_exp - 1.0) / (delta * epsilon * epsilon);
    if (!std::isfinite(min_samples) || min_samples < 0.0) {
        Log::error("Invalid automatic QALSH sampling sample count lower bound: {}", min_samples);
    }
    if (min_samples > static_cast<double>(std::numeric_limits<unsigned int>::max())) {
        Log::error("Automatic QALSH sampling selected too many samples: {}", min_samples);
    }

    return std::max(1U, static_cast<unsigned int>(std::ceil(min_samples)));
}

// Draws indices in proportion to double-precision weights. Pool estimates divide by these exact
// probabilities, so the cumulative sums must not lose precision on large source sets.
class DoubleWeightedSampler {
   public:
    explicit DoubleWeightedSampler(const std::vector<double>& weights) : gen_(Random::SeededGenerator()) {
        cumulative_.reserve(weights.size());
        double running_total = 0.0;
        for (const double weight : weights) {
            running_total += weight;
            cumulative_.push_back(running_total);
        }
        total_ = running_total;
        if (!(total_ > 0.0) || !std::isfinite(total_)) {
            Log::error("Pool sampling requires a positive finite total weight, got {}", total_);
        }
    }

    unsigned int Sample() { return Sample(gen_); }

    // Draws with the caller's generator, so that several samplers can share one random stream.
    unsigned int Sample(std::mt19937& gen) const {
        std::uniform_real_distribution<double> dis(0.0, total_);
        auto it = std::ranges::upper_bound(cumulative_, dis(gen));
        if (it == cumulative_.end()) {
            // Rounding can return the total itself; map it to the last index with positive weight.
            it = std::ranges::lower_bound(cumulative_, total_);
        }
        return static_cast<unsigned int>(std::distance(cumulative_.begin(), it));
    }

    std::mt19937& Generator() { return gen_; }

   private:
    std::vector<double> cumulative_;
    double total_{0.0};
    std::mt19937 gen_;
};

// Draws `num_draws` indices with replacement and returns (index, multiplicity) pairs sorted by index.
std::vector<std::pair<unsigned int, unsigned int>> DrawWithReplacement(const DoubleWeightedSampler& sampler,
                                                                       unsigned int num_draws, std::mt19937& gen) {
    std::unordered_map<unsigned int, unsigned int> count_map;
    count_map.reserve(num_draws);
    for (unsigned int draw = 0; draw < num_draws; ++draw) {
        ++count_map[sampler.Sample(gen)];
    }
    std::vector<std::pair<unsigned int, unsigned int>> counts(count_map.begin(), count_map.end());
    std::ranges::sort(counts);
    return counts;
}

std::vector<std::pair<unsigned int, unsigned int>> DrawWithReplacement(DoubleWeightedSampler& sampler,
                                                                       unsigned int num_draws) {
    return DrawWithReplacement(sampler, num_draws, sampler.Generator());
}

std::uint64_t HashPoint(const Point& point) { return Fingerprint::Of(point.data(), point.size()); }

// Marks source points that coincide with a target point, whose nearest-neighbor distance is zero.
std::vector<char> ExactTargetMatches(const std::vector<Point>& sources, const std::vector<Point>& targets) {
    std::vector<std::uint64_t> target_hashes(targets.size());
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long target_index = 0; target_index < static_cast<long long>(targets.size()); ++target_index) {
        target_hashes[static_cast<size_t>(target_index)] = HashPoint(targets[static_cast<size_t>(target_index)]);
    }

    std::unordered_multimap<std::uint64_t, unsigned int> target_index_by_hash;
    target_index_by_hash.reserve(targets.size());
    for (size_t target_index = 0; target_index < targets.size(); ++target_index) {
        target_index_by_hash.emplace(target_hashes[target_index], static_cast<unsigned int>(target_index));
    }

    std::vector<char> matches(sources.size(), 0);
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long source_index = 0; source_index < static_cast<long long>(sources.size()); ++source_index) {
        const Point& source = sources[static_cast<size_t>(source_index)];
        const auto [begin, end] = target_index_by_hash.equal_range(HashPoint(source));
        for (auto it = begin; it != end; ++it) {
            if (targets[it->second] == source) {
                matches[static_cast<size_t>(source_index)] = 1;
                break;
            }
        }
    }
    return matches;
}

// Ids of a uniform random subset of min(subsample_size, num_targets) target points, in ascending order.
std::vector<unsigned int> UniformSubsampleIds(size_t num_targets, unsigned int subsample_size) {
    std::vector<unsigned int> subsample_ids(num_targets);
    std::iota(subsample_ids.begin(), subsample_ids.end(), 0U);
    if (subsample_size < num_targets) {
        // Partial Fisher-Yates shuffle: the first subsample_size ids form a uniform subset.
        std::mt19937 gen = Random::SeededGenerator();
        for (size_t index = 0; index < subsample_size; ++index) {
            std::uniform_int_distribution<size_t> dis(index, num_targets - 1);
            std::swap(subsample_ids[index], subsample_ids[dis(gen)]);
        }
        subsample_ids.resize(subsample_size);
        std::ranges::sort(subsample_ids);
    }
    return subsample_ids;
}

// Distance from every source point to its nearest neighbor in a uniform target subsample. Any target point
// gives a valid upper bound on a source point's nearest-neighbor distance.
std::vector<double> SubsampleUpperBounds(const std::vector<Point>& sources, const std::vector<Point>& targets,
                                         unsigned int subsample_size, float norm_order) {
    const std::vector<unsigned int> subsample_ids = UniformSubsampleIds(targets.size(), subsample_size);

    std::vector<double> upper_bounds(sources.size(), 0.0);
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long source_index = 0; source_index < static_cast<long long>(sources.size()); ++source_index) {
        const Point& source = sources[static_cast<size_t>(source_index)];
        Distance best = std::numeric_limits<Distance>::infinity();
        for (const unsigned int target_id : subsample_ids) {
            best = std::min(best, Calc::LpDistanceBounded(targets[target_id], source, norm_order, best));
        }
        upper_bounds[static_cast<size_t>(source_index)] = static_cast<double>(best);
    }
    return upper_bounds;
}

// Exact nearest-neighbor distances for the given queries. Each scan starts from the query's upper bound, which
// must be a distance to a real target point so that the scan still returns the exact distance.
std::vector<Distance> ExactDistancesFromUpperBounds(const std::vector<Point>& queries,
                                                    const std::vector<Distance>& upper_bounds,
                                                    const LinearScanAnnSearcher& shared_searcher,
                                                    unsigned int batch_size) {
    std::vector<Distance> distances(queries.size(), 0.0F);
    if (shared_searcher.UsesBatchedScan()) {
        const std::vector<AnnResult> results = shared_searcher.SearchBatched(queries, upper_bounds);
        std::ranges::transform(results, distances.begin(), &AnnResult::distance);
        return distances;
    }
    const size_t num_batches = (queries.size() + batch_size - 1) / batch_size;

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel
#endif
    {
        LinearScanAnnSearcher searcher = shared_searcher;
        std::vector<Point> batch_queries;
        std::vector<Distance> batch_upper_bounds;

#ifdef QALSH4C_USE_OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (long long batch = 0; batch < static_cast<long long>(num_batches); ++batch) {
            const size_t begin = static_cast<size_t>(batch) * batch_size;
            const size_t end = std::min(begin + static_cast<size_t>(batch_size), queries.size());
            batch_queries.assign(queries.begin() + static_cast<std::ptrdiff_t>(begin),
                                 queries.begin() + static_cast<std::ptrdiff_t>(end));
            batch_upper_bounds.assign(upper_bounds.begin() + static_cast<std::ptrdiff_t>(begin),
                                      upper_bounds.begin() + static_cast<std::ptrdiff_t>(end));
            const std::vector<AnnResult> results =
                searcher.SearchBatchWithUpperBounds(batch_queries, batch_upper_bounds);
            for (size_t index = 0; index < results.size(); ++index) {
                distances[begin + index] = results[index].distance;
            }
        }
    }
    return distances;
}

double ElapsedMilliseconds(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

// Distances already computed for source points. Modified QALSH on a fixed index and an exact scan return the same
// distance whenever a point is queried again, so the production stages reuse a pilot's work; they never reuse its
// random draws.
struct DistanceCache {
    std::unordered_map<unsigned int, Distance> crude;  // Modified QALSH distance, clipped with U_a when fusing
    std::unordered_map<unsigned int, double> crude_query_ms;  // Time one thread spent on each of those queries
    std::unordered_map<unsigned int, unsigned int> crude_target;  // Target point each of those queries returned
    std::unordered_map<unsigned int, Distance> exact;
};

// What the pool stages read once every source point has its upper bound.
struct PoolStageInputs {
    const std::vector<Point>& sources;
    const std::vector<double>& upper_bounds;  // U_a, zero for exact target matches
    double total_upper_bound;
    const QalshAnnSearcher* qalsh_searcher;  // Null for the direct baseline, which builds no index
    const LinearScanAnnSearcher& exact_searcher;
    unsigned int exact_batch_size;  // Requested batch size for scalar exact scans (0: auto)
    bool fuse;
};

// Runs Modified QALSH on the given distinct points that have no cached crude distance yet.
void QueryCrudeDistances(const std::vector<unsigned int>& point_ids, const PoolStageInputs& inputs,
                         DistanceCache& cache) {
    std::vector<unsigned int> new_ids;
    std::ranges::copy_if(point_ids, std::back_inserter(new_ids),
                         [&cache](unsigned int point_id) { return !cache.crude.contains(point_id); });
    if (new_ids.empty()) {
        return;  // Spares cloning the searcher for every thread.
    }
    std::vector<Distance> distances(new_ids.size(), 0.0F);
    std::vector<unsigned int> found(new_ids.size(), 0);
    std::vector<double> query_ms(new_ids.size(), 0.0);
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel
#endif
    {
        std::unique_ptr<AnnSearcher> local_searcher = inputs.qalsh_searcher->Clone();
#ifdef QALSH4C_USE_OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (long long index = 0; index < static_cast<long long>(new_ids.size()); ++index) {
            const unsigned int point_id = new_ids[static_cast<size_t>(index)];
            const auto begin = std::chrono::steady_clock::now();
            const AnnResult result = local_searcher->Search(inputs.sources[point_id]);
            const Distance qalsh_distance = result.distance;
            found[static_cast<size_t>(index)] = result.point_id;
            query_ms[static_cast<size_t>(index)] = ElapsedMilliseconds(begin, std::chrono::steady_clock::now());
            // Both bounds are distances to real target points, so their minimum is still an upper bound.
            distances[static_cast<size_t>(index)] =
                inputs.fuse ? std::min(qalsh_distance, static_cast<Distance>(inputs.upper_bounds[point_id]))
                            : qalsh_distance;
        }
    }
    for (size_t index = 0; index < new_ids.size(); ++index) {
        cache.crude.emplace(new_ids[index], distances[index]);
        cache.crude_query_ms.emplace(new_ids[index], query_ms[index]);
        cache.crude_target.emplace(new_ids[index], found[index]);
    }
}

// Exact nearest-neighbor distances for the given distinct points that have no cached exact distance yet. Each scan
// starts from the point's bound, which must be a distance to a real target point.
void ScanExactDistances(const std::vector<unsigned int>& point_ids, const std::vector<Distance>& start_bounds,
                        const PoolStageInputs& inputs, DistanceCache& cache) {
    std::vector<unsigned int> new_ids;
    std::vector<Point> queries;
    std::vector<Distance> bounds;
    for (size_t index = 0; index < point_ids.size(); ++index) {
        if (!cache.exact.contains(point_ids[index])) {
            new_ids.push_back(point_ids[index]);
            queries.push_back(inputs.sources[point_ids[index]]);
            bounds.push_back(start_bounds[index]);
        }
    }
    if (new_ids.empty()) {
        return;
    }
    const std::vector<Distance> distances = ExactDistancesFromUpperBounds(
        queries, bounds, inputs.exact_searcher, ResolveExactBatchSize(inputs.exact_batch_size, queries.size()));
    for (size_t index = 0; index < new_ids.size(); ++index) {
        cache.exact.emplace(new_ids[index], distances[index]);
    }
}

enum class PoolDesign {
    kPooled,  // K draws in proportion to U_a, QALSH on the distinct ones, T resamples in proportion to D_a / U_a
    kFull,    // QALSH on every source point and T draws in proportion to D_a (the --full ablation)
    kDirect,  // No QALSH: T draws in proportion to U_a (the --direct baseline)
    kUniform,  // No bounds and no QALSH: T uniform draws, scanned from scratch (the --uniform baseline)
};

const char* PoolDesignName(PoolDesign design) {
    switch (design) {
        case PoolDesign::kPooled:
            return "pooled";
        case PoolDesign::kFull:
            return "full";
        case PoolDesign::kDirect:
            return "direct";
        case PoolDesign::kUniform:
            return "uniform";
    }
    return "unknown";
}

struct PoolRunStats {
    unsigned int pool_draws{0};
    size_t pool_distinct{0};
    size_t exact_distinct{0};
    double qalsh_ms{0.0};
    double exact_ms{0.0};
    double crude_total{0.0};
};

// Where an exact scan of a point starts: its crude distance once QALSH ran, else U_a, which is a real distance
// for every design but the uniform one, whose constant weights bound nothing.
Distance ExactStartBound(PoolDesign design, unsigned int point_id, const PoolStageInputs& inputs,
                         const DistanceCache& cache) {
    if (design == PoolDesign::kUniform) {
        return std::numeric_limits<Distance>::infinity();
    }
    if (design == PoolDesign::kDirect) {
        return static_cast<Distance>(inputs.upper_bounds[point_id]);
    }
    return cache.crude.at(point_id);
}

// One estimate with the given design and budget. Draws come from `gen` when it is given, and otherwise from each
// sampler's own generator.
double RunPoolDesign(PoolDesign design, unsigned int pool_size, unsigned int num_samples, const PoolStageInputs& inputs,
                     DistanceCache& cache, std::mt19937* gen, PoolRunStats& stats) {
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();
    const auto draw = [gen](DoubleWeightedSampler& sampler, unsigned int num_draws) {
        return DrawWithReplacement(sampler, num_draws, gen != nullptr ? *gen : sampler.Generator());
    };

    if (design == PoolDesign::kDirect || design == PoolDesign::kUniform) {
        // Baselines without QALSH: scan T points drawn in proportion to U_a, which is constant for the uniform
        // design.
        DoubleWeightedSampler direct_sampler(inputs.upper_bounds);
        const std::vector<std::pair<unsigned int, unsigned int>> selected = draw(direct_sampler, num_samples);
        std::vector<unsigned int> point_ids;
        std::vector<Distance> bounds;
        for (const auto& [point_id, multiplicity] : selected) {
            (void)multiplicity;
            point_ids.push_back(point_id);
            bounds.push_back(ExactStartBound(design, point_id, inputs, cache));
        }
        ScanExactDistances(point_ids, bounds, inputs, cache);
        double ratio_sum = 0.0;
        for (const auto& [point_id, multiplicity] : selected) {
            ratio_sum += static_cast<double>(multiplicity) * static_cast<double>(cache.exact.at(point_id)) /
                         inputs.upper_bounds[point_id];
        }
        stats.exact_distinct = selected.size();
        stats.exact_ms = ElapsedMilliseconds(begin, Clock::now());
        return inputs.total_upper_bound * ratio_sum / static_cast<double>(num_samples);
    }

    // Stage 2: K draws in proportion to U_a, or every source point with a positive bound once for the full design;
    // Modified QALSH runs once per distinct pooled point.
    DoubleWeightedSampler pool_sampler(inputs.upper_bounds);
    std::vector<std::pair<unsigned int, unsigned int>> pool;
    if (design == PoolDesign::kFull) {
        for (unsigned int point_id = 0; point_id < inputs.sources.size(); ++point_id) {
            if (inputs.upper_bounds[point_id] > 0.0) {
                pool.emplace_back(point_id, 1U);
            }
        }
    } else {
        pool = draw(pool_sampler, pool_size);
        stats.pool_draws = pool_size;
    }
    std::vector<unsigned int> pool_ids(pool.size());
    std::ranges::transform(pool, pool_ids.begin(), [](const auto& entry) { return entry.first; });
    QueryCrudeDistances(pool_ids, inputs, cache);
    stats.pool_distinct = pool.size();
    const auto pool_end = Clock::now();
    stats.qalsh_ms = ElapsedMilliseconds(begin, pool_end);

    // Stage 3: resample pool draws in proportion to D_a / U_a (D_a for the full design), then scan exactly from D_a.
    std::vector<double> resample_weights(pool.size(), 0.0);
    double pool_ratio_sum = 0.0;
    for (size_t pool_index = 0; pool_index < pool.size(); ++pool_index) {
        const auto& [point_id, multiplicity] = pool[pool_index];
        const auto crude = static_cast<double>(cache.crude.at(point_id));
        resample_weights[pool_index] = static_cast<double>(multiplicity) *
                                       (design == PoolDesign::kFull ? crude : crude / inputs.upper_bounds[point_id]);
        pool_ratio_sum += resample_weights[pool_index];
    }
    if (pool_ratio_sum <= 0.0) {
        Log::info("Every pooled point matches a target point exactly; the pooled estimate is zero.");
        return 0.0;
    }
    // W estimates the total crude distance sum_a D_a from the pool; the full design knows it exactly.
    stats.crude_total = design == PoolDesign::kFull
                            ? pool_ratio_sum
                            : inputs.total_upper_bound / static_cast<double>(pool_size) * pool_ratio_sum;

    DoubleWeightedSampler resampler(resample_weights);
    const std::vector<std::pair<unsigned int, unsigned int>> selected = draw(resampler, num_samples);
    std::vector<unsigned int> point_ids;
    std::vector<Distance> bounds;
    for (const auto& [pool_index, multiplicity] : selected) {
        (void)multiplicity;
        point_ids.push_back(pool[pool_index].first);
        bounds.push_back(cache.crude.at(pool[pool_index].first));
    }
    ScanExactDistances(point_ids, bounds, inputs, cache);

    double ratio_sum = 0.0;
    for (const auto& [pool_index, multiplicity] : selected) {
        const unsigned int point_id = pool[pool_index].first;
        ratio_sum += static_cast<double>(multiplicity) * static_cast<double>(cache.exact.at(point_id)) /
                     static_cast<double>(cache.crude.at(point_id));
    }
    stats.exact_distinct = selected.size();
    stats.exact_ms = ElapsedMilliseconds(pool_end, Clock::now());
    return stats.crude_total * ratio_sum / static_cast<double>(num_samples);
}

// Uniform sample, without replacement, of the source points with a positive bound whose Modified QALSH times price
// querying every source point.
constexpr unsigned int kFullCostProbeSize = 64;
// Pool sizes tried around the square-root rule, in steps of 2^(1/4) on either side.
constexpr int kPoolSizeSearchSteps = 8;
// Pools are never larger than this many draws per source point; beyond it the full design is cheaper anyway.
constexpr double kMaxPoolDrawsPerPoint = 64.0;
// Share of the failure probability delta that the certified plan spends on the pilot.
constexpr double kPilotFailureShare = 0.5;
// The model plan's pilot stops growing at this many draws.
constexpr size_t kMaxPilotSize = 4096;

struct PoolPlanSettings {
    PoolPlanMode mode;
    PoolDesign requested;  // kPooled lets the plan switch to kFull when that is predicted to be cheaper
    double target_error;
    double delta;
    unsigned int pilot_size;
    double pilot_relative_error;  // Target relative standard error of the model plan's coefficients (0: fixed pilot)
    unsigned int num_strata;      // Equal-count strata by U_a for the stratified design (0: unstratified)
};

struct PoolPlanChoice {
    PoolDesign design;
    unsigned int pool_size;
    unsigned int num_samples;
};

std::vector<unsigned int> UniformProbeIds(const std::vector<double>& upper_bounds, unsigned int probe_size,
                                          std::mt19937& gen) {
    std::vector<unsigned int> candidates;
    for (unsigned int point_id = 0; point_id < upper_bounds.size(); ++point_id) {
        if (upper_bounds[point_id] > 0.0) {
            candidates.push_back(point_id);
        }
    }
    const size_t probe_count = std::min<size_t>(probe_size, candidates.size());
    for (size_t index = 0; index < probe_count; ++index) {
        std::uniform_int_distribution<size_t> dis(index, candidates.size() - 1);
        std::swap(candidates[index], candidates[dis(gen)]);
    }
    candidates.resize(probe_count);
    return candidates;
}

double MeanQueryMs(const std::vector<unsigned int>& point_ids, const DistanceCache& cache) {
    double total = 0.0;
    for (const unsigned int point_id : point_ids) {
        total += cache.crude_query_ms.at(point_id);
    }
    return total / static_cast<double>(point_ids.size());
}

// Expected number of distinct source points without a cached crude distance among `pool_size` draws in proportion
// to U_a.
double ExpectedNewDistinct(const PoolStageInputs& inputs, unsigned int pool_size, const DistanceCache& cache) {
    const auto num_points = static_cast<long long>(inputs.upper_bounds.size());
    double expected = 0.0;
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for reduction(+ : expected) schedule(static)
#endif
    for (long long point_id = 0; point_id < num_points; ++point_id) {
        const double probability = inputs.upper_bounds[static_cast<size_t>(point_id)] / inputs.total_upper_bound;
        if (probability > 0.0 && !cache.crude.contains(static_cast<unsigned int>(point_id))) {
            expected -= std::expm1(static_cast<double>(pool_size) * std::log1p(-probability));
        }
    }
    return expected;
}

// Source points with a positive bound and no cached crude distance.
size_t UncachedPositivePoints(const PoolStageInputs& inputs, const DistanceCache& cache) {
    size_t count = 0;
    for (unsigned int point_id = 0; point_id < inputs.upper_bounds.size(); ++point_id) {
        if (inputs.upper_bounds[point_id] > 0.0 && !cache.crude.contains(point_id)) {
            ++count;
        }
    }
    return count;
}

// Exact scans for the distinct points without a cached exact distance, each starting from its design's bound. Adds
// their time to exact_ms and returns how many there were.
size_t TimedExactScans(const std::vector<unsigned int>& distinct, PoolDesign design, const PoolStageInputs& inputs,
                       DistanceCache& cache, double& exact_ms) {
    using Clock = std::chrono::steady_clock;
    std::vector<unsigned int> new_exact;
    std::vector<Distance> bounds;
    for (const unsigned int point_id : distinct) {
        if (!cache.exact.contains(point_id)) {
            new_exact.push_back(point_id);
            bounds.push_back(ExactStartBound(design, point_id, inputs, cache));
        }
    }
    const auto begin = Clock::now();
    ScanExactDistances(new_exact, bounds, inputs, cache);
    exact_ms += ElapsedMilliseconds(begin, Clock::now());
    return new_exact.size();
}

struct PooledBudget {
    unsigned int pool_size{0};
    unsigned int num_samples{0};
    double predicted_ms{std::numeric_limits<double>::infinity()};  // Infinite when no pool size meets the target
};

// The cheapest pooled budget among pool sizes around the square-root rule, each with the smallest sample count that
// meets eta, priced by the expected number of new QALSH queries and by the exact scans.
PooledBudget PlanPooledBudget(PoolPlan::Coefficients coefficients, double eta, double qalsh_ms_per_query,
                              double exact_ms_per_query, const PoolStageInputs& inputs, const DistanceCache& cache) {
    const unsigned int square_root_size =
        PoolPlan::SquareRootPoolSize(coefficients, eta, qalsh_ms_per_query, exact_ms_per_query);
    const double max_pool_size =
        kMaxPoolDrawsPerPoint *
        static_cast<double>(std::ranges::count_if(inputs.upper_bounds, [](double bound) { return bound > 0.0; }));
    PooledBudget best;
    for (int step = -kPoolSizeSearchSteps; step <= kPoolSizeSearchSteps; ++step) {
        const double scaled = std::round(static_cast<double>(square_root_size) * std::exp2(step / 4.0));
        const auto pool_size = static_cast<unsigned int>(std::clamp(scaled, 1.0, std::max(1.0, max_pool_size)));
        const std::optional<unsigned int> num_samples = PoolPlan::SamplesForPool(coefficients, eta, pool_size);
        if (!num_samples) {
            continue;
        }
        const double cost = qalsh_ms_per_query * ExpectedNewDistinct(inputs, pool_size, cache) +
                            exact_ms_per_query * static_cast<double>(*num_samples);
        if (cost < best.predicted_ms) {
            best = PooledBudget{.pool_size = pool_size, .num_samples = *num_samples, .predicted_ms = cost};
        }
    }
    return best;
}

// Runs the pilot and picks the design, pool size and sample count. The pilot draws points in proportion to U_a;
// every distinct one gets Modified QALSH (unless the design is direct) and an exact scan, which give the ratios
// x = v / U and z = D / U. Disjoint pairs of draws estimate or certify the variance coefficients, and the pilot's
// timings price a QALSH query and an exact scan. The plan then sizes the requested design for the target, and a
// pooled request switches to the full design when that is predicted to finish sooner.
PoolPlanChoice PlanPoolDesign(const PoolPlanSettings& settings, const PoolStageInputs& inputs, DistanceCache& cache,
                              std::mt19937& gen) {
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();
    const bool uses_qalsh =
        settings.requested != PoolDesign::kDirect && settings.requested != PoolDesign::kUniform;

    const DoubleWeightedSampler sampler(inputs.upper_bounds);
    // The model plan grows its pilot, doubling from pilot_size, until both coefficient estimates reach the estimated
    // relative standard error pilot_relative_error, a stability heuristic. The certified plan keeps its pilot fixed,
    // because its confidence bounds assume a sample size chosen in advance.
    const bool adaptive = settings.mode == PoolPlanMode::kModel && settings.pilot_relative_error > 0.0;
    std::vector<unsigned int> draws;
    std::vector<unsigned int> distinct;
    std::vector<PoolPlan::PilotDraw> pilot;
    PoolPlan::Coefficients relative_errors{};
    double exact_ms = 0.0;
    size_t exact_scans = 0;
    size_t pilot_target = settings.pilot_size;
    while (true) {
        while (draws.size() < pilot_target) {
            draws.push_back(sampler.Sample(gen));
        }
        distinct = draws;
        std::ranges::sort(distinct);
        distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
        if (uses_qalsh) {
            QueryCrudeDistances(distinct, inputs, cache);
        }
        exact_scans += TimedExactScans(distinct, settings.requested, inputs, cache, exact_ms);

        pilot.resize(draws.size());
        for (size_t index = 0; index < draws.size(); ++index) {
            const unsigned int point_id = draws[index];
            const double bound = inputs.upper_bounds[point_id];
            const double crude = uses_qalsh ? static_cast<double>(cache.crude.at(point_id)) : bound;
            pilot[index] = PoolPlan::PilotDraw{.x = static_cast<double>(cache.exact.at(point_id)) / bound,
                                               .z = crude / bound};
        }
        relative_errors = PoolPlan::RelativeStandardErrors(pilot);
        if (!adaptive || draws.size() >= kMaxPilotSize ||
            std::max(relative_errors.lambda0, relative_errors.lambda1) <= settings.pilot_relative_error) {
            break;
        }
        pilot_target = std::min(2 * pilot_target, kMaxPilotSize);
    }

    // A pilot-sized batch spends much of its wall time starting threads and waiting for the last query, so a QALSH
    // query is priced by its own thread time spread over every thread, as in a large batch.
    const auto num_threads = static_cast<double>(MaxExactScanThreads());
    const double qalsh_ms_per_query = uses_qalsh ? MeanQueryMs(distinct, cache) / num_threads : 0.0;
    const double exact_ms_per_query = exact_ms / static_cast<double>(std::max<size_t>(exact_scans, 1));
    if (!Global::kPilotDumpPath.empty()) {
        std::ofstream dump(Global::kPilotDumpPath);
        dump.precision(9);
        dump << "point_id,upper_bound,crude,exact,qalsh_target\n";
        for (const unsigned int point_id : draws) {
            const double bound = inputs.upper_bounds[point_id];
            dump << point_id << ',' << bound << ',' << (uses_qalsh ? cache.crude.at(point_id) : bound) << ','
                 << cache.exact.at(point_id) << ','
                 << (uses_qalsh ? static_cast<long long>(cache.crude_target.at(point_id)) : -1LL) << '\n';
        }
    }

    // RMS targets bound the relative variance by epsilon^2; the certified plan applies Chebyshev with the part of
    // delta that the pilot does not spend.
    double eta = settings.target_error * settings.target_error;
    std::optional<PoolPlan::Coefficients> coefficients;
    if (settings.mode == PoolPlanMode::kCertified) {
        const double alpha = kPilotFailureShare * settings.delta;
        coefficients = PoolPlan::CertifyCoefficients(pilot, alpha);
        if (!coefficients) {
            Log::error("A pilot of {} draws is too small to certify the variance coefficients; increase --pilot-size.",
                       draws.size());
        }
        eta *= settings.delta - alpha;
    } else {
        coefficients = PoolPlan::EstimateCoefficients(pilot);
    }

    PoolPlanChoice choice{.design = settings.requested, .pool_size = 0, .num_samples = 0};
    unsigned int pooled_size = 0;
    unsigned int pooled_samples = 0;
    double pooled_ms = 0.0;
    double full_ms = 0.0;
    double full_qalsh_ms_per_query = qalsh_ms_per_query;
    const unsigned int full_samples = PoolPlan::OneStageSamples(coefficients->lambda1, eta);
    if (settings.requested == PoolDesign::kDirect || settings.requested == PoolDesign::kUniform) {
        choice.num_samples = PoolPlan::OneStageSamples(coefficients->lambda0, eta);
    } else if (settings.requested == PoolDesign::kFull) {
        choice.num_samples = full_samples;
    } else {
        // The pilot's points come in proportion to U_a like the pool's, while querying every source point costs what
        // a uniform sample of them costs.
        const std::vector<unsigned int> probe = UniformProbeIds(inputs.upper_bounds, kFullCostProbeSize, gen);
        QueryCrudeDistances(probe, inputs, cache);
        full_qalsh_ms_per_query = MeanQueryMs(probe, cache) / num_threads;

        const PooledBudget budget =
            PlanPooledBudget(*coefficients, eta, qalsh_ms_per_query, exact_ms_per_query, inputs, cache);
        pooled_size = budget.pool_size;
        pooled_samples = budget.num_samples;
        pooled_ms = budget.predicted_ms;
        full_ms = full_qalsh_ms_per_query * static_cast<double>(UncachedPositivePoints(inputs, cache)) +
                  exact_ms_per_query * static_cast<double>(full_samples);
        if (pooled_ms <= full_ms) {
            choice = PoolPlanChoice{.design = PoolDesign::kPooled, .pool_size = pooled_size, .num_samples = pooled_samples};
        } else {
            choice = PoolPlanChoice{.design = PoolDesign::kFull, .pool_size = 0, .num_samples = full_samples};
        }
    }

    Log::info(
        "Pool plan: mode={} design={} target_error={:.4g} eta={:.4e} pilot_draws={} pilot_distinct={} lambda0={:.6f} "
        "lambda1={:.6f} pilot_rse0={:.3f} pilot_rse1={:.3f} qalsh_ms={:.5f} full_qalsh_ms={:.5f} exact_ms={:.5f} "
        "pooled_K={} pooled_T={} pooled_predicted_ms={:.2f} full_T={} full_predicted_ms={:.2f} pilot_ms={:.3f}",
        settings.mode == PoolPlanMode::kCertified ? "certified" : "model", PoolDesignName(choice.design),
        settings.target_error, eta, draws.size(), distinct.size(), coefficients->lambda0, coefficients->lambda1,
        relative_errors.lambda0, relative_errors.lambda1, qalsh_ms_per_query, full_qalsh_ms_per_query,
        exact_ms_per_query, pooled_size, pooled_samples, pooled_ms, full_samples, full_ms,
        ElapsedMilliseconds(begin, Clock::now()));
    return choice;
}

// Stratified design (--strata): the source points with a positive bound are split into equal-count strata by U_a.
// A stratum either draws in proportion to U_a without QALSH, or runs Modified QALSH on every one of its points and
// draws in proportion to D_a. The stratum estimates are independent, so their variances add up.
struct StratifiedPlan {
    std::vector<std::vector<unsigned int>> members;  // Source point ids of each stratum, by increasing U_a
    std::vector<char> queried;                       // Whether the stratum runs QALSH on all of its points
    std::vector<unsigned int> num_samples;           // Exact draws T_h of each stratum
    std::optional<PoolPlanChoice> pooled;            // Set when the unstratified pooled design is predicted cheaper
};

// More strata than this would make the exhaustive choice of queried strata slow and leave few pilot draws for each.
constexpr unsigned int kMaxStrata = 8;

std::vector<std::vector<unsigned int>> BuildStrata(const std::vector<double>& upper_bounds, unsigned int num_strata) {
    std::vector<unsigned int> ids;
    for (unsigned int point_id = 0; point_id < upper_bounds.size(); ++point_id) {
        if (upper_bounds[point_id] > 0.0) {
            ids.push_back(point_id);
        }
    }
    std::ranges::stable_sort(ids, [&upper_bounds](unsigned int left, unsigned int right) {
        return upper_bounds[left] < upper_bounds[right];
    });
    const size_t count = std::min<size_t>(num_strata, ids.size());
    std::vector<std::vector<unsigned int>> strata(count);
    for (size_t stratum = 0; stratum < count; ++stratum) {
        strata[stratum].assign(ids.begin() + static_cast<std::ptrdiff_t>(ids.size() * stratum / count),
                               ids.begin() + static_cast<std::ptrdiff_t>(ids.size() * (stratum + 1) / count));
    }
    return strata;
}

// The weight a stratum draws its points with: D_a once QALSH ran on the whole stratum, U_a otherwise.
double StratumWeight(bool queried, unsigned int point_id, const PoolStageInputs& inputs, const DistanceCache& cache) {
    return queried ? static_cast<double>(cache.crude.at(point_id)) : inputs.upper_bounds[point_id];
}

// Runs a pilot in every stratum, then picks the strata that run QALSH and the draws of each stratum. Each stratum
// gets an equal share of the pilot, drawn in proportion to U_a within the stratum; every distinct pilot point gets
// Modified QALSH and an exact scan. The pilot moments give each stratum's per-draw standard deviation with and
// without QALSH, the strata's draws follow the Neyman allocation for the target, and the plan queries the subset of
// strata that minimizes the predicted QALSH and exact cost. A model plan also prices the unstratified pooled design
// from the same pilot and switches to it when that is predicted to be cheaper.
StratifiedPlan PlanStratifiedDesign(const PoolPlanSettings& settings, const PoolStageInputs& inputs,
                                    DistanceCache& cache, std::mt19937& gen) {
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();
    StratifiedPlan plan{.members = BuildStrata(inputs.upper_bounds, settings.num_strata), .queried = {},
                        .num_samples = {}, .pooled = std::nullopt};
    const size_t num_strata = plan.members.size();
    const auto draws_per_stratum = std::max<size_t>(4, settings.pilot_size / num_strata);

    std::vector<double> bound_totals(num_strata, 0.0);
    std::vector<std::vector<unsigned int>> pilot_ids(num_strata);
    std::vector<unsigned int> distinct;
    for (size_t stratum = 0; stratum < num_strata; ++stratum) {
        const std::vector<unsigned int>& members = plan.members[stratum];
        std::vector<double> weights(members.size());
        std::ranges::transform(members, weights.begin(),
                               [&inputs](unsigned int point_id) { return inputs.upper_bounds[point_id]; });
        bound_totals[stratum] = std::accumulate(weights.begin(), weights.end(), 0.0);
        const DoubleWeightedSampler sampler(weights);
        for (size_t draw = 0; draw < draws_per_stratum; ++draw) {
            pilot_ids[stratum].push_back(members[sampler.Sample(gen)]);
        }
        distinct.insert(distinct.end(), pilot_ids[stratum].begin(), pilot_ids[stratum].end());
    }
    std::ranges::sort(distinct);
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    QueryCrudeDistances(distinct, inputs, cache);
    double exact_ms = 0.0;
    const size_t exact_scans = TimedExactScans(distinct, PoolDesign::kPooled, inputs, cache, exact_ms);
    const double exact_ms_per_query = exact_ms / static_cast<double>(std::max<size_t>(exact_scans, 1));
    const auto num_threads = static_cast<double>(MaxExactScanThreads());

    // The certified plan spends its share of delta on the pilot bounds, split evenly across strata, and applies
    // Chebyshev with the rest.
    double eta = settings.target_error * settings.target_error;
    const double alpha = kPilotFailureShare * settings.delta;
    if (settings.mode == PoolPlanMode::kCertified) {
        eta *= settings.delta - alpha;
    }
    std::vector<double> sigma_bound(num_strata);
    std::vector<double> sigma_crude(num_strata);
    std::vector<double> qalsh_ms(num_strata);
    double total = 0.0;
    // The unstratified pooled design reads every stratum's pilot, weighted by the stratum's share of the bound total,
    // and pays the QALSH cost of a query drawn in proportion to U_a.
    std::vector<std::vector<PoolPlan::PilotDraw>> strata_pilots(num_strata);
    std::vector<double> shares(num_strata);
    double pool_query_ms = 0.0;
    for (size_t stratum = 0; stratum < num_strata; ++stratum) {
        std::vector<PoolPlan::PilotDraw> pilot;
        for (const unsigned int point_id : pilot_ids[stratum]) {
            const double bound = inputs.upper_bounds[point_id];
            pilot.push_back(PoolPlan::PilotDraw{.x = static_cast<double>(cache.exact.at(point_id)) / bound,
                                                .z = static_cast<double>(cache.crude.at(point_id)) / bound});
        }
        std::optional<PoolPlan::StratumMoments> moments;
        if (settings.mode == PoolPlanMode::kCertified) {
            moments = PoolPlan::CertifyStratumMoments(pilot, alpha / static_cast<double>(num_strata));
            if (!moments) {
                Log::error("A pilot of {} draws per stratum is too small to certify the stratum moments; increase "
                           "--pilot-size.",
                           draws_per_stratum);
            }
        } else {
            moments = PoolPlan::EstimateStratumMoments(pilot);
        }
        sigma_bound[stratum] = bound_totals[stratum] * std::sqrt(moments->h0);
        sigma_crude[stratum] = bound_totals[stratum] * std::sqrt(moments->h1);
        total += bound_totals[stratum] * moments->mean_ratio;
        // QALSH on the stratum's uncached points, priced by the stratum's own pilot queries.
        std::vector<unsigned int> stratum_distinct = pilot_ids[stratum];
        std::ranges::sort(stratum_distinct);
        stratum_distinct.erase(std::unique(stratum_distinct.begin(), stratum_distinct.end()), stratum_distinct.end());
        const auto uncached = std::ranges::count_if(
            plan.members[stratum], [&cache](unsigned int point_id) { return !cache.crude.contains(point_id); });
        const double query_ms = MeanQueryMs(stratum_distinct, cache) / num_threads;
        qalsh_ms[stratum] = query_ms * static_cast<double>(uncached);

        shares[stratum] = bound_totals[stratum] / inputs.total_upper_bound;
        pool_query_ms += shares[stratum] * query_ms;
        strata_pilots[stratum] = std::move(pilot);
    }
    std::optional<PooledBudget> pooled_budget;
    PoolPlan::Coefficients coefficients{.lambda0 = 0.0, .lambda1 = 0.0};
    if (settings.mode == PoolPlanMode::kModel) {
        coefficients = PoolPlan::PooledCoefficients(strata_pilots, shares);
        pooled_budget = PlanPooledBudget(coefficients, eta, pool_query_ms, exact_ms_per_query, inputs, cache);
    }

    double best_ms = std::numeric_limits<double>::infinity();
    for (unsigned int mask = 0; mask < (1U << num_strata); ++mask) {
        std::vector<double> sigmas(num_strata);
        double predicted_ms = 0.0;
        for (size_t stratum = 0; stratum < num_strata; ++stratum) {
            const bool queried = ((mask >> stratum) & 1U) != 0;
            sigmas[stratum] = queried ? sigma_crude[stratum] : sigma_bound[stratum];
            predicted_ms += queried ? qalsh_ms[stratum] : 0.0;
        }
        const std::vector<unsigned int> samples = PoolPlan::StratifiedSamples(sigmas, eta, total);
        predicted_ms += exact_ms_per_query * static_cast<double>(std::accumulate(samples.begin(), samples.end(), 0ULL));
        if (predicted_ms < best_ms) {
            best_ms = predicted_ms;
            plan.num_samples = samples;
            plan.queried.assign(num_strata, 0);
            for (size_t stratum = 0; stratum < num_strata; ++stratum) {
                plan.queried[stratum] = static_cast<char>((mask >> stratum) & 1U);
            }
        }
    }

    if (pooled_budget && pooled_budget->predicted_ms < best_ms) {
        plan.pooled = PoolPlanChoice{.design = PoolDesign::kPooled,
                                     .pool_size = pooled_budget->pool_size,
                                     .num_samples = pooled_budget->num_samples};
    }

    std::string queried_strata;
    std::string samples_text;
    size_t qalsh_points = 0;
    for (size_t stratum = 0; stratum < num_strata; ++stratum) {
        queried_strata += plan.queried[stratum] != 0 ? 'Q' : '.';
        samples_text += (stratum == 0 ? "" : "/") + std::to_string(plan.num_samples[stratum]);
        qalsh_points += plan.queried[stratum] != 0 ? plan.members[stratum].size() : 0;
    }
    Log::info(
        "Pool plan: mode={} design={} target_error={:.4g} eta={:.4e} pilot_draws={} pilot_distinct={} lambda0={:.6f} "
        "lambda1={:.6f} qalsh_ms={:.5f} exact_ms={:.5f} pooled_K={} pooled_T={} pooled_predicted_ms={:.2f} strata={} "
        "queried={} stratum_T={} qalsh_points={} stratified_predicted_ms={:.2f} pilot_ms={:.3f}",
        settings.mode == PoolPlanMode::kCertified ? "certified" : "model", plan.pooled ? "pooled" : "stratified",
        settings.target_error, eta, draws_per_stratum * num_strata, distinct.size(), coefficients.lambda0,
        coefficients.lambda1, pool_query_ms, exact_ms_per_query, pooled_budget ? pooled_budget->pool_size : 0U,
        pooled_budget ? pooled_budget->num_samples : 0U, pooled_budget ? pooled_budget->predicted_ms : 0.0, num_strata,
        queried_strata, samples_text, qalsh_points, best_ms, ElapsedMilliseconds(begin, Clock::now()));
    return plan;
}

// One estimate with a stratified plan: QALSH on every point of the queried strata, then each stratum's draws in
// proportion to its weights, all scanned in one batch.
double RunStratifiedDesign(const StratifiedPlan& plan, const PoolStageInputs& inputs, DistanceCache& cache,
                           std::mt19937& gen, PoolRunStats& stats) {
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();
    std::vector<unsigned int> queried_ids;
    for (size_t stratum = 0; stratum < plan.members.size(); ++stratum) {
        if (plan.queried[stratum] != 0) {
            queried_ids.insert(queried_ids.end(), plan.members[stratum].begin(), plan.members[stratum].end());
        }
    }
    QueryCrudeDistances(queried_ids, inputs, cache);
    stats.pool_distinct = queried_ids.size();
    const auto qalsh_end = Clock::now();
    stats.qalsh_ms = ElapsedMilliseconds(begin, qalsh_end);

    std::vector<std::vector<std::pair<unsigned int, unsigned int>>> selected(plan.members.size());
    std::vector<double> weight_totals(plan.members.size());
    std::vector<unsigned int> point_ids;
    std::vector<Distance> bounds;
    for (size_t stratum = 0; stratum < plan.members.size(); ++stratum) {
        const bool queried = plan.queried[stratum] != 0;
        const std::vector<unsigned int>& members = plan.members[stratum];
        std::vector<double> weights(members.size());
        std::ranges::transform(members, weights.begin(), [&](unsigned int point_id) {
            return StratumWeight(queried, point_id, inputs, cache);
        });
        weight_totals[stratum] = std::accumulate(weights.begin(), weights.end(), 0.0);
        selected[stratum] = DrawWithReplacement(DoubleWeightedSampler(weights), plan.num_samples[stratum], gen);
        for (auto& [index, multiplicity] : selected[stratum]) {
            (void)multiplicity;
            index = members[index];
            point_ids.push_back(index);
            bounds.push_back(static_cast<Distance>(StratumWeight(queried, index, inputs, cache)));
        }
    }
    // Strata are disjoint, so no point is listed twice.
    ScanExactDistances(point_ids, bounds, inputs, cache);

    double estimate = 0.0;
    for (size_t stratum = 0; stratum < plan.members.size(); ++stratum) {
        const bool queried = plan.queried[stratum] != 0;
        double ratio_sum = 0.0;
        for (const auto& [point_id, multiplicity] : selected[stratum]) {
            ratio_sum += static_cast<double>(multiplicity) * static_cast<double>(cache.exact.at(point_id)) /
                         StratumWeight(queried, point_id, inputs, cache);
        }
        estimate += weight_totals[stratum] * ratio_sum / static_cast<double>(plan.num_samples[stratum]);
    }
    stats.exact_distinct = point_ids.size();
    stats.exact_ms = ElapsedMilliseconds(qalsh_end, Clock::now());
    return estimate;
}

}  // namespace

void Estimator::PreflightBothDirections([[maybe_unused]] const PointSetMetadata& a_metadata,
                                        [[maybe_unused]] const PointSetMetadata& b_metadata,
                                        [[maybe_unused]] float norm_order) {}

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

void AnnEstimator::PreflightBothDirections(const PointSetMetadata& a_metadata, const PointSetMetadata& b_metadata,
                                           float norm_order) {
    if (!disk_based_) {
        return;
    }
    auto* disk_qalsh_searcher = dynamic_cast<DiskBasedQalshAnnSearcher*>(ann_searcher_.get());
    if (disk_qalsh_searcher == nullptr) {
        return;
    }

    disk_qalsh_searcher->PreflightIndex(b_metadata, norm_order);
    disk_qalsh_searcher->PreflightIndex(a_metadata, norm_order);
}

float AnnEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, float norm_order) {
    // Check the ANN searcher
    if (!ann_searcher_) {
        Log::error("The ANN searcher is not set.");
    }

    if (disk_based_) {
        if (dynamic_cast<LinearScanAnnSearcher*>(ann_searcher_.get()) != nullptr) {
            return EstimateDiskBasedLinearScan(from, to, norm_order, disk_block_size_, disk_query_batch_size_);
        }
        auto* disk_qalsh_searcher = dynamic_cast<DiskBasedQalshAnnSearcher*>(ann_searcher_.get());
        if (disk_qalsh_searcher != nullptr) {
            return EstimateDiskBasedAnnSearch(from, to, norm_order, *disk_qalsh_searcher, disk_query_batch_size_);
        }
        Log::error("Disk-based mode is currently implemented only for ANN linear_scan and qalsh.");
    }

    ann_searcher_->Init(to, norm_order);
    std::vector<Point> query_points = PointSet::LoadPointsFromFile(from.file_path, from.num_points, from.num_dimensions);
    std::vector<Distance> distances(from.num_points, 0.0F);

    const auto* linear_scan = dynamic_cast<const LinearScanAnnSearcher*>(ann_searcher_.get());
    if (linear_scan != nullptr && linear_scan->UsesBatchedScan()) {
        const std::vector<AnnResult> results = linear_scan->SearchBatched(
            query_points, std::vector<Distance>(query_points.size(), std::numeric_limits<Distance>::infinity()));
        std::ranges::transform(results, distances.begin(), &AnnResult::distance);
        return std::accumulate(distances.begin(), distances.end(), 0.0F);
    }

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

    return std::accumulate(distances.begin(), distances.end(), 0.0F);
}

// --------------------------------------------------
// SamplingEstimator Implementation
// --------------------------------------------------
SamplingEstimator::SamplingEstimator(std::unique_ptr<WeightsGenerator> weights_generator,
                                     unsigned int num_samples_default, unsigned int num_samples_ab,
                                     unsigned int num_samples_ba, float epsilon, float delta, bool use_cache,
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

void SamplingEstimator::PreflightBothDirections(const PointSetMetadata& a_metadata,
                                                const PointSetMetadata& b_metadata, float norm_order) {
    if (!disk_based_) {
        return;
    }
    auto* qalsh_weights_generator = dynamic_cast<QalshWeightsGenerator*>(weights_generator_.get());
    if (qalsh_weights_generator == nullptr) {
        return;
    }

    qalsh_weights_generator->PreflightGenerate(a_metadata, b_metadata, norm_order, use_cache_);
    qalsh_weights_generator->PreflightGenerate(b_metadata, a_metadata, norm_order, use_cache_);
}

float SamplingEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to,
                                           float norm_order) {
    // Check the ANN searcher
    if (!weights_generator_) {
        Log::error("The ANN searcher is not set.");
    }

    auto* qalsh_weights_generator = dynamic_cast<QalshWeightsGenerator*>(weights_generator_.get());
    auto* uniform_weights_generator = dynamic_cast<UniformWeightsGenerator*>(weights_generator_.get());
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
    float updated_num_samples = static_cast<float>(base_num_samples);
    if (updated_num_samples == 0.0F) {
        if (weights_are_upper_bounds) {
            const float epsilon = epsilon_;
            const float delta = delta_;
            if (epsilon <= 0.0F) {
                Log::error("Epsilon must be positive for automatic QALSH sampling.");
            }
            if (delta <= 0.0F || delta >= 1.0F) {
                Log::error("Delta must be in (0, 1) for automatic QALSH sampling.");
            }

            QalshConfig qalsh_config = qalsh_weights_generator->BuildConfig(from, to, norm_order, use_cache_);

            const unsigned int m = qalsh_config.num_hash_tables;
            if (m == 0) {
                Log::error("QALSH index has zero hash tables.");
            }

            const unsigned int selected_samples =
                AutomaticQalshSampleCount(qalsh_config, norm_order, epsilon, delta);
            updated_num_samples = static_cast<float>(selected_samples);

            Log::info("SamplingEstimator automatically selected {} samples.", selected_samples);
        } else if (uniform_weights_generator != nullptr) {
            Log::error("You must specify num_samples for UniformWeightsGenerator.");
            return 0.0F;
        } else {
            Log::error("Unknown WeightsGenerator type.");
            return 0.0F;
        }
    }

    const unsigned int num_samples = static_cast<unsigned int>(updated_num_samples);
    if (disk_based_) {
        if (uniform_weights_generator != nullptr) {
            return EstimateDiskBasedUniformSampling(from, to, norm_order, num_samples, disk_block_size_,
                                                    disk_query_batch_size_);
        }
        if (qalsh_weights_generator != nullptr) {
            Log::info("Generating weights...");
            std::vector<Weight> weights = weights_generator_->Generate(from, to, norm_order, use_cache_);
            return EstimateDiskBasedWeightedSampling(from, to, norm_order, num_samples, weights,
                                                     weights_are_upper_bounds, disk_block_size_,
                                                     disk_query_batch_size_);
        }
        Log::error("Disk-based mode is currently implemented only for uniform and QALSH sampling estimators.");
    }

    // Generate weights.
    Log::info("Generating weights...");
    std::vector<Weight> weights = weights_generator_->Generate(from, to, norm_order, use_cache_);

    // Check the size of weights.
    if (weights.size() != from.num_points) {
        Log::error("Weights size does not match the number of query points in the dataset");
    }
    const std::shared_ptr<const std::vector<Point>> loaded_target_points =
        qalsh_weights_generator == nullptr ? nullptr : qalsh_weights_generator->TargetPoints();

    // Sample the points using the generated weights.
    WeightedSampler sampler(weights);
    const Weight sum = sampler.Total();
    Log::info("Total sum of weights: {}", sum);

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

    std::vector<Point> query_points = LoadSampledQueryPoints(from, sample_counts);
    LinearScanAnnSearcher shared_ann_searcher;
    InitExactScanSearcher(shared_ann_searcher, to, norm_order, loaded_target_points);
    Estimate estimation = 0.0F;

    if (shared_ann_searcher.UsesBatchedScan()) {
        std::vector<Distance> upper_bounds(sample_counts.size());
        for (size_t sample_index = 0; sample_index < sample_counts.size(); ++sample_index) {
            upper_bounds[sample_index] = weights_are_upper_bounds ? weights[sample_counts[sample_index].first]
                                                                  : std::numeric_limits<float>::infinity();
        }
        const std::vector<AnnResult> results = shared_ann_searcher.SearchBatched(query_points, upper_bounds);
        for (size_t sample_index = 0; sample_index < sample_counts.size(); ++sample_index) {
            const auto& [point_id, count] = sample_counts[sample_index];
            estimation += (static_cast<float>(count) * sum * results[sample_index].distance / weights[point_id]);
        }
        return estimation / updated_num_samples;
    }

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel reduction(+ : estimation)
    {
        LinearScanAnnSearcher ann_searcher = shared_ann_searcher;

        if (exact_batch_size == 1) {
#pragma omp for schedule(dynamic)
            for (long long sample_index = 0; sample_index < static_cast<long long>(sample_counts.size());
                 ++sample_index) {
                const auto& [point_id, count] = sample_counts[static_cast<size_t>(sample_index)];
                const Distance initial_upper_bound =
                    weights_are_upper_bounds ? weights[point_id] : std::numeric_limits<float>::infinity();
                const Distance nearest_distance =
                    ann_searcher.SearchWithUpperBound(query_points[static_cast<size_t>(sample_index)],
                                                      initial_upper_bound)
                        .distance;
                estimation += (static_cast<float>(count) * sum * nearest_distance / weights[point_id]);
            }
        } else {
            std::vector<Point> batch_queries;
            std::vector<Distance> batch_upper_bounds;
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
                                                                          : std::numeric_limits<float>::infinity());
                }

                std::vector<AnnResult> batch_results =
                    ann_searcher.SearchBatchWithUpperBounds(batch_queries, batch_upper_bounds);
                for (size_t batch_index = 0; batch_index < batch_results.size(); ++batch_index) {
                    const unsigned int point_id = batch_point_ids[batch_index];
                    const unsigned int count = batch_counts[batch_index];
                    const Distance nearest_distance = batch_results[batch_index].distance;
                    estimation += (static_cast<float>(count) * sum * nearest_distance / weights[point_id]);
                }
            }
        }
    }
#else
    LinearScanAnnSearcher& ann_searcher = shared_ann_searcher;

    if (exact_batch_size == 1) {
        for (size_t sample_index = 0; sample_index < sample_counts.size(); ++sample_index) {
            const auto& [point_id, count] = sample_counts[sample_index];
            const Distance initial_upper_bound =
                weights_are_upper_bounds ? weights[point_id] : std::numeric_limits<float>::infinity();
            const Distance nearest_distance =
                ann_searcher.SearchWithUpperBound(query_points[sample_index], initial_upper_bound).distance;
            estimation += (static_cast<float>(count) * sum * nearest_distance / weights[point_id]);
        }
    } else {
        std::vector<Point> batch_queries;
        std::vector<Distance> batch_upper_bounds;
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
                                                                      : std::numeric_limits<float>::infinity());
            }

            std::vector<AnnResult> batch_results =
                ann_searcher.SearchBatchWithUpperBounds(batch_queries, batch_upper_bounds);
            for (size_t batch_index = 0; batch_index < batch_results.size(); ++batch_index) {
                const unsigned int point_id = batch_point_ids[batch_index];
                const unsigned int count = batch_counts[batch_index];
                const Distance nearest_distance = batch_results[batch_index].distance;
                estimation += (static_cast<float>(count) * sum * nearest_distance / weights[point_id]);
            }
        }
    }
#endif

    return estimation / updated_num_samples;
}

// --------------------------------------------------
// PoolSamplingEstimator Implementation
// --------------------------------------------------
PoolSamplingEstimator::PoolSamplingEstimator(unsigned int pool_size, unsigned int num_samples,
                                             unsigned int subsample_size, float approximation_ratio, float epsilon,
                                             float delta, unsigned int exact_batch_size, bool disk_based,
                                             unsigned int disk_block_size, unsigned int disk_query_batch_size,
                                             std::optional<float> expected_approximation_ratio,
                                             DiskQalshSearchOptions disk_qalsh_options, bool direct, bool full,
                                             bool fuse, PoolPlanMode plan_mode, float target_error,
                                             unsigned int pilot_size, bool uniform, float pilot_relative_error,
                                             unsigned int num_strata)
    : pool_size_(pool_size),
      num_samples_(num_samples),
      subsample_size_(subsample_size),
      approximation_ratio_(approximation_ratio),
      epsilon_(epsilon),
      delta_(delta),
      exact_batch_size_(exact_batch_size),
      disk_based_(disk_based),
      disk_block_size_(disk_block_size),
      disk_query_batch_size_(disk_query_batch_size),
      expected_approximation_ratio_(expected_approximation_ratio),
      disk_qalsh_options_(disk_qalsh_options),
      direct_(direct),
      full_(full),
      fuse_(fuse),
      plan_mode_(plan_mode),
      target_error_(target_error),
      pilot_size_(pilot_size),
      uniform_(uniform),
      pilot_relative_error_(pilot_relative_error),
      num_strata_(num_strata) {
    if (full_ && disk_based_) {
        Log::error("The --full ablation is available only in memory.");
    }
    if (subsample_size_ == 0) {
        Log::error("Pool sampling requires a positive subsample size.");
    }
    if (!std::isfinite(approximation_ratio_) || approximation_ratio_ <= 1.0F) {
        Log::error("Pool sampling requires approximation ratio c > 1, got {}", approximation_ratio_);
    }
    if (disk_block_size_ == 0 || disk_query_batch_size_ == 0) {
        Log::error("Disk block size and disk query batch size must be positive.");
    }
    if (uniform_ && (direct_ || full_ || disk_based_)) {
        Log::error("The --uniform baseline runs in memory and cannot be combined with --direct or --full.");
    }
    if (plan_mode_ != PoolPlanMode::kFixed) {
        if (disk_based_) {
            Log::error("--plan is available only in memory.");
        }
        if (pool_size_ != 0 || num_samples_ != 0) {
            Log::error("--plan chooses the pool size and the sample count; do not pass -K or -n with it.");
        }
        if (!(target_error_ > 0.0F && target_error_ < 1.0F)) {
            Log::error("--plan needs a target error in (0, 1), got {}", target_error_);
        }
        if (pilot_size_ < 4) {
            Log::error("--plan needs a pilot of at least four draws, got {}", pilot_size_);
        }
        if (!(pilot_relative_error_ >= 0.0F)) {
            Log::error("--pilot-relative-error must not be negative, got {}", pilot_relative_error_);
        }
    }
    if (num_strata_ != 0) {
        if (plan_mode_ == PoolPlanMode::kFixed) {
            Log::error("--strata needs --plan model or certified.");
        }
        if (direct_ || full_ || uniform_) {
            Log::error("--strata decides per stratum whether to run QALSH; do not combine it with --direct, --full or "
                       "--uniform.");
        }
        if (num_strata_ > kMaxStrata) {
            Log::error("--strata must be at most {}, got {}", kMaxStrata, num_strata_);
        }
        if (pilot_size_ < 4 * num_strata_) {
            Log::error("--strata needs a pilot of at least four draws per stratum, got {} for {} strata", pilot_size_,
                       num_strata_);
        }
    }
    if (plan_mode_ == PoolPlanMode::kCertified) {
        if (!(delta_ > 0.0F && delta_ < 1.0F)) {
            Log::error("The certified plan needs a failure probability in (0, 1), got {}", delta_);
        }
        if (!fuse_ && !direct_) {
            // The certificate needs D_a <= U_a, which only clipping guarantees.
            Log::error("The certified plan cannot be combined with --no-fuse.");
        }
        if (uniform_) {
            // The certificate needs ratios v_a / U_a in (0, 1], which constant weights do not give.
            Log::error("The certified plan cannot be combined with --uniform.");
        }
    }
}

bool PoolSamplingEstimator::RequiresIndex() const { return disk_based_; }

void PoolSamplingEstimator::PreflightBothDirections(const PointSetMetadata& a_metadata,
                                                    const PointSetMetadata& b_metadata, float norm_order) {
    if (!disk_based_) {
        return;
    }
    const DiskBasedQalshAnnSearcher searcher(expected_approximation_ratio_, disk_qalsh_options_);
    searcher.PreflightIndex(b_metadata, norm_order);
    searcher.PreflightIndex(a_metadata, norm_order);
}

unsigned int PoolSamplingEstimator::ResolveNumSamples(const QalshConfig& qalsh_config, float norm_order) const {
    if (num_samples_ != 0) {
        return num_samples_;
    }
    const unsigned int num_samples = AutomaticQalshSampleCount(qalsh_config, norm_order, epsilon_, delta_);
    Log::info("PoolSamplingEstimator automatically selected {} second-stage samples.", num_samples);
    return num_samples;
}

unsigned int PoolSamplingEstimator::ResolvePoolSize(unsigned int num_samples) const {
    if (pool_size_ != 0) {
        return pool_size_;
    }
    const unsigned int pool_size = kAutoPoolSizePerSample * num_samples;
    Log::info("PoolSamplingEstimator automatically selected pool size {} ({} x second-stage samples).", pool_size,
              kAutoPoolSizePerSample);
    return pool_size;
}

float PoolSamplingEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to,
                                               float norm_order) {
    if (disk_based_) {
        return EstimateDiskBased(from, to, norm_order);
    }
    using Clock = std::chrono::steady_clock;
    if (from.num_dimensions != to.num_dimensions) {
        Log::error("Source and target dimension mismatch: {} vs {}.", from.num_dimensions, to.num_dimensions);
    }

    // A plan picks K and T after its pilot instead.
    const bool planned = plan_mode_ != PoolPlanMode::kFixed;
    QalshConfig qalsh_config{.approximation_ratio = approximation_ratio_};
    qalsh_config.Regularize(to.num_points, norm_order);
    const unsigned int num_samples = planned ? 0 : ResolveNumSamples(qalsh_config, norm_order);
    const unsigned int pool_size = planned || direct_ || full_ || uniform_ ? 0 : ResolvePoolSize(num_samples);

    // Build the QALSH index on the target set; its loaded target points are shared by every later stage.
    const auto index_begin = Clock::now();
    QalshAnnSearcher qalsh_searcher(approximation_ratio_);
    std::shared_ptr<const std::vector<Point>> targets;
    if (direct_ || uniform_) {
        targets = std::make_shared<const std::vector<Point>>(
            PointSet::LoadPointsFromFile(to.file_path, to.num_points, to.num_dimensions));
    } else {
        qalsh_searcher.Init(to, norm_order);
        targets = qalsh_searcher.TargetPoints();
    }
    const auto index_end = Clock::now();

    // Stage 1: a cheap upper bound U_a for every source point; exact target matches contribute zero. The uniform
    // baseline weighs every source point equally instead.
    const std::vector<Point> sources = PointSet::LoadPointsFromFile(from.file_path, from.num_points, from.num_dimensions);
    std::vector<double> upper_bounds(sources.size(), 1.0);
    unsigned int num_exact_matches = 0;
    if (!uniform_) {
        upper_bounds = SubsampleUpperBounds(sources, *targets, subsample_size_, norm_order);
        const std::vector<char> exact_matches = ExactTargetMatches(sources, *targets);
        for (size_t source_index = 0; source_index < sources.size(); ++source_index) {
            if (exact_matches[source_index] != 0) {
                upper_bounds[source_index] = 0.0;
                ++num_exact_matches;
            }
        }
    }
    const double total_upper_bound = std::accumulate(upper_bounds.begin(), upper_bounds.end(), 0.0);
    const auto upper_bound_end = Clock::now();
    if (total_upper_bound <= 0.0) {
        Log::info("Every source point matches a target point exactly; the distance is zero.");
        return 0.0F;
    }

    LinearScanAnnSearcher exact_searcher;
    exact_searcher.InitLoaded(targets, to.num_dimensions, norm_order);
    const PoolStageInputs inputs{
        .sources = sources,
        .upper_bounds = upper_bounds,
        .total_upper_bound = total_upper_bound,
        .qalsh_searcher = direct_ || uniform_ ? nullptr : &qalsh_searcher,
        .exact_searcher = exact_searcher,
        .exact_batch_size = exact_batch_size_,
        .fuse = fuse_,
    };
    PoolDesign design = uniform_  ? PoolDesign::kUniform
                        : direct_ ? PoolDesign::kDirect
                        : full_   ? PoolDesign::kFull
                                  : PoolDesign::kPooled;
    unsigned int run_pool_size = pool_size;
    unsigned int run_num_samples = num_samples;
    DistanceCache cache;
    PoolRunStats stats;
    double estimate = 0.0;
    bool ran_stratified = false;
    if (planned) {
        // One random stream serves the pilot and the production draws, which keeps them independent even with a
        // fixed seed.
        std::mt19937 gen = Random::SeededGenerator();
        const PoolPlanSettings settings{.mode = plan_mode_,
                                        .requested = design,
                                        .target_error = target_error_,
                                        .delta = delta_,
                                        .pilot_size = pilot_size_,
                                        .pilot_relative_error = pilot_relative_error_,
                                        .num_strata = num_strata_};
        if (num_strata_ != 0) {
            const StratifiedPlan plan = PlanStratifiedDesign(settings, inputs, cache, gen);
            if (plan.pooled) {
                design = plan.pooled->design;
                run_pool_size = plan.pooled->pool_size;
                run_num_samples = plan.pooled->num_samples;
                estimate = RunPoolDesign(design, run_pool_size, run_num_samples, inputs, cache, &gen, stats);
            } else {
                ran_stratified = true;
                run_num_samples = std::accumulate(plan.num_samples.begin(), plan.num_samples.end(), 0U);
                estimate = RunStratifiedDesign(plan, inputs, cache, gen, stats);
            }
        } else {
            const PoolPlanChoice choice = PlanPoolDesign(settings, inputs, cache, gen);
            design = choice.design;
            run_pool_size = choice.pool_size;
            run_num_samples = choice.num_samples;
            estimate = RunPoolDesign(design, run_pool_size, run_num_samples, inputs, cache, &gen, stats);
        }
    } else {
        estimate = RunPoolDesign(design, run_pool_size, run_num_samples, inputs, cache, nullptr, stats);
    }

    Log::info(
        "Pool sampling stats: subsample_size={} exact_matches={} pool_draws={} pool_distinct={} exact_draws={} "
        "exact_distinct={} index_ms={:.3f} upper_bound_ms={:.3f} pool_qalsh_ms={:.3f} exact_ms={:.3f}",
        uniform_ ? 0 : std::min<size_t>(subsample_size_, targets->size()), num_exact_matches, stats.pool_draws,
        stats.pool_distinct,
        run_num_samples, stats.exact_distinct, ElapsedMilliseconds(index_begin, index_end),
        ElapsedMilliseconds(index_end, upper_bound_end), stats.qalsh_ms, stats.exact_ms);
    if (!ran_stratified && (design == PoolDesign::kPooled || design == PoolDesign::kFull)) {
        Log::info("Total upper bound: {:.6f}; estimated crude total: {:.6f}", total_upper_bound, stats.crude_total);
    }
    return static_cast<float>(estimate);
}

float PoolSamplingEstimator::EstimateDiskBased(const PointSetMetadata& from, const PointSetMetadata& to,
                                               float norm_order) {
    using Clock = std::chrono::steady_clock;
    ValidateDiskScanMetadata(from, to);
    const unsigned int num_dimensions = from.num_dimensions;
    const unsigned int num_samples =
        num_samples_ != 0
            ? num_samples_
            : ResolveNumSamples(DiskQalsh::ValidateIndex(to, norm_order, expected_approximation_ratio_).ToConfig(),
                                norm_order);
    const unsigned int pool_size = direct_ ? 0 : ResolvePoolSize(num_samples);
    std::ifstream query_input;
    std::ifstream target_input;
    OpenPointSetForRead(query_input, from.file_path, "query");
    OpenPointSetForRead(target_input, to.file_path, "target");

    // Stage 1: stream A once against a uniform target subsample held in memory.
    const auto upper_bound_begin = Clock::now();
    const std::vector<unsigned int> subsample_ids = UniformSubsampleIds(to.num_points, subsample_size_);
    std::vector<Coordinate> subsample(subsample_ids.size() * num_dimensions);
    for (size_t index = 0; index < subsample_ids.size(); ++index) {
        ReadPointIntoBuffer(target_input, to, subsample_ids[index], subsample.data() + index * num_dimensions,
                            "target");
    }
    // Target fingerprints come from the persistent index; indexes built before they were stored fall back to hashing
    // every target point in one sequential pass. Source points whose hash matches are verified below.
    std::unordered_multimap<std::uint64_t, unsigned int> target_index_by_hash;
    target_index_by_hash.reserve(to.num_points);
    std::vector<Coordinate> target_block;
    std::vector<std::uint64_t> block_hashes;
    const std::optional<std::vector<std::uint64_t>> stored_fingerprints = DiskQalsh::ReadFingerprints(to, norm_order);
    if (stored_fingerprints.has_value()) {
        for (unsigned int target_index = 0; target_index < to.num_points; ++target_index) {
            target_index_by_hash.emplace((*stored_fingerprints)[target_index], target_index);
        }
    }
    target_input.clear();
    target_input.seekg(0, std::ios::beg);
    for (unsigned int block_begin = 0; !stored_fingerprints.has_value() && block_begin < to.num_points;
         block_begin += disk_block_size_) {
        const unsigned int target_count = std::min(disk_block_size_, to.num_points - block_begin);
        ReadContiguousPointBatch(target_input, to, block_begin, target_count, target_block, "target");
        block_hashes.resize(target_count);
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long target_index = 0; target_index < static_cast<long long>(target_count); ++target_index) {
            block_hashes[static_cast<size_t>(target_index)] = Fingerprint::Of(
                target_block.data() + static_cast<size_t>(target_index) * num_dimensions, num_dimensions);
        }
        for (unsigned int target_index = 0; target_index < target_count; ++target_index) {
            target_index_by_hash.emplace(block_hashes[target_index], block_begin + target_index);
        }
    }

    std::vector<double> upper_bounds(from.num_points, 0.0);
    std::vector<std::uint64_t> source_hashes(from.num_points, 0);
    std::vector<Coordinate> query_batch;
    for (unsigned int batch_begin = 0; batch_begin < from.num_points; batch_begin += disk_block_size_) {
        const unsigned int query_count = std::min(disk_block_size_, from.num_points - batch_begin);
        ReadContiguousPointBatch(query_input, from, batch_begin, query_count, query_batch, "query");
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long query_index = 0; query_index < static_cast<long long>(query_count); ++query_index) {
            const Coordinate* query = query_batch.data() + static_cast<size_t>(query_index) * num_dimensions;
            Distance best = std::numeric_limits<Distance>::infinity();
            for (size_t index = 0; index < subsample_ids.size(); ++index) {
                best = std::min(best, Calc::LpDistanceBoundedRaw(subsample.data() + index * num_dimensions, query,
                                                                 num_dimensions, norm_order, best));
            }
            upper_bounds[batch_begin + static_cast<size_t>(query_index)] = static_cast<double>(best);
            source_hashes[batch_begin + static_cast<size_t>(query_index)] = Fingerprint::Of(query, num_dimensions);
        }
    }

    // Exact target matches contribute zero; hash hits are confirmed by reading both points.
    unsigned int num_exact_matches = 0;
    std::vector<Coordinate> source_point(num_dimensions);
    std::vector<Coordinate> target_point(num_dimensions);
    for (unsigned int source_index = 0; source_index < from.num_points; ++source_index) {
        const auto [begin, end] = target_index_by_hash.equal_range(source_hashes[source_index]);
        if (begin == end || upper_bounds[source_index] == 0.0) {
            continue;
        }
        ReadPointIntoBuffer(query_input, from, source_index, source_point.data(), "query");
        for (auto it = begin; it != end; ++it) {
            ReadPointIntoBuffer(target_input, to, it->second, target_point.data(), "target");
            if (source_point == target_point) {
                upper_bounds[source_index] = 0.0;
                ++num_exact_matches;
                break;
            }
        }
    }
    const double total_upper_bound = std::accumulate(upper_bounds.begin(), upper_bounds.end(), 0.0);
    const auto upper_bound_end = Clock::now();
    if (total_upper_bound <= 0.0) {
        Log::info("Every source point matches a target point exactly; the distance is zero.");
        return 0.0F;
    }

    if (direct_) {
        // Baseline without QALSH: scan T points drawn in proportion to U_a, starting from U_a.
        DoubleWeightedSampler direct_sampler(upper_bounds);
        const std::vector<std::pair<unsigned int, unsigned int>> selected =
            DrawWithReplacement(direct_sampler, num_samples);
        double ratio_sum = 0.0;
        for (size_t batch_begin = 0; batch_begin < selected.size(); batch_begin += disk_query_batch_size_) {
            const auto query_count =
                static_cast<unsigned int>(std::min<size_t>(disk_query_batch_size_, selected.size() - batch_begin));
            query_batch.resize(static_cast<size_t>(query_count) * num_dimensions);
            std::vector<AnnResult> batch_results(query_count);
            for (unsigned int batch_index = 0; batch_index < query_count; ++batch_index) {
                const unsigned int point_id = selected[batch_begin + batch_index].first;
                ReadPointIntoBuffer(query_input, from, point_id,
                                    query_batch.data() + static_cast<size_t>(batch_index) * num_dimensions, "query");
                batch_results[batch_index] =
                    AnnResult{.distance = static_cast<Distance>(upper_bounds[point_id]), .point_id = 0};
            }
            ScanDiskTargetBlocks(query_batch, query_count, to, norm_order, disk_block_size_, target_input,
                                 target_block, batch_results);
            for (unsigned int batch_index = 0; batch_index < query_count; ++batch_index) {
                const auto& [point_id, multiplicity] = selected[batch_begin + batch_index];
                ratio_sum += static_cast<double>(multiplicity) *
                             static_cast<double>(batch_results[batch_index].distance) / upper_bounds[point_id];
            }
        }
        const auto exact_end = Clock::now();
        Log::info(
            "Pool sampling stats: subsample_size={} exact_matches={} pool_draws=0 pool_distinct=0 exact_draws={} "
            "exact_distinct={} index_ms=0.000 upper_bound_ms={:.3f} pool_qalsh_ms=0.000 exact_ms={:.3f}",
            subsample_ids.size(), num_exact_matches, num_samples, selected.size(),
            ElapsedMilliseconds(upper_bound_begin, upper_bound_end), ElapsedMilliseconds(upper_bound_end, exact_end));
        return static_cast<float>(total_upper_bound * ratio_sum / static_cast<double>(num_samples));
    }

    // Stage 2: read the pooled points and probe the persistent QALSH index only for them.
    DoubleWeightedSampler pool_sampler(upper_bounds);
    const std::vector<std::pair<unsigned int, unsigned int>> pool = DrawWithReplacement(pool_sampler, pool_size);
    std::vector<Coordinate> pooled_points(pool.size() * num_dimensions);
    for (size_t pool_index = 0; pool_index < pool.size(); ++pool_index) {
        ReadPointIntoBuffer(query_input, from, pool[pool_index].first,
                            pooled_points.data() + pool_index * num_dimensions, "query");
    }

    DiskBasedQalshAnnSearcher qalsh_searcher(expected_approximation_ratio_, disk_qalsh_options_);
    qalsh_searcher.ResetProfile();
    qalsh_searcher.Init(to, norm_order);
    const auto index_open_end = Clock::now();
    std::vector<Distance> crude_distances(pool.size(), 0.0F);
#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel
#endif
    {
        std::unique_ptr<AnnSearcher> local_searcher = qalsh_searcher.Clone();
        Point query_point(num_dimensions);
#ifdef QALSH4C_USE_OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (long long pool_index = 0; pool_index < static_cast<long long>(pool.size()); ++pool_index) {
            const auto index = static_cast<size_t>(pool_index);
            std::copy_n(pooled_points.data() + index * num_dimensions, num_dimensions, query_point.begin());
            const Distance qalsh_distance = local_searcher->Search(query_point).distance;
            crude_distances[index] =
                fuse_ ? std::min(qalsh_distance, static_cast<Distance>(upper_bounds[pool[index].first]))
                      : qalsh_distance;
        }
        auto* local_disk_searcher = dynamic_cast<DiskBasedQalshAnnSearcher*>(local_searcher.get());
        if (local_disk_searcher != nullptr) {
            const DiskQalshProfileSnapshot snapshot = local_disk_searcher->ProfileSnapshot();
#ifdef QALSH4C_USE_OPENMP
#pragma omp critical(pool_disk_qalsh_profile_merge)
#endif
            qalsh_searcher.MergeProfile(snapshot);
        }
    }
    qalsh_searcher.PrintProfile();
    const auto pool_end = Clock::now();

    // Stage 3: resample pool draws in proportion to D_a / U_a, then scan B sequentially once per query batch.
    std::vector<double> resample_weights(pool.size(), 0.0);
    double pool_ratio_sum = 0.0;
    for (size_t pool_index = 0; pool_index < pool.size(); ++pool_index) {
        const auto& [point_id, multiplicity] = pool[pool_index];
        resample_weights[pool_index] = static_cast<double>(multiplicity) *
                                       static_cast<double>(crude_distances[pool_index]) / upper_bounds[point_id];
        pool_ratio_sum += resample_weights[pool_index];
    }
    if (pool_ratio_sum <= 0.0) {
        Log::info("Every pooled point matches a target point exactly; the pooled estimate is zero.");
        return 0.0F;
    }
    const double crude_total = total_upper_bound / static_cast<double>(pool_size) * pool_ratio_sum;
    DoubleWeightedSampler resampler(resample_weights);
    const std::vector<std::pair<unsigned int, unsigned int>> selected = DrawWithReplacement(resampler, num_samples);
    double ratio_sum = 0.0;
    for (size_t batch_begin = 0; batch_begin < selected.size(); batch_begin += disk_query_batch_size_) {
        const auto query_count =
            static_cast<unsigned int>(std::min<size_t>(disk_query_batch_size_, selected.size() - batch_begin));
        query_batch.resize(static_cast<size_t>(query_count) * num_dimensions);
        std::vector<AnnResult> batch_results(query_count);
        for (unsigned int batch_index = 0; batch_index < query_count; ++batch_index) {
            const unsigned int pool_index = selected[batch_begin + batch_index].first;
            std::copy_n(pooled_points.data() + static_cast<size_t>(pool_index) * num_dimensions, num_dimensions,
                        query_batch.data() + static_cast<size_t>(batch_index) * num_dimensions);
            batch_results[batch_index] = AnnResult{.distance = crude_distances[pool_index], .point_id = 0};
        }
        ScanDiskTargetBlocks(query_batch, query_count, to, norm_order, disk_block_size_, target_input, target_block,
                             batch_results);
        for (unsigned int batch_index = 0; batch_index < query_count; ++batch_index) {
            const auto& [pool_index, multiplicity] = selected[batch_begin + batch_index];
            ratio_sum += static_cast<double>(multiplicity) * static_cast<double>(batch_results[batch_index].distance) /
                         static_cast<double>(crude_distances[pool_index]);
        }
    }
    const double estimate = crude_total * ratio_sum / static_cast<double>(num_samples);
    const auto exact_end = Clock::now();

    Log::info(
        "Pool sampling stats: subsample_size={} exact_matches={} pool_draws={} pool_distinct={} exact_draws={} "
        "exact_distinct={} index_ms={:.3f} upper_bound_ms={:.3f} pool_qalsh_ms={:.3f} exact_ms={:.3f}",
        subsample_ids.size(), num_exact_matches, pool_size, pool.size(), num_samples, selected.size(),
        ElapsedMilliseconds(upper_bound_end, index_open_end), ElapsedMilliseconds(upper_bound_begin, upper_bound_end),
        ElapsedMilliseconds(index_open_end, pool_end), ElapsedMilliseconds(pool_end, exact_end));
    Log::info("Total upper bound: {:.6f}; estimated crude total: {:.6f}", total_upper_bound, crude_total);
    return static_cast<float>(estimate);
}
