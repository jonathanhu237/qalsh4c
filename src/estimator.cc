#include "estimator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <random>
#include <string>
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

#ifdef QALSH4C_USE_OPENMP
    std::vector<Point> query_points = LoadSampledQueryPoints(from, sample_counts);
    LinearScanAnnSearcher shared_ann_searcher;
    InitExactScanSearcher(shared_ann_searcher, to, norm_order, loaded_target_points);
    Estimate estimation = 0.0F;

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
    Estimate estimation = 0.0F;
    LinearScanAnnSearcher ann_searcher;
    InitExactScanSearcher(ann_searcher, to, norm_order, loaded_target_points);

    std::vector<Point> query_points = LoadSampledQueryPoints(from, sample_counts);

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
