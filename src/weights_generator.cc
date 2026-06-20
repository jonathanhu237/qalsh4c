#include "weights_generator.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "ann_searcher.h"
#include "config.h"
#include "disk_qalsh_index.h"
#include "log.h"
#include "point_set.h"

#ifdef QALSH4C_USE_OPENMP
#include <omp.h>
#endif

namespace {

std::filesystem::path QalshWeightsPath(const PointSetMetadata& from_metadata, float norm_order) {
    std::filesystem::path parent_directory = from_metadata.file_path.parent_path();
    std::string stem = from_metadata.file_path.stem();
    return parent_directory / std::format("l{}_qalsh_weights_{}.bin", norm_order, stem);
}

std::streamsize ExpectedWeightsBytes(const PointSetMetadata& from_metadata) {
    return static_cast<std::streamsize>(sizeof(Weight) * static_cast<size_t>(from_metadata.num_points));
}

bool HasValidQalshWeightsCache(const PointSetMetadata& from_metadata, float norm_order) {
    const std::filesystem::path weights_path = QalshWeightsPath(from_metadata, norm_order);
    const std::streamsize expected_bytes = ExpectedWeightsBytes(from_metadata);
    if (!std::filesystem::exists(weights_path)) {
        return false;
    }

    const auto actual_bytes = static_cast<std::streamsize>(std::filesystem::file_size(weights_path));
    if (actual_bytes != expected_bytes) {
        Log::error("QALSH weights cache file has unexpected size: expected {} bytes, got {} bytes from {}",
                   expected_bytes, actual_bytes, weights_path.string());
    }
    return true;
}

bool ReadQalshWeightsCacheIfAvailable(const PointSetMetadata& from_metadata, float norm_order,
                                      std::vector<Weight>& weights) {
    if (!HasValidQalshWeightsCache(from_metadata, norm_order)) {
        return false;
    }

    const std::filesystem::path weights_path = QalshWeightsPath(from_metadata, norm_order);
    const std::streamsize expected_bytes = ExpectedWeightsBytes(from_metadata);
    std::ifstream input(weights_path, std::ios::binary);
    if (!input.is_open()) {
        Log::error("Failed to open QALSH weights cache file: {}", weights_path.string());
    }
    input.read(reinterpret_cast<char*>(weights.data()), expected_bytes);
    if (input.gcount() != expected_bytes) {
        Log::error("Failed to read QALSH weights cache file: expected {} bytes, read {} bytes from {}",
                   expected_bytes, input.gcount(), weights_path.string());
    }
    return true;
}

void WriteQalshWeightsCache(const PointSetMetadata& from_metadata, float norm_order,
                            const std::vector<Weight>& weights) {
    const std::filesystem::path weights_path = QalshWeightsPath(from_metadata, norm_order);
    const std::streamsize expected_bytes = ExpectedWeightsBytes(from_metadata);
    std::ofstream output(weights_path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        Log::error("Failed to open QALSH weights cache file for writing: {}", weights_path.string());
    }
    output.write(reinterpret_cast<const char*>(weights.data()), expected_bytes);
}

void ReadContiguousPointBatch(std::ifstream& input, const PointSetMetadata& metadata, unsigned int batch_begin,
                              unsigned int point_count, std::vector<Coordinate>& buffer) {
    const size_t coordinate_count = static_cast<size_t>(point_count) * metadata.num_dimensions;
    buffer.resize(coordinate_count);
    const auto bytes_to_read =
        static_cast<std::streamsize>(coordinate_count * static_cast<size_t>(sizeof(Coordinate)));
    input.read(reinterpret_cast<char*>(buffer.data()), bytes_to_read);
    if (input.gcount() != bytes_to_read) {
        Log::error("Failed to read query batch starting at point {} from {}.", batch_begin,
                   metadata.file_path.string());
    }
}

void GenerateWeightsFromSearcherStreaming(AnnSearcher& ann_searcher, const PointSetMetadata& from_metadata,
                                          unsigned int query_batch_size, std::vector<Weight>& weights) {
    if (query_batch_size == 0) {
        Log::error("Disk query batch size must be positive.");
    }

    auto* root_disk_qalsh_searcher = dynamic_cast<DiskBasedQalshAnnSearcher*>(&ann_searcher);
    if (root_disk_qalsh_searcher != nullptr) {
        root_disk_qalsh_searcher->ResetProfile();
    }

    std::ifstream query_input(from_metadata.file_path, std::ios::binary);
    if (!query_input.is_open()) {
        Log::error("Could not open query point set file: {}", from_metadata.file_path.string());
    }

    std::vector<Coordinate> query_batch;
    query_batch.reserve(static_cast<size_t>(query_batch_size) * from_metadata.num_dimensions);

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel
    {
        std::unique_ptr<AnnSearcher> local_searcher = ann_searcher.Clone();
        Point query_point(from_metadata.num_dimensions);

        for (unsigned int batch_begin = 0; batch_begin < from_metadata.num_points; batch_begin += query_batch_size) {
            const unsigned int query_count = std::min(query_batch_size, from_metadata.num_points - batch_begin);

#pragma omp single
            ReadContiguousPointBatch(query_input, from_metadata, batch_begin, query_count, query_batch);

#pragma omp for schedule(dynamic)
            for (long long query_index = 0; query_index < static_cast<long long>(query_count); ++query_index) {
                const auto local_query_index = static_cast<unsigned int>(query_index);
                const Coordinate* query_data =
                    query_batch.data() + static_cast<size_t>(local_query_index) * from_metadata.num_dimensions;
                std::copy_n(query_data, from_metadata.num_dimensions, query_point.begin());
                weights[batch_begin + local_query_index] = local_searcher->Search(query_point).distance;
            }
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
    for (unsigned int batch_begin = 0; batch_begin < from_metadata.num_points; batch_begin += query_batch_size) {
        const unsigned int query_count = std::min(query_batch_size, from_metadata.num_points - batch_begin);
        ReadContiguousPointBatch(query_input, from_metadata, batch_begin, query_count, query_batch);

        Point query_point(from_metadata.num_dimensions);
        for (unsigned int local_query_index = 0; local_query_index < query_count; ++local_query_index) {
            const Coordinate* query_data =
                query_batch.data() + static_cast<size_t>(local_query_index) * from_metadata.num_dimensions;
            std::copy_n(query_data, from_metadata.num_dimensions, query_point.begin());
            weights[batch_begin + local_query_index] = ann_searcher.Search(query_point).distance;
        }
    }
#endif
}

}  // namespace

// --------------------------------------------------
// UniformWeightsGenerator Implementation
// --------------------------------------------------
bool UniformWeightsGenerator::RequiresIndex() const {
    return false;
}

std::vector<Weight> UniformWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                      [[maybe_unused]] const PointSetMetadata& to_metadata,
                                                      [[maybe_unused]] float norm_order,
                                                      [[maybe_unused]] bool use_cache) {
    std::vector<Weight> weights(from_metadata.num_points, 1.0);
    return weights;
}

// --------------------------------------------------
// QalshWeightsGenerator Implementation
// --------------------------------------------------
QalshWeightsGenerator::QalshWeightsGenerator(float approximation_ratio)
    : approximation_ratio_(approximation_ratio),
      ann_searcher_(std::make_unique<QalshAnnSearcher>(approximation_ratio_)) {}

QalshWeightsGenerator::QalshWeightsGenerator(std::optional<float> expected_approximation_ratio,
                                             unsigned int disk_query_batch_size,
                                             DiskQalshSearchOptions disk_qalsh_options)
    : approximation_ratio_(expected_approximation_ratio.value_or(Global::kDefaultApproximationRatio)),
      disk_based_(true),
      disk_query_batch_size_(disk_query_batch_size),
      disk_qalsh_options_(disk_qalsh_options),
      expected_approximation_ratio_(expected_approximation_ratio),
      ann_searcher_(std::make_unique<DiskBasedQalshAnnSearcher>(expected_approximation_ratio_,
                                                                 disk_qalsh_options_)) {}

bool QalshWeightsGenerator::RequiresIndex() const {
    return false;
}

QalshConfig QalshWeightsGenerator::BuildConfig(const PointSetMetadata& from_metadata,
                                               const PointSetMetadata& to_metadata, float norm_order,
                                               bool use_cache) const {
    if (disk_based_) {
        if (use_cache && HasValidQalshWeightsCache(from_metadata, norm_order)) {
            const DiskQalsh::Target target = DiskQalsh::ParseTargetFromPointSetPath(to_metadata.file_path);
            const std::filesystem::path index_directory =
                DiskQalsh::IndexDirectory(to_metadata.file_path.parent_path(), norm_order, target);
            if (std::filesystem::exists(index_directory)) {
                return DiskQalsh::ValidateIndex(to_metadata, norm_order, expected_approximation_ratio_).ToConfig();
            }

            QalshConfig config{.approximation_ratio = approximation_ratio_};
            config.Regularize(to_metadata.num_points, norm_order);
            return config;
        }

        return DiskQalsh::ValidateIndex(to_metadata, norm_order, expected_approximation_ratio_).ToConfig();
    }

    QalshConfig config{.approximation_ratio = approximation_ratio_};
    config.Regularize(to_metadata.num_points, norm_order);
    return config;
}

std::shared_ptr<const std::vector<Point>> QalshWeightsGenerator::TargetPoints() const {
    const auto* qalsh_searcher = dynamic_cast<const QalshAnnSearcher*>(ann_searcher_.get());
    if (qalsh_searcher == nullptr) {
        return {};
    }
    return qalsh_searcher->TargetPoints();
}

void QalshWeightsGenerator::PreflightGenerate(const PointSetMetadata& from_metadata,
                                              const PointSetMetadata& to_metadata, float norm_order,
                                              bool use_cache) const {
    if (!disk_based_) {
        return;
    }

    if (use_cache && HasValidQalshWeightsCache(from_metadata, norm_order)) {
        return;
    }

    const auto* disk_searcher = dynamic_cast<const DiskBasedQalshAnnSearcher*>(ann_searcher_.get());
    if (disk_searcher == nullptr) {
        Log::error("Disk QALSH weights generator is not configured with a disk QALSH searcher.");
    }
    disk_searcher->PreflightIndex(to_metadata, norm_order);
}

std::vector<Weight> QalshWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                    const PointSetMetadata& to_metadata, float norm_order,
                                                    bool use_cache) {
    std::vector<Weight> weights(from_metadata.num_points);

    if (use_cache) {
        Log::warn("The weights will be read from the cache file if it exists.");
        if (ReadQalshWeightsCacheIfAvailable(from_metadata, norm_order, weights)) {
            return weights;
        }
        Log::warn("Cache file doesn't exist. Generating the new one...");
    }

    // Generate weights based on QALSH algorithm.
    Log::info("Generating weights using QALSH...");

    ann_searcher_->Init(to_metadata, norm_order);

    if (disk_based_) {
        GenerateWeightsFromSearcherStreaming(*ann_searcher_, from_metadata, disk_query_batch_size_, weights);
        auto* disk_searcher = dynamic_cast<DiskBasedQalshAnnSearcher*>(ann_searcher_.get());
        if (disk_searcher != nullptr) {
            disk_searcher->PrintProfile();
        }
        if (use_cache) {
            WriteQalshWeightsCache(from_metadata, norm_order, weights);
        }
        return weights;
    }

#ifdef QALSH4C_USE_OPENMP
    std::vector<Point> query_points =
        PointSet::LoadPointsFromFile(from_metadata.file_path, from_metadata.num_points, from_metadata.num_dimensions);

#pragma omp parallel
    {
        std::unique_ptr<AnnSearcher> local_searcher = ann_searcher_->Clone();

#pragma omp for schedule(dynamic)
        for (long long query_index = 0; query_index < static_cast<long long>(from_metadata.num_points); ++query_index) {
            const auto point_id = static_cast<std::size_t>(query_index);
            weights[point_id] = local_searcher->Search(query_points[point_id]).distance;
        }
    }
#else
    std::vector<Point> query_points =
        PointSet::LoadPointsFromFile(from_metadata.file_path, from_metadata.num_points, from_metadata.num_dimensions);

    for (unsigned int i = 0; i < from_metadata.num_points; i++) {
        weights[i] = ann_searcher_->Search(query_points[i]).distance;
    }
#endif

    if (use_cache) {
        WriteQalshWeightsCache(from_metadata, norm_order, weights);
    }

    return weights;
}
