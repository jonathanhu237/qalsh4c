#include "weights_generator.h"

#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "ann_searcher.h"
#include "config.h"
#include "log.h"
#include "point_set.h"

// --------------------------------------------------
// UniformWeightsGenerator Implementation
// --------------------------------------------------
bool UniformWeightsGenerator::RequiresIndex() const {
    return false;
}

std::vector<double> UniformWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                      [[maybe_unused]] const PointSetMetadata& to_metadata,
                                                      [[maybe_unused]] double norm_order,
                                                      [[maybe_unused]] bool use_cache) {
    std::vector<double> weights(from_metadata.num_points, 1.0);
    return weights;
}

// --------------------------------------------------
// QalshWeightsGenerator Implementation
// --------------------------------------------------
QalshWeightsGenerator::QalshWeightsGenerator(double approximation_ratio)
    : approximation_ratio_(approximation_ratio),
      ann_searcher_(std::make_unique<QalshAnnSearcher>(approximation_ratio_)) {}

bool QalshWeightsGenerator::RequiresIndex() const {
    return false;
}

QalshConfig QalshWeightsGenerator::BuildConfig(unsigned int num_points, double norm_order) const {
    QalshConfig config{.approximation_ratio = approximation_ratio_};
    config.Regularize(num_points, norm_order);
    return config;
}

std::shared_ptr<const std::vector<Point>> QalshWeightsGenerator::TargetPoints() const {
    const auto* qalsh_searcher = dynamic_cast<const QalshAnnSearcher*>(ann_searcher_.get());
    if (qalsh_searcher == nullptr) {
        return {};
    }
    return qalsh_searcher->TargetPoints();
}

std::vector<double> QalshWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                    const PointSetMetadata& to_metadata, double norm_order,
                                                    bool use_cache) {
    std::vector<double> weights(from_metadata.num_points);
    std::filesystem::path parent_directory = from_metadata.file_path.parent_path();
    std::string stem = from_metadata.file_path.stem();
    std::filesystem::path weights_path = parent_directory / std::format("l{}_qalsh_weights_{}.bin", norm_order, stem);
    const auto expected_bytes =
        static_cast<std::streamsize>(sizeof(double) * static_cast<size_t>(from_metadata.num_points));

    if (use_cache) {
        Log::warn("The weights will be read from the cache file if it exists.");
        if (std::filesystem::exists(weights_path)) {
            std::ifstream ifs(weights_path, std::ios::binary);
            if (!ifs.is_open()) {
                Log::error("Failed to open QALSH weights cache file: {}", weights_path.string());
            }
            ifs.read(reinterpret_cast<char*>(weights.data()), expected_bytes);
            if (ifs.gcount() != expected_bytes) {
                Log::error("QALSH weights cache file is incomplete: expected {} bytes, read {} bytes from {}",
                           expected_bytes, ifs.gcount(), weights_path.string());
            }
            return weights;
        }
        Log::warn("Cache file doesn't exist. Generating the new one...");
    }

    // Generate weights based on QALSH algorithm.
    Log::info("Generating weights using QALSH...");

    ann_searcher_->Init(to_metadata, norm_order);

#ifdef QALSH4C_USE_OPENMP
    std::vector<Point> query_points =
        PointSet::LoadPointsFromFile(from_metadata.file_path, from_metadata.num_points, from_metadata.num_dimensions);

#pragma omp parallel
    {
        std::unique_ptr<AnnSearcher> local_searcher = ann_searcher_->Clone();

#pragma omp for schedule(static)
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
        std::ofstream ofs(weights_path, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) {
            Log::error("Failed to open QALSH weights cache file for writing: {}", weights_path.string());
        }
        ofs.write(reinterpret_cast<const char*>(weights.data()), expected_bytes);
    }

    return weights;
}
