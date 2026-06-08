#include "weights_generator.h"

#include <cstddef>
#include <memory>
#include <vector>

#include "ann_searcher.h"
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
QalshWeightsGenerator::QalshWeightsGenerator() : ann_searcher_(std::make_unique<QalshAnnSearcher>()) {}

bool QalshWeightsGenerator::RequiresIndex() const {
    return true;
}

std::vector<double> QalshWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                    const PointSetMetadata& to_metadata, double norm_order,
                                                    bool use_cache) {
    std::vector<double> weights(from_metadata.num_points);
    std::filesystem::path parent_directory = from_metadata.file_path.parent_path();
    std::string stem = from_metadata.file_path.stem();
    std::filesystem::path weights_path = parent_directory / std::format("l{}_qalsh_weights_{}.bin", norm_order, stem);

    if (use_cache) {
        Log::warn("The weights will be read from the cache file if it exists.");
        if (std::filesystem::exists(weights_path)) {
            std::ifstream ifs(weights_path);
            ifs.read(reinterpret_cast<char*>(weights.data()),
                     static_cast<std::streamoff>(sizeof(double) * from_metadata.num_points));
            return weights;
        }
        Log::warn("Cache file doesn't exist. Generating the new one...");
    }

    // Generate weights based on QALSH algorithm.
    Log::info("Generating weights using QALSH...");

#ifdef QALSH4C_USE_OPENMP
    std::vector<Point> query_points =
        PointSet::LoadPointsFromFile(from_metadata.file_path, from_metadata.num_points, from_metadata.num_dimensions);

#pragma omp parallel
    {
        std::unique_ptr<AnnSearcher> local_searcher = ann_searcher_->Clone();
        local_searcher->Init(to_metadata, norm_order);

#pragma omp for schedule(dynamic)
        for (long long query_index = 0; query_index < static_cast<long long>(from_metadata.num_points); ++query_index) {
            const auto point_id = static_cast<std::size_t>(query_index);
            weights[point_id] = local_searcher->Search(query_points[point_id]).distance;
        }
    }
#else
    ann_searcher_->Init(to_metadata, norm_order);
    PointSet base;
    base.Init(from_metadata);

    for (unsigned int i = 0; i < from_metadata.num_points; i++) {
        weights[i] = ann_searcher_->Search(base.ReadPoint(i)).distance;
    }
#endif

    if (use_cache) {
        std::ofstream ofs(weights_path);
        ofs.write(reinterpret_cast<const char*>(weights.data()),
                  static_cast<std::streamoff>(sizeof(double) * from_metadata.num_points));
    }

    return weights;
}
