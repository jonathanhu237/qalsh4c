#include "weights_generator.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <vector>

#include "ann_searcher.h"
#include "point_set.h"

// --------------------------------------------------
// UniformWeightsGenerator Implementation
// --------------------------------------------------
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

std::vector<double> QalshWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                    const PointSetMetadata& to_metadata, double norm_order,
                                                    bool use_cache) {
    std::vector<double> weights(from_metadata.num_points);
    std::filesystem::path parent_directory = from_metadata.file_path.parent_path();
    std::string stem = from_metadata.file_path.stem();
    std::filesystem::path weights_path = parent_directory / std::format("l{}_qalsh_weights_{}.bin", norm_order, stem);

    if (use_cache) {
        spdlog::warn("The weights will be read from the cache file if it exists.");
        if (std::filesystem::exists(weights_path)) {
            std::ifstream ifs(weights_path);
            ifs.read(reinterpret_cast<char*>(weights.data()),
                     static_cast<std::streamoff>(sizeof(double) * from_metadata.num_points));
            return weights;
        }
        spdlog::warn("Cache file doesn't exist. Generating the new one...");
    }

    // Generate weights based on QALSH algorithm.
    spdlog::info("Generating weights using QALSH (Disk)...");
    ann_searcher_->Init(to_metadata.file_path, to_metadata.num_points, to_metadata.num_dimensions, norm_order);

    PointSet base;
    base.Init(from_metadata.file_path, from_metadata.num_points, from_metadata.num_dimensions);

    for (unsigned int i = 0; i < from_metadata.num_points; i++) {
        weights[i] = ann_searcher_->Search(base.ReadPoint(i)).distance;
    }

    if (use_cache) {
        std::ofstream ofs(weights_path);
        ofs.write(reinterpret_cast<const char*>(weights.data()),
                  static_cast<std::streamoff>(sizeof(double) * from_metadata.num_points));
    }

    return weights;
}