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
                                                      [[maybe_unused]] bool in_memory,
                                                      [[maybe_unused]] bool use_cache) {
    std::vector<double> weights(from_metadata.num_points, 1.0);
    return weights;
}

// --------------------------------------------------
// QalshWeightsGenerator Implementation
// --------------------------------------------------
QalshWeightsGenerator::QalshWeightsGenerator(double approximation_ratio) : approximation_ratio_(approximation_ratio) {}

std::vector<double> QalshWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                    const PointSetMetadata& to_metadata, bool in_memory,
                                                    bool use_cache) {
    std::vector<double> weights(from_metadata.num_points);
    std::filesystem::path parent_directory = from_metadata.file_path.parent_path();
    std::string stem = from_metadata.file_path.stem();
    std::filesystem::path weights_path = parent_directory / std::format("qalsh_weights_{}", stem);

    if (use_cache) {
        if (std::filesystem::exists(weights_path)) {
            spdlog::info("The cache file exist, reading the weights from the file.");
            std::ifstream ifs(weights_path);
            ifs.read(reinterpret_cast<char*>(weights.data()),
                     static_cast<std::streamoff>(sizeof(double) * from_metadata.num_points));
            return weights;
        }
        spdlog::warn("Cache file doesn't exist. Generating the new one...");
    }

    // Generate weights based on QALSH algorithm.
    spdlog::info("Generating weights using QALSH...");
    QalshAnnSearcher qalsh_ann_searcher(approximation_ratio_);
    qalsh_ann_searcher.Init(to_metadata, in_memory);

    std::unique_ptr<PointSet> from_set;
    if (in_memory) {
        from_set = std::make_unique<InMemoryPointSet>(from_metadata);
    } else {
        from_set = std::make_unique<DiskPointSet>(from_metadata);
    }

    for (unsigned int i = 0; i < from_metadata.num_points; i++) {
        Point query_point = from_set->GetPoint(i);
        AnnResult result = qalsh_ann_searcher.Search(query_point);
        weights[i] = result.distance;  // Use the distance as the weight
    }

    if (use_cache) {
        std::ofstream ofs(weights_path);
        ofs.write(reinterpret_cast<const char*>(weights.data()),
                  static_cast<std::streamoff>(sizeof(double) * from_metadata.num_points));
    }

    return weights;
}