#include "weights_generator.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <vector>

#include "ann_searcher.h"
#include "utils.h"

// --------------------------------------------------
// UniformWeightsGenerator Implementation
// --------------------------------------------------
std::vector<double> UniformWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                      [[maybe_unused]] const PointSetMetadata& to_metadata,
                                                      [[maybe_unused]] bool use_cache) {
    std::vector<double> weights(from_metadata.num_points, 1.0);
    return weights;
}

// --------------------------------------------------
// InMemoryQalshWeightsGenerator Implementation
// --------------------------------------------------
InMemoryQalshWeightsGenerator::InMemoryQalshWeightsGenerator(double approximation_ratio, double error_probability)
    : approximation_ratio_(approximation_ratio), error_probability_(error_probability) {}

std::vector<double> InMemoryQalshWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                            const PointSetMetadata& to_metadata, bool use_cache) {
    std::vector<double> weights(from_metadata.num_points);
    std::filesystem::path parent_directory = from_metadata.file_path.parent_path();
    std::string stem = from_metadata.file_path.stem();
    std::filesystem::path weights_path = parent_directory / std::format("qalsh_weights_{}.bin", stem);

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
    spdlog::info("Generating weights using QALSH (In Memory)...");

    std::unique_ptr<AnnSearcher> ann_searcher =
        std::make_unique<InMemoryQalshAnnSearcher>(approximation_ratio_, error_probability_);
    ann_searcher->Init(to_metadata);

    std::vector<Point> base_points =
        Utils::LoadPointsFromFile(from_metadata.file_path, from_metadata.num_points, from_metadata.num_dimensions);

    for (unsigned int i = 0; i < from_metadata.num_points; i++) {
        weights[i] = ann_searcher->Search(base_points[i]).distance;
    }

    if (use_cache) {
        std::ofstream ofs(weights_path);
        ofs.write(reinterpret_cast<const char*>(weights.data()),
                  static_cast<std::streamoff>(sizeof(double) * from_metadata.num_points));
    }

    return weights;
}

// --------------------------------------------------
// DiskQalshWeightsGenerator Implementation
// --------------------------------------------------
std::vector<double> DiskQalshWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                        const PointSetMetadata& to_metadata, bool use_cache) {
    std::vector<double> weights(from_metadata.num_points);
    std::filesystem::path parent_directory = from_metadata.file_path.parent_path();
    std::string stem = from_metadata.file_path.stem();
    std::filesystem::path weights_path = parent_directory / std::format("qalsh_weights_{}.bin", stem);

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

    std::unique_ptr<AnnSearcher> ann_searcher = std::make_unique<DiskQalshAnnSearcher>();
    ann_searcher->Init(to_metadata);

    std::ifstream base_file(from_metadata.file_path, std::ios::binary);
    if (!base_file.is_open()) {
        spdlog::error("Failed to open base file: {}", from_metadata.file_path.string());
        return {};
    }

    for (unsigned int i = 0; i < from_metadata.num_points; i++) {
        weights[i] = ann_searcher->Search(Utils::ReadPoint(base_file, from_metadata.num_dimensions, i)).distance;
    }

    if (use_cache) {
        std::ofstream ofs(weights_path);
        ofs.write(reinterpret_cast<const char*>(weights.data()),
                  static_cast<std::streamoff>(sizeof(double) * from_metadata.num_points));
    }

    return weights;
}