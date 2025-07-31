#include "dataset_metadata.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

void DatasetMetadata::Load(const std::filesystem::path& file_path) {
    if (!std::filesystem::exists(file_path)) {
        spdlog::error("The dataset metadata file does not exists, file path: {}", file_path.string());
    }

    std::ifstream ifs(file_path);
    nlohmann::json metadata = nlohmann::json::parse(ifs);

    try {
        metadata.at("num_points_a").get_to(num_points_a);
        metadata.at("num_points_b").get_to(num_points_b);
        metadata.at("num_dimensions").get_to(num_dimensions);
        metadata.at("chamfer_distance").get_to(chamfer_distance);
    } catch (nlohmann::json::exception& e) {
        spdlog::error("JSON format error: {}", e.what());
    }
}

void DatasetMetadata::Save(const std::filesystem::path& file_path) const {
    nlohmann::json metadata;
    metadata["num_points_a"] = num_points_a;
    metadata["num_points_b"] = num_points_b;
    metadata["num_dimensions"] = num_dimensions;
    metadata["chamfer_distance"] = chamfer_distance;

    std::ofstream ofs(file_path);
    if (!ofs.is_open()) {
        spdlog::error("Failed to open file for writing: {}", file_path.string());
        return;
    }

    ofs << metadata.dump(4);
}