#include "types.h"

#include <spdlog/spdlog.h>
#include <toml++/toml.h>

#include <fstream>

#include "utils.h"

auto DatasetMetadata::Save(const std::filesystem::path& file_path) const -> void {
    toml::table metadata;
    metadata.insert("data_type", data_type_);
    metadata.insert("base_num_points", base_num_points_);
    metadata.insert("query_num_points", query_num_points_);
    metadata.insert("num_dimensions", num_dimensions_);
    metadata.insert("chamfer_distance", chamfer_distance_);

    std::ofstream metadata_ofs;
    metadata_ofs.open(file_path);
    if (!metadata_ofs.is_open()) {
        throw std::runtime_error("Failed to open metadata file for writing.");
    }
    metadata_ofs << metadata;
    spdlog::info("Metadata saved to {}", file_path.string());
}

auto DatasetMetadata::Load(const std::filesystem::path& file_path) -> void {
    toml::table tbl = toml::parse_file(file_path.string());

    base_num_points_ = Utils::GetValueFromTomlTable<unsigned int>(tbl, "base_num_points");
    query_num_points_ = Utils::GetValueFromTomlTable<unsigned int>(tbl, "query_num_points");
    num_dimensions_ = Utils::GetValueFromTomlTable<unsigned int>(tbl, "num_dimensions");
    data_type_ = Utils::GetValueFromTomlTable<std::string>(tbl, "data_type");
    chamfer_distance_ = Utils::GetValueFromTomlTable<double>(tbl, "chamfer_distance");
}