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

    std::ofstream metadata_ofs(file_path);
    if (!metadata_ofs.is_open()) {
        throw std::runtime_error("Failed to open metadata file for writing.");
    }
    metadata_ofs << metadata;
}

auto DatasetMetadata::Load(const std::filesystem::path& file_path) -> void {
    toml::table tbl = toml::parse_file(file_path.string());

    base_num_points_ = Utils::GetValueFromTomlTable<unsigned int>(tbl, "base_num_points");
    query_num_points_ = Utils::GetValueFromTomlTable<unsigned int>(tbl, "query_num_points");
    num_dimensions_ = Utils::GetValueFromTomlTable<unsigned int>(tbl, "num_dimensions");
    data_type_ = Utils::GetValueFromTomlTable<std::string>(tbl, "data_type");
    chamfer_distance_ = Utils::GetValueFromTomlTable<double>(tbl, "chamfer_distance");
}

auto QalshConfiguration::Save(const std::filesystem::path& file_path) const -> void {
    toml::table config;
    config.insert("approximation_ratio", approximation_ratio_);
    config.insert("bucket_width", bucket_width_);
    config.insert("beta", beta_);
    config.insert("error_probability", error_probability_);
    config.insert("num_hash_tables", num_hash_tables_);
    config.insert("collision_threshold", collision_threshold_);
    config.insert("page_size", page_size_);

    std::ofstream ofs(file_path);
    if (!ofs.is_open()) {
        throw std::runtime_error(std::format("Failed to open configuration file: {}", file_path.string()));
    }
    ofs << config;
}

auto QalshConfiguration::Load(const std::filesystem::path& file_path) -> void {
    toml::table tbl = toml::parse_file(file_path.string());

    approximation_ratio_ = Utils::GetValueFromTomlTable<double>(tbl, "approximation_ratio");
    bucket_width_ = Utils::GetValueFromTomlTable<double>(tbl, "bucket_width");
    beta_ = Utils::GetValueFromTomlTable<double>(tbl, "beta");
    error_probability_ = Utils::GetValueFromTomlTable<double>(tbl, "error_probability");
    num_hash_tables_ = Utils::GetValueFromTomlTable<unsigned int>(tbl, "num_hash_tables");
    collision_threshold_ = Utils::GetValueFromTomlTable<unsigned int>(tbl, "collision_threshold");
    page_size_ = Utils::GetValueFromTomlTable<unsigned int>(tbl, "page_size");
}