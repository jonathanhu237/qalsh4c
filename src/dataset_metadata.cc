#include "dataset_metadata.h"

#include <spdlog/spdlog.h>
#include <toml++/toml.h>

#include <fstream>

DatasetMetadata::DatasetMetadata(unsigned int base_num_points, unsigned int query_num_points,
                                 unsigned int num_dimensions, std::string data_type)
    : base_num_points_(base_num_points),
      query_num_points_(query_num_points),
      num_dimensions_(num_dimensions),
      data_type_(std::move(data_type)) {}

DatasetMetadata::DatasetMetadata(const std::filesystem::path& file_path) {
    toml::table tbl = toml::parse_file(file_path.string());

    base_num_points_ = tbl["base_num_points"].value_or(0U);
    if (base_num_points_ == 0) {
        throw std::runtime_error("base_num_points is not specified in metadata.toml");
    }

    query_num_points_ = tbl["query_num_points"].value_or(0U);
    if (query_num_points_ == 0) {
        throw std::runtime_error("query_num_points is not specified in metadata.toml");
    }

    num_dimensions_ = tbl["num_dimensions"].value_or(0U);
    if (num_dimensions_ == 0) {
        throw std::runtime_error("num_dimensions is not specified in metadata.toml");
    }

    data_type_ = tbl["data_type"].value_or("");
    if (data_type_.empty()) {
        throw std::runtime_error("data_type is not specified in metadata.toml");
    }
}

auto DatasetMetadata::Save(const std::filesystem::path& file_path) const -> void {
    toml::table metadata;
    metadata.insert("data_type", data_type_);
    metadata.insert("base_num_points", base_num_points_);
    metadata.insert("query_num_points", query_num_points_);
    metadata.insert("num_dimensions", num_dimensions_);

    std::ofstream metadata_ofs;
    metadata_ofs.open(file_path);
    if (!metadata_ofs.is_open()) {
        throw std::runtime_error("Failed to open metadata file for writing.");
    }
    metadata_ofs << metadata;
    spdlog::info("Metadata saved to {}", file_path.string());
}