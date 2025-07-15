#include "types.h"

#include <spdlog/spdlog.h>
#include <toml++/toml.h>

#include <format>
#include <fstream>

#include "constants.h"
#include "utils.h"

// ---------------------------------------------
// DatasetMetadata Implementation
// ---------------------------------------------

void DatasetMetadata::Save(const std::filesystem::path& file_path) const {
    toml::table metadata;
    metadata.insert("data_type", data_type);
    metadata.insert("base_num_points", base_num_points);
    metadata.insert("query_num_points", query_num_points);
    metadata.insert("num_dimensions", num_dimensions);
    metadata.insert("chamfer_distance", chamfer_distance);

    std::ofstream metadata_ofs(file_path);
    if (!metadata_ofs.is_open()) {
        spdlog::critical("Failed to open metadata file for writing.");
    }
    metadata_ofs << metadata;
}

void DatasetMetadata::Load(const std::filesystem::path& file_path) {
    toml::table tbl = toml::parse_file(file_path.string());

    base_num_points = Utils::GetValueFromTomlTable<unsigned int>(tbl, "base_num_points");
    query_num_points = Utils::GetValueFromTomlTable<unsigned int>(tbl, "query_num_points");
    num_dimensions = Utils::GetValueFromTomlTable<unsigned int>(tbl, "num_dimensions");
    data_type = Utils::GetValueFromTomlTable<std::string>(tbl, "data_type");
    chamfer_distance = Utils::GetValueFromTomlTable<double>(tbl, "chamfer_distance");
}

std::string DatasetMetadata::Details() const {
    return std::format(
        "Data Type: {}\n"
        "Number of Points in the Base Set: {}\n"
        "Number of Points in the Query Set: {}\n"
        "Number of Dimensions: {}\n"
        "Chamfer Distance: {}",
        data_type, base_num_points, query_num_points, num_dimensions, chamfer_distance);
}

// ---------------------------------------------
// DatasetMetadata Implementation
// ---------------------------------------------

void QalshConfiguration::Save(const std::filesystem::path& file_path) const {
    toml::table config;
    config.insert("approximation_ratio", approximation_ratio);
    config.insert("bucket_width", bucket_width);
    config.insert("beta", beta);
    config.insert("error_probability", error_probability);
    config.insert("num_hash_tables", num_hash_tables);
    config.insert("collision_threshold", collision_threshold);
    config.insert("page_size", page_size);

    std::ofstream ofs(file_path);
    if (!ofs.is_open()) {
        spdlog::critical(std::format("Failed to open configuration file: {}", file_path.string()));
    }
    ofs << config;
}

void QalshConfiguration::Load(const std::filesystem::path& file_path) {
    toml::table tbl = toml::parse_file(file_path.string());

    approximation_ratio = Utils::GetValueFromTomlTable<double>(tbl, "approximation_ratio");
    bucket_width = Utils::GetValueFromTomlTable<double>(tbl, "bucket_width");
    beta = Utils::GetValueFromTomlTable<double>(tbl, "beta");
    error_probability = Utils::GetValueFromTomlTable<double>(tbl, "error_probability");
    num_hash_tables = Utils::GetValueFromTomlTable<unsigned int>(tbl, "num_hash_tables");
    collision_threshold = Utils::GetValueFromTomlTable<unsigned int>(tbl, "collision_threshold");
    page_size = Utils::GetValueFromTomlTable<unsigned int>(tbl, "page_size");
}

void QalshConfiguration::Regularize(unsigned int num_points) {
    if (bucket_width <= Constants::kEpsilon) {
        bucket_width = 2.0 * std::sqrt(approximation_ratio);
    }
    if (beta <= Constants::kEpsilon) {
        beta = 100.0 / static_cast<double>(num_points);
    }

    double term1 = std::sqrt(std::log(2.0 / beta));
    double term2 = std::sqrt(std::log(1.0 / error_probability));
    double p1 = 2.0 / std::numbers::pi_v<double> * atan(bucket_width / 2.0);
    double p2 = 2.0 / std::numbers::pi_v<double> * atan(bucket_width / (2.0 * approximation_ratio));

    if (num_hash_tables == 0) {
        double numerator = std::pow(term1 + term2, 2.0);
        double denominator = 2.0 * std::pow(p1 - p2, 2.0);
        num_hash_tables = static_cast<unsigned int>(std::ceil(numerator / denominator));
    }

    if (collision_threshold == 0) {
        double eta = term1 / term2;
        double alpha = (eta * p1 + p2) / (1 + eta);
        collision_threshold = static_cast<unsigned int>(std::ceil(alpha * num_hash_tables));
    }
}

std::string QalshConfiguration::Details() const {
    return std::format(
        "Approximation Ratio: {}\n"
        "Bucket Width: {}\n"
        "Beta: {}\n"
        "Error Probability: {}\n"
        "Number of Hash Tables: {}\n"
        "Collision Threshold: {}\n"
        "Page Size: {}",
        approximation_ratio, bucket_width, beta, error_probability, num_hash_tables, collision_threshold, page_size);
}
