#include "qalsh_config.h"

#include <spdlog/spdlog.h>

#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numbers>

#include "global.h"

// NOLINTBEGIN: readability-magic-numbers
void QalshConfig::Regularize(unsigned int num_points) {
    bucket_width = 2.0 * std::sqrt(approximation_ratio);
    double beta = Global::kNumCandidates / static_cast<double>(num_points);
    error_probability = Global::kDefaultErrorProbability;

    double term1 = std::sqrt(std::log(2.0 / beta));
    double term2 = std::sqrt(std::log(1.0 / error_probability));
    double p1 = 2.0 / std::numbers::pi_v<double> * atan(bucket_width / 2.0);
    double p2 = 2.0 / std::numbers::pi_v<double> * atan(bucket_width / (2.0 * approximation_ratio));
    double numerator = std::pow(term1 + term2, 2.0);
    double denominator = 2.0 * std::pow(p1 - p2, 2.0);
    num_hash_tables = static_cast<unsigned int>(std::ceil(numerator / denominator));

    double eta = term1 / term2;
    double alpha = (eta * p1 + p2) / (1 + eta);
    collision_threshold = static_cast<unsigned int>(std::ceil(alpha * num_hash_tables));
}
// NOLINTEND

void QalshConfig::Save(const std::filesystem::path& file_path) {
    nlohmann::json metadata;
    metadata["approximation_ratio"] = approximation_ratio;
    metadata["bucket_width"] = bucket_width;
    metadata["error_probability"] = error_probability;
    metadata["num_hash_tables"] = num_hash_tables;
    metadata["collision_threshold"] = collision_threshold;
    metadata["page_size"] = page_size;

    std::ofstream ofs(file_path);
    if (!ofs.is_open()) {
        spdlog::error("Failed to open file for writing: {}", file_path.string());
        return;
    }

    ofs << metadata.dump(4);
}

void QalshConfig::Load(const std::filesystem::path& file_path) {
    if (!std::filesystem::exists(file_path)) {
        spdlog::error("The QALSH config file does not exists, file path: {}", file_path.string());
    }

    std::ifstream ifs(file_path);
    nlohmann::json metadata = nlohmann::json::parse(ifs);

    try {
        metadata.at("approximation_ratio").get_to(approximation_ratio);
        metadata.at("bucket_width").get_to(bucket_width);
        metadata.at("error_probability").get_to(error_probability);
        metadata.at("num_hash_tables").get_to(num_hash_tables);
        metadata.at("collision_threshold").get_to(collision_threshold);
        metadata.at("page_size").get_to(page_size);
    } catch (nlohmann::json::exception& e) {
        spdlog::error("JSON format error: {}", e.what());
    }
}