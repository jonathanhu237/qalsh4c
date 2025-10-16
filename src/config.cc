#include "config.h"

#include <fstream>
#include <nlohmann/json.hpp>

#include "global.h"
#include "utils.h"

void DatasetMetadata::Load(const std::filesystem::path &file_path) {
    if (!std::filesystem::exists(file_path)) {
        spdlog::error("The dataset metadata file does not exists, file path: {}", file_path.string());
    }

    std::ifstream ifs(file_path);
    nlohmann::json metadata = nlohmann::json::parse(ifs);

    try {
        metadata.at("num_points_a").get_to(num_points_a);
        metadata.at("num_points_b").get_to(num_points_b);
        metadata.at("num_dimensions").get_to(num_dimensions);
        metadata.at("chamfer_distance_l1").get_to(chamfer_distance_l1);
        metadata.at("chamfer_distance_l2").get_to(chamfer_distance_l2);
    } catch (nlohmann::json::exception &e) {
        spdlog::error("JSON format error: {}", e.what());
    }
}

void QalshConfig::Regularize(unsigned int num_points, double norm_order) {
    double beta = Global::kNumCandidates / static_cast<double>(num_points);
    error_probability = Global::kQalshDefaultErrorProbability;
    double term1 = std::sqrt(std::log(2.0 / beta));
    double term2 = std::sqrt(std::log(1.0 / error_probability));
    double numerator = std::pow(term1 + term2, 2.0);
    double eta = term1 / term2;

    double p1{0.0};
    double p2{0.0};

    if (std::abs(norm_order - 1.0) < Global::kEpsilon) {
        bucket_width = 2.0 * std::sqrt(approximation_ratio);
        p1 = Calc::L1Probability(bucket_width / 2.0);
        p2 = Calc::L1Probability(bucket_width / (2.0 * approximation_ratio));
    } else if (std::abs(norm_order - 2.0) < Global::kEpsilon) {
        bucket_width = std::sqrt((8.0 * std::pow(approximation_ratio, 2.0) * std::log(approximation_ratio)) /
                                 (std::pow(approximation_ratio, 2.0) - 1));
        p1 = Calc::L2Probability(bucket_width / 2.0);
        p2 = Calc::L2Probability(bucket_width / (2.0 * approximation_ratio));
    } else {
        spdlog::error("Unsupported norm order: {}", norm_order);
    }

    double denominator = 2.0 * std::pow(p1 - p2, 2.0);
    num_hash_tables = static_cast<unsigned int>(std::ceil(numerator / denominator));

    double alpha = (eta * p1 + p2) / (1 + eta);
    collision_threshold = static_cast<unsigned int>(std::ceil(alpha * num_hash_tables));
}

void QalshConfig::Save(const std::filesystem::path &file_path) {
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

void QalshConfig::Load(const std::filesystem::path &file_path) {
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
    } catch (nlohmann::json::exception &e) {
        spdlog::error("JSON format error: {}", e.what());
    }
}