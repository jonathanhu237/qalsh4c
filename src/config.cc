#include "config.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

#include "global.h"
#include "log.h"
#include "utils.h"

namespace {

std::optional<double> ReadOptionalDouble(const nlohmann::json& value) {
    if (value.is_null()) {
        return std::nullopt;
    }
    return value.get<double>();
}

void WriteOptionalDouble(nlohmann::json& metadata, const std::string& norm, const std::string& direction,
                         const std::optional<double>& value) {
    if (value.has_value()) {
        metadata["chamfer_distances"][norm][direction] = *value;
    } else {
        metadata["chamfer_distances"][norm][direction] = nullptr;
    }
}

std::optional<double> SumOptional(const std::optional<double>& left, const std::optional<double>& right) {
    if (!left.has_value() || !right.has_value()) {
        return std::nullopt;
    }
    return *left + *right;
}

}  // namespace

ChamferDirection ParseChamferDirection(const std::string &value) {
    if (value == "ab") {
        return ChamferDirection::kAb;
    }
    if (value == "ba") {
        return ChamferDirection::kBa;
    }
    if (value == "both") {
        return ChamferDirection::kBoth;
    }

    Log::error("Unsupported Chamfer direction: {}", value);
}

std::string ChamferDirectionValue(ChamferDirection direction) {
    switch (direction) {
        case ChamferDirection::kAb:
            return "ab";
        case ChamferDirection::kBa:
            return "ba";
        case ChamferDirection::kBoth:
            return "both";
    }

    Log::error("Unsupported Chamfer direction.");
}

std::string ChamferDirectionLabel(ChamferDirection direction) {
    switch (direction) {
        case ChamferDirection::kAb:
            return "A -> B";
        case ChamferDirection::kBa:
            return "B -> A";
        case ChamferDirection::kBoth:
            return "A <-> B";
    }

    Log::error("Unsupported Chamfer direction.");
}

void DatasetMetadata::Load(const std::filesystem::path &file_path) {
    if (!std::filesystem::exists(file_path)) {
        Log::error("The dataset metadata file does not exists, file path: {}", file_path.string());
    }

    std::ifstream ifs(file_path);
    nlohmann::json metadata = nlohmann::json::parse(ifs);

    try {
        metadata.at("num_points").at("a").get_to(num_points_a);
        metadata.at("num_points").at("b").get_to(num_points_b);
        metadata.at("num_dimensions").get_to(num_dimensions);
        chamfer_distance_l1_ab = ReadOptionalDouble(metadata.at("chamfer_distances").at("l1").at("ab"));
        chamfer_distance_l1_ba = ReadOptionalDouble(metadata.at("chamfer_distances").at("l1").at("ba"));
        chamfer_distance_l2_ab = ReadOptionalDouble(metadata.at("chamfer_distances").at("l2").at("ab"));
        chamfer_distance_l2_ba = ReadOptionalDouble(metadata.at("chamfer_distances").at("l2").at("ba"));
    } catch (nlohmann::json::exception &e) {
        Log::error("JSON format error: {}", e.what());
    }
}

void DatasetMetadata::Save(const std::filesystem::path &file_path) const {
    nlohmann::json metadata;
    metadata["num_dimensions"] = num_dimensions;
    metadata["num_points"]["a"] = num_points_a;
    metadata["num_points"]["b"] = num_points_b;
    WriteOptionalDouble(metadata, "l1", "ab", chamfer_distance_l1_ab);
    WriteOptionalDouble(metadata, "l1", "ba", chamfer_distance_l1_ba);
    WriteOptionalDouble(metadata, "l2", "ab", chamfer_distance_l2_ab);
    WriteOptionalDouble(metadata, "l2", "ba", chamfer_distance_l2_ba);

    std::ofstream ofs(file_path);
    if (!ofs.is_open()) {
        Log::error("Failed to open file for writing: {}", file_path.string());
    }
    ofs << metadata.dump(4);
}

std::optional<double> DatasetMetadata::GroundTruth(double norm_order, ChamferDirection direction) const {
    std::optional<double> distance_ab;
    std::optional<double> distance_ba;

    if (std::abs(norm_order - 1.0) < Global::kEpsilon) {
        distance_ab = chamfer_distance_l1_ab;
        distance_ba = chamfer_distance_l1_ba;
    } else if (std::abs(norm_order - 2.0) < Global::kEpsilon) {
        distance_ab = chamfer_distance_l2_ab;
        distance_ba = chamfer_distance_l2_ba;
    } else {
        Log::error("Unsupported norm order: {}", norm_order);
    }

    switch (direction) {
        case ChamferDirection::kAb:
            return distance_ab;
        case ChamferDirection::kBa:
            return distance_ba;
        case ChamferDirection::kBoth:
            return SumOptional(distance_ab, distance_ba);
    }

    Log::error("Unsupported Chamfer direction.");
}

void QalshConfig::Regularize(unsigned int num_points, double norm_order) {
    double beta = std::min(Global::kNumCandidates / static_cast<double>(num_points), 1.0);
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
        Log::error("Unsupported norm order: {}", norm_order);
    }

    double denominator = 2.0 * std::pow(p1 - p2, 2.0);
    num_hash_tables = static_cast<unsigned int>(std::ceil(numerator / denominator));

    double alpha = (eta * p1 + p2) / (1 + eta);
    collision_threshold = static_cast<unsigned int>(std::ceil(alpha * num_hash_tables));
}
