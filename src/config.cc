#include "config.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

#include "global.h"
#include "log.h"
#include "utils.h"

namespace {

std::optional<float> ReadOptionalFloat(const nlohmann::json& value) {
    if (value.is_null()) {
        return std::nullopt;
    }
    return value.get<float>();
}

void WriteOptionalFloat(nlohmann::json& metadata, const std::string& norm, const std::string& direction,
                         const std::optional<float>& value) {
    if (value.has_value()) {
        metadata["chamfer_distances"][norm][direction] = *value;
    } else {
        metadata["chamfer_distances"][norm][direction] = nullptr;
    }
}

std::optional<float> SumOptional(const std::optional<float>& left, const std::optional<float>& right) {
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
        chamfer_distance_l1_ab = ReadOptionalFloat(metadata.at("chamfer_distances").at("l1").at("ab"));
        chamfer_distance_l1_ba = ReadOptionalFloat(metadata.at("chamfer_distances").at("l1").at("ba"));
        chamfer_distance_l2_ab = ReadOptionalFloat(metadata.at("chamfer_distances").at("l2").at("ab"));
        chamfer_distance_l2_ba = ReadOptionalFloat(metadata.at("chamfer_distances").at("l2").at("ba"));
    } catch (nlohmann::json::exception &e) {
        Log::error("JSON format error: {}", e.what());
    }
}

void DatasetMetadata::Save(const std::filesystem::path &file_path) const {
    nlohmann::json metadata;
    metadata["num_dimensions"] = num_dimensions;
    metadata["num_points"]["a"] = num_points_a;
    metadata["num_points"]["b"] = num_points_b;
    WriteOptionalFloat(metadata, "l1", "ab", chamfer_distance_l1_ab);
    WriteOptionalFloat(metadata, "l1", "ba", chamfer_distance_l1_ba);
    WriteOptionalFloat(metadata, "l2", "ab", chamfer_distance_l2_ab);
    WriteOptionalFloat(metadata, "l2", "ba", chamfer_distance_l2_ba);

    std::ofstream ofs(file_path);
    if (!ofs.is_open()) {
        Log::error("Failed to open file for writing: {}", file_path.string());
    }
    ofs << metadata.dump(4);
}

std::optional<float> DatasetMetadata::GroundTruth(float norm_order, ChamferDirection direction) const {
    std::optional<float> distance_ab;
    std::optional<float> distance_ba;

    if (std::abs(norm_order - 1.0F) < Global::kEpsilon) {
        distance_ab = chamfer_distance_l1_ab;
        distance_ba = chamfer_distance_l1_ba;
    } else if (std::abs(norm_order - 2.0F) < Global::kEpsilon) {
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

void QalshConfig::Regularize(unsigned int num_points, float norm_order) {
    // QALSH's candidate-budget fraction; QALSH-CrudeNN uses beta = 1 unless --qalsh-beta says otherwise.
    float beta = Global::kQalshBeta > 0.0F ? std::min(Global::kQalshBeta, 1.0F)
                                           : std::min(Global::kNumCandidates / static_cast<float>(num_points), 1.0F);
    error_probability = Global::kQalshDefaultErrorProbability;
    float term1 = std::sqrt(std::log(2.0F / beta));
    float term2 = std::sqrt(std::log(1.0F / error_probability));
    float numerator = std::pow(term1 + term2, 2.0F);
    float eta = term1 / term2;

    float p1{0.0F};
    float p2{0.0F};

    if (std::abs(norm_order - 1.0F) < Global::kEpsilon) {
        bucket_width = 2.0F * std::sqrt(approximation_ratio);
        p1 = Calc::L1Probability(bucket_width / 2.0F);
        p2 = Calc::L1Probability(bucket_width / (2.0F * approximation_ratio));
    } else if (std::abs(norm_order - 2.0F) < Global::kEpsilon) {
        bucket_width = std::sqrt((8.0F * std::pow(approximation_ratio, 2.0F) * std::log(approximation_ratio)) /
                                 (std::pow(approximation_ratio, 2.0F) - 1.0F));
        p1 = Calc::L2Probability(bucket_width / 2.0F);
        p2 = Calc::L2Probability(bucket_width / (2.0F * approximation_ratio));
    } else {
        Log::error("Unsupported norm order: {}", norm_order);
    }
    if (Global::kQalshBucketWidth > 0.0F) {
        bucket_width = Global::kQalshBucketWidth;
        if (std::abs(norm_order - 1.0F) < Global::kEpsilon) {
            p1 = Calc::L1Probability(bucket_width / 2.0F);
            p2 = Calc::L1Probability(bucket_width / (2.0F * approximation_ratio));
        } else {
            p1 = Calc::L2Probability(bucket_width / 2.0F);
            p2 = Calc::L2Probability(bucket_width / (2.0F * approximation_ratio));
        }
    }

    float denominator = 2.0F * std::pow(p1 - p2, 2.0F);
    num_hash_tables = static_cast<unsigned int>(std::ceil(numerator / denominator));

    float alpha = (eta * p1 + p2) / (1.0F + eta);
    collision_threshold = static_cast<unsigned int>(std::ceil(alpha * static_cast<float>(num_hash_tables)));

    if (Global::kQalshTables > 0) {
        num_hash_tables = Global::kQalshTables;
        collision_threshold = static_cast<unsigned int>(std::ceil(alpha * static_cast<float>(num_hash_tables)));
    }
    if (Global::kQalshThreshold > 0) {
        collision_threshold = Global::kQalshThreshold;
    }
    if (collision_threshold == 0 || collision_threshold > num_hash_tables) {
        Log::error("Invalid QALSH parameters: m = {}, l = {}", num_hash_tables, collision_threshold);
    }
}
