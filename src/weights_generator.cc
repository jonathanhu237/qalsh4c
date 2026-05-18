#include "weights_generator.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ann_searcher.h"
#include "global.h"
#include "log.h"
#include "point_set.h"

namespace {

constexpr std::string_view kWeightsFormat = "qalsh4c-qalsh-weights-v1";

std::string DirectionValue(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata) {
    const std::string from_name = from_metadata.file_path.stem().string();
    const std::string to_name = to_metadata.file_path.stem().string();
    if (from_name == "A" && to_name == "B") {
        return "ab";
    }
    if (from_name == "B" && to_name == "A") {
        return "ba";
    }
    return std::format("{}-to-{}", from_name, to_name);
}

int64_t LastWriteTimeCount(const std::filesystem::path& path) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::filesystem::last_write_time(path).time_since_epoch())
        .count();
}

nlohmann::json PointSetIdentity(const PointSetMetadata& metadata) {
    return nlohmann::json{
        {"stem", metadata.file_path.stem().string()},
        {"path", std::filesystem::absolute(metadata.file_path).string()},
        {"num_points", metadata.num_points},
        {"num_dimensions", metadata.num_dimensions},
        {"file_size", std::filesystem::file_size(metadata.file_path)},
        {"mtime", LastWriteTimeCount(metadata.file_path)},
    };
}

nlohmann::json QalshConfigJson(const QalshConfig& config) {
    return nlohmann::json{
        {"approximation_ratio", config.approximation_ratio},
        {"bucket_width", config.bucket_width},
        {"error_probability", config.error_probability},
        {"num_hash_tables", config.num_hash_tables},
        {"collision_threshold", config.collision_threshold},
        {"page_size", config.page_size},
    };
}

QalshConfig ReadQalshConfig(const nlohmann::json& metadata) {
    QalshConfig config;
    const nlohmann::json& value = metadata.at("qalsh_config");
    value.at("approximation_ratio").get_to(config.approximation_ratio);
    value.at("bucket_width").get_to(config.bucket_width);
    value.at("error_probability").get_to(config.error_probability);
    value.at("num_hash_tables").get_to(config.num_hash_tables);
    value.at("collision_threshold").get_to(config.collision_threshold);
    value.at("page_size").get_to(config.page_size);
    return config;
}

bool NearlyEqual(double lhs, double rhs) {
    return std::abs(lhs - rhs) < Global::kEpsilon;
}

void ValidateJsonValue(const nlohmann::json& metadata, const std::string& key, const nlohmann::json& expected) {
    if (!metadata.contains(key) || metadata.at(key) != expected) {
        Log::error("QALSH weights metadata mismatch for '{}'.", key);
    }
}

void ValidateIdentity(const nlohmann::json& metadata, const std::string& key, const PointSetMetadata& expected) {
    const nlohmann::json expected_identity = PointSetIdentity(expected);
    if (!metadata.contains(key) || metadata.at(key) != expected_identity) {
        Log::error("QALSH weights metadata mismatch for '{}' point set.", key);
    }
}

void ValidateConfig(const QalshConfig& actual, const QalshConfig& expected) {
    if (!NearlyEqual(actual.approximation_ratio, expected.approximation_ratio) ||
        !NearlyEqual(actual.bucket_width, expected.bucket_width) ||
        !NearlyEqual(actual.error_probability, expected.error_probability) ||
        actual.num_hash_tables != expected.num_hash_tables ||
        actual.collision_threshold != expected.collision_threshold || actual.page_size != expected.page_size) {
        Log::error("QALSH weights metadata mismatch for QALSH config.");
    }
}

QalshConfig ExpectedConfig(const PointSetMetadata& to_metadata, double norm_order, double approximation_ratio) {
    QalshConfig config{
        .approximation_ratio = approximation_ratio,
        .page_size = Global::kDefaultPageSize,
    };
    config.Regularize(to_metadata.num_points, norm_order);
    return config;
}

std::vector<double> ReadWeights(const std::filesystem::path& directory, size_t num_weights) {
    std::vector<double> weights(num_weights);
    const std::filesystem::path weights_path = directory / "weights.bin";
    std::ifstream ifs(weights_path, std::ios::binary);
    if (!ifs.is_open()) {
        Log::error("Failed to open QALSH weights file: {}", weights_path.string());
    }
    ifs.read(reinterpret_cast<char*>(weights.data()), static_cast<std::streamsize>(weights.size() * sizeof(double)));
    if (ifs.gcount() != static_cast<std::streamsize>(weights.size() * sizeof(double))) {
        Log::error("QALSH weights file has unexpected size: {}", weights_path.string());
    }
    return weights;
}

void WriteWeights(const std::filesystem::path& directory, const std::vector<double>& weights) {
    std::filesystem::create_directories(directory);
    const std::filesystem::path weights_path = directory / "weights.bin";
    std::ofstream ofs(weights_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        Log::error("Failed to open QALSH weights file for writing: {}", weights_path.string());
    }
    ofs.write(reinterpret_cast<const char*>(weights.data()),
              static_cast<std::streamsize>(weights.size() * sizeof(double)));
}

void WriteMetadata(const std::filesystem::path& directory, const PointSetMetadata& from_metadata,
                   const PointSetMetadata& to_metadata, double norm_order, const QalshConfig& config) {
    std::filesystem::create_directories(directory);
    const nlohmann::json metadata{
        {"format", kWeightsFormat},
        {"norm_order", norm_order},
        {"direction", DirectionValue(from_metadata, to_metadata)},
        {"from", PointSetIdentity(from_metadata)},
        {"to", PointSetIdentity(to_metadata)},
        {"qalsh_config", QalshConfigJson(config)},
    };

    const std::filesystem::path metadata_path = directory / "metadata.json";
    std::ofstream ofs(metadata_path);
    if (!ofs.is_open()) {
        Log::error("Failed to open QALSH weights metadata for writing: {}", metadata_path.string());
    }
    ofs << metadata.dump(4);
}

QalshConfig ValidateAndReadMetadata(const std::filesystem::path& directory, const PointSetMetadata& from_metadata,
                                    const PointSetMetadata& to_metadata, double norm_order,
                                    double approximation_ratio) {
    const std::filesystem::path metadata_path = directory / "metadata.json";
    std::ifstream ifs(metadata_path);
    if (!ifs.is_open()) {
        Log::error("Failed to open QALSH weights metadata: {}", metadata_path.string());
    }

    nlohmann::json metadata = nlohmann::json::parse(ifs);
    ValidateJsonValue(metadata, "format", std::string(kWeightsFormat));
    ValidateJsonValue(metadata, "direction", DirectionValue(from_metadata, to_metadata));
    if (!metadata.contains("norm_order") || !NearlyEqual(metadata.at("norm_order").get<double>(), norm_order)) {
        Log::error("QALSH weights metadata mismatch for 'norm_order'.");
    }
    ValidateIdentity(metadata, "from", from_metadata);
    ValidateIdentity(metadata, "to", to_metadata);

    QalshConfig actual = ReadQalshConfig(metadata);
    QalshConfig expected = ExpectedConfig(to_metadata, norm_order, approximation_ratio);
    ValidateConfig(actual, expected);
    return actual;
}

}  // namespace

std::optional<QalshConfig> WeightsGenerator::GetQalshConfig() const {
    return std::nullopt;
}

// --------------------------------------------------
// UniformWeightsGenerator Implementation
// --------------------------------------------------
bool UniformWeightsGenerator::RequiresIndex() const {
    return false;
}

std::vector<double> UniformWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                      [[maybe_unused]] const PointSetMetadata& to_metadata,
                                                      [[maybe_unused]] double norm_order) {
    std::vector<double> weights(from_metadata.num_points, 1.0);
    return weights;
}

// --------------------------------------------------
// QalshWeightsGenerator Implementation
// --------------------------------------------------
QalshWeightsGenerator::QalshWeightsGenerator(double approximation_ratio,
                                             std::optional<std::filesystem::path> save_weights_directory,
                                             std::optional<std::filesystem::path> load_weights_directory)
    : approximation_ratio_(approximation_ratio),
      save_weights_directory_(std::move(save_weights_directory)),
      load_weights_directory_(std::move(load_weights_directory)) {}

bool QalshWeightsGenerator::RequiresIndex() const {
    return false;
}

std::vector<double> QalshWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                    const PointSetMetadata& to_metadata, double norm_order) {
    if (save_weights_directory_.has_value() && load_weights_directory_.has_value()) {
        Log::error("--save-weights and --load-weights cannot be used together.");
    }

    if (load_weights_directory_.has_value()) {
        qalsh_config_ =
            ValidateAndReadMetadata(*load_weights_directory_, from_metadata, to_metadata, norm_order,
                                    approximation_ratio_);
        return ReadWeights(*load_weights_directory_, from_metadata.num_points);
    }

    // Generate weights based on QALSH algorithm.
    Log::info("Generating weights using QALSH...");
    QalshAnnSearcher ann_searcher(false, approximation_ratio_);
    ann_searcher.Init(to_metadata, norm_order);
    qalsh_config_ = ann_searcher.GetConfig();
    if (!qalsh_config_.has_value()) {
        Log::error("QALSH config is unavailable after weight generation.");
    }

    std::vector<double> weights(from_metadata.num_points);
    PointSet base;
    base.Init(from_metadata);

    for (unsigned int i = 0; i < from_metadata.num_points; i++) {
        weights[i] = ann_searcher.Search(base.ReadPoint(i)).distance;
    }

    if (save_weights_directory_.has_value()) {
        WriteWeights(*save_weights_directory_, weights);
        WriteMetadata(*save_weights_directory_, from_metadata, to_metadata, norm_order, *qalsh_config_);
    }

    return weights;
}

std::optional<QalshConfig> QalshWeightsGenerator::GetQalshConfig() const {
    return qalsh_config_;
}
