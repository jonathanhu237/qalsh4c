#include "disk_qalsh_index.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

#include "global.h"
#include "log.h"
#include "qalsh_index.h"
#include "utils.h"

#ifdef QALSH4C_USE_OPENMP
#include <omp.h>
#endif

namespace {

constexpr float kMetadataFloatTolerance = 1e-5F;

bool NearlyEqual(float lhs, float rhs) {
    const float scale = std::max({1.0F, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= kMetadataFloatTolerance * scale;
}

float DotProductRaw(const Point& lhs, const Point& rhs) {
    float result = 0.0F;
    const float* lhs_data = lhs.data();
    const float* rhs_data = rhs.data();
    const auto num_dimensions = static_cast<long long>(lhs.size());

#ifdef QALSH4C_USE_OPENMP
#pragma omp simd reduction(+ : result)
#endif
    for (long long dimension = 0; dimension < num_dimensions; ++dimension) {
        result += lhs_data[dimension] * rhs_data[dimension];
    }

    return result;
}

void ValidateApproximationRatio(float approximation_ratio) {
    if (!(approximation_ratio > 1.0F)) {
        Log::error("QALSH approximation ratio must be greater than 1, got {}.", approximation_ratio);
    }
}

void ValidateTargetMetadata(const PointSetMetadata& metadata) {
    if (metadata.num_points == 0 || metadata.num_dimensions == 0) {
        Log::error("Disk QALSH requires a non-empty target point set with non-zero dimensions.");
    }
    if (!std::filesystem::exists(metadata.file_path)) {
        Log::error("Target point set file does not exist: {}", metadata.file_path.string());
    }
    const std::uintmax_t expected_bytes = static_cast<std::uintmax_t>(metadata.num_points) *
                                          static_cast<std::uintmax_t>(metadata.num_dimensions) *
                                          static_cast<std::uintmax_t>(sizeof(Coordinate));
    const std::uintmax_t actual_bytes = std::filesystem::file_size(metadata.file_path);
    if (actual_bytes != expected_bytes) {
        Log::error("Target point set file has unexpected size: expected {} bytes, got {} bytes from {}.",
                   expected_bytes, actual_bytes, metadata.file_path.string());
    }
}

std::function<float()> MakeCoordinateGenerator(float norm_order, unsigned int seed) {
    std::mt19937 gen(seed);
    if (std::abs(norm_order - 1.0F) < Global::kEpsilon) {
        std::cauchy_distribution<float> dist(0.0F, 1.0F);
        return [dist, gen]() mutable { return dist(gen); };
    }
    if (std::abs(norm_order - 2.0F) < Global::kEpsilon) {
        std::normal_distribution<float> dist(0.0F, 1.0F);
        return [dist, gen]() mutable { return dist(gen); };
    }
    Log::error("Unsupported norm order: {}", norm_order);
}

void ValidateMetadataKeys(const nlohmann::json& value, const std::filesystem::path& path) {
    static constexpr std::array<std::string_view, 7> kKeys{
        "approximation_ratio", "seed",          "num_points",          "num_dimensions",
        "num_hash_tables",     "collision_threshold", "bucket_width",
    };

    if (!value.is_object() || value.size() != kKeys.size()) {
        Log::error("Disk QALSH metadata must contain exactly {} fields: {}.", kKeys.size(), path.string());
    }
    for (const std::string_view key : kKeys) {
        if (!value.contains(key)) {
            Log::error("Disk QALSH metadata is missing field '{}' in {}.", key, path.string());
        }
    }
}

bool MetadataMatches(const DiskQalsh::IndexMetadata& lhs, const DiskQalsh::IndexMetadata& rhs) {
    return NearlyEqual(lhs.approximation_ratio, rhs.approximation_ratio) && lhs.seed == rhs.seed &&
           lhs.num_points == rhs.num_points && lhs.num_dimensions == rhs.num_dimensions &&
           lhs.num_hash_tables == rhs.num_hash_tables && lhs.collision_threshold == rhs.collision_threshold &&
           NearlyEqual(lhs.bucket_width, rhs.bucket_width);
}

void ValidateFileSize(const std::filesystem::path& path, std::uintmax_t expected_bytes) {
    if (!std::filesystem::exists(path)) {
        Log::error("Required disk QALSH file is missing: {}", path.string());
    }
    const std::uintmax_t actual_bytes = std::filesystem::file_size(path);
    if (actual_bytes != expected_bytes) {
        Log::error("Disk QALSH file has unexpected size: expected {} bytes, got {} bytes from {}.", expected_bytes,
                   actual_bytes, path.string());
    }
}

void BuildOneIndex(const DiskQalsh::BuildRequest& request, DiskQalsh::Target target,
                   const DatasetMetadata& dataset_metadata) {
    const PointSetMetadata target_metadata =
        DiskQalsh::MakePointSetMetadata(request.dataset_directory, dataset_metadata, target);
    ValidateTargetMetadata(target_metadata);

    const DiskQalsh::IndexMetadata expected =
        DiskQalsh::ExpectedMetadata(target_metadata, request.norm_order, request.approximation_ratio, request.seed);
    const std::filesystem::path index_directory =
        DiskQalsh::IndexDirectory(request.dataset_directory, request.norm_order, target);

    if (std::filesystem::exists(index_directory)) {
        if (!request.overwrite) {
            const DiskQalsh::IndexMetadata existing = DiskQalsh::ReadMetadata(index_directory);
            if (MetadataMatches(existing, expected)) {
                (void)DiskQalsh::ValidateIndex(target_metadata, request.norm_order, request.approximation_ratio);
                Log::info("Disk QALSH index already exists at {}.", index_directory.string());
                return;
            }
            Log::error(
                "Disk QALSH index already exists at {} but does not match requested parameters. Use --overwrite to "
                "rebuild it.",
                index_directory.string());
        }

        std::filesystem::remove_all(index_directory);
    }

    Log::info("Building disk QALSH index at {}.", index_directory.string());
    std::filesystem::create_directories(index_directory);

    std::vector<Point> base_points =
        PointSet::LoadPointsFromFile(target_metadata.file_path, target_metadata.num_points,
                                     target_metadata.num_dimensions);
    if (base_points.size() != target_metadata.num_points) {
        Log::error("Failed to load all target points for disk QALSH index build.");
    }

    std::function<float()> coordinate_generator = MakeCoordinateGenerator(request.norm_order, request.seed);
    std::vector<Point> dot_vectors;
    dot_vectors.reserve(expected.num_hash_tables);
    for (unsigned int table_id = 0; table_id < expected.num_hash_tables; ++table_id) {
        Point dot_vector;
        dot_vector.reserve(expected.num_dimensions);
        for (unsigned int dimension = 0; dimension < expected.num_dimensions; ++dimension) {
            dot_vector.push_back(coordinate_generator());
        }
        dot_vectors.push_back(std::move(dot_vector));
    }

    {
        std::ofstream dot_output(DiskQalsh::DotVectorsPath(index_directory), std::ios::binary | std::ios::trunc);
        if (!dot_output.is_open()) {
            Log::error("Failed to open disk QALSH dot vector file for writing: {}.",
                       DiskQalsh::DotVectorsPath(index_directory).string());
        }
        for (const Point& dot_vector : dot_vectors) {
            dot_output.write(reinterpret_cast<const char*>(dot_vector.data()),
                             static_cast<std::streamsize>(dot_vector.size() * sizeof(Coordinate)));
        }
    }

    std::ofstream table_output(DiskQalsh::TablesPath(index_directory), std::ios::binary | std::ios::trunc);
    if (!table_output.is_open()) {
        Log::error("Failed to open disk QALSH table file for writing: {}.",
                   DiskQalsh::TablesPath(index_directory).string());
    }

    std::vector<DiskQalsh::DiskQalshEntry> entries(target_metadata.num_points);
    for (unsigned int table_id = 0; table_id < expected.num_hash_tables; ++table_id) {
        const Point& dot_vector = dot_vectors[table_id];

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long point_index = 0; point_index < static_cast<long long>(target_metadata.num_points);
             ++point_index) {
            const auto point_id = static_cast<unsigned int>(point_index);
            entries[point_id] = DiskQalsh::DiskQalshEntry{
                .projection = DotProductRaw(base_points[point_id], dot_vector),
                .point_id = point_id,
            };
        }

        std::ranges::sort(entries, [](const DiskQalsh::DiskQalshEntry& lhs,
                                      const DiskQalsh::DiskQalshEntry& rhs) {
            if (lhs.projection != rhs.projection) {
                return lhs.projection < rhs.projection;
            }
            return lhs.point_id < rhs.point_id;
        });
        table_output.write(reinterpret_cast<const char*>(entries.data()),
                           static_cast<std::streamsize>(entries.size() * sizeof(DiskQalsh::DiskQalshEntry)));
    }

    DiskQalsh::WriteMetadata(index_directory, expected);
    Log::info("Built disk QALSH index at {}.", index_directory.string());
}

}  // namespace

namespace DiskQalsh {

QalshConfig IndexMetadata::ToConfig() const {
    return QalshConfig{.approximation_ratio = approximation_ratio,
                       .bucket_width = bucket_width,
                       .error_probability = Global::kQalshDefaultErrorProbability,
                       .num_hash_tables = num_hash_tables,
                       .collision_threshold = collision_threshold};
}

std::string NormDirectoryName(float norm_order) {
    if (std::abs(norm_order - 1.0F) < Global::kEpsilon) {
        return "l1";
    }
    if (std::abs(norm_order - 2.0F) < Global::kEpsilon) {
        return "l2";
    }
    Log::error("Unsupported norm order: {}", norm_order);
}

std::string TargetDirectoryName(Target target) {
    switch (target) {
        case Target::kA:
            return "a";
        case Target::kB:
            return "b";
    }
    Log::error("Unsupported disk QALSH target.");
}

TargetSelection ParseTargetSelection(std::string_view value) {
    if (value == "a") {
        return TargetSelection::kA;
    }
    if (value == "b") {
        return TargetSelection::kB;
    }
    if (value == "both") {
        return TargetSelection::kBoth;
    }
    Log::error("Unsupported disk QALSH target selection: {}", value);
}

Target ParseTargetFromPointSetPath(const std::filesystem::path& file_path) {
    const std::string stem = file_path.stem().string();
    if (stem == "A") {
        return Target::kA;
    }
    if (stem == "B") {
        return Target::kB;
    }
    Log::error("Cannot infer disk QALSH target from point set file name: {}.", file_path.string());
}

std::filesystem::path IndexDirectory(const std::filesystem::path& dataset_directory, float norm_order, Target target) {
    return dataset_directory / "index" / "qalsh" / NormDirectoryName(norm_order) / TargetDirectoryName(target);
}

std::filesystem::path MetadataPath(const std::filesystem::path& index_directory) {
    return index_directory / "meta.json";
}

std::filesystem::path DotVectorsPath(const std::filesystem::path& index_directory) {
    return index_directory / "dot_vectors.bin";
}

std::filesystem::path TablesPath(const std::filesystem::path& index_directory) {
    return index_directory / "tables.bin";
}

PointSetMetadata MakePointSetMetadata(const std::filesystem::path& dataset_directory,
                                      const DatasetMetadata& dataset_metadata, Target target) {
    const bool is_a = target == Target::kA;
    return PointSetMetadata{
        .file_path = dataset_directory / (is_a ? "A.bin" : "B.bin"),
        .num_points = is_a ? dataset_metadata.num_points_a : dataset_metadata.num_points_b,
        .num_dimensions = dataset_metadata.num_dimensions,
    };
}

IndexMetadata ReadMetadata(const std::filesystem::path& index_directory) {
    const std::filesystem::path path = MetadataPath(index_directory);
    if (!std::filesystem::exists(path)) {
        Log::error("Disk QALSH metadata does not exist. Run `index qalsh` first: {}", path.string());
    }

    std::ifstream input(path);
    if (!input.is_open()) {
        Log::error("Failed to open disk QALSH metadata: {}", path.string());
    }

    nlohmann::json value;
    try {
        value = nlohmann::json::parse(input);
        ValidateMetadataKeys(value, path);

        return IndexMetadata{
            .approximation_ratio = value.at("approximation_ratio").get<float>(),
            .seed = value.at("seed").get<unsigned int>(),
            .num_points = value.at("num_points").get<unsigned int>(),
            .num_dimensions = value.at("num_dimensions").get<unsigned int>(),
            .num_hash_tables = value.at("num_hash_tables").get<unsigned int>(),
            .collision_threshold = value.at("collision_threshold").get<unsigned int>(),
            .bucket_width = value.at("bucket_width").get<float>(),
        };
    } catch (const nlohmann::json::exception& e) {
        Log::error("Disk QALSH metadata JSON format error in {}: {}", path.string(), e.what());
    }
}

void WriteMetadata(const std::filesystem::path& index_directory, const IndexMetadata& metadata) {
    nlohmann::json value;
    value["approximation_ratio"] = metadata.approximation_ratio;
    value["seed"] = metadata.seed;
    value["num_points"] = metadata.num_points;
    value["num_dimensions"] = metadata.num_dimensions;
    value["num_hash_tables"] = metadata.num_hash_tables;
    value["collision_threshold"] = metadata.collision_threshold;
    value["bucket_width"] = metadata.bucket_width;

    const std::filesystem::path path = MetadataPath(index_directory);
    std::ofstream output(path, std::ios::trunc);
    if (!output.is_open()) {
        Log::error("Failed to open disk QALSH metadata for writing: {}", path.string());
    }
    output << value.dump(4);
}

IndexMetadata ExpectedMetadata(const PointSetMetadata& target_metadata, float norm_order, float approximation_ratio,
                               unsigned int seed) {
    ValidateApproximationRatio(approximation_ratio);
    ValidateTargetMetadata(target_metadata);

    QalshConfig config{.approximation_ratio = approximation_ratio};
    config.Regularize(target_metadata.num_points, norm_order);
    if (config.num_hash_tables == 0 || config.collision_threshold == 0) {
        Log::error("Invalid derived disk QALSH configuration: hash tables {}, collision threshold {}.",
                   config.num_hash_tables, config.collision_threshold);
    }

    return IndexMetadata{.approximation_ratio = approximation_ratio,
                         .seed = seed,
                         .num_points = target_metadata.num_points,
                         .num_dimensions = target_metadata.num_dimensions,
                         .num_hash_tables = config.num_hash_tables,
                         .collision_threshold = config.collision_threshold,
                         .bucket_width = config.bucket_width};
}

IndexMetadata ValidateIndex(const PointSetMetadata& target_metadata, float norm_order,
                            std::optional<float> expected_approximation_ratio) {
    ValidateTargetMetadata(target_metadata);

    const Target target = ParseTargetFromPointSetPath(target_metadata.file_path);
    const std::filesystem::path dataset_directory = target_metadata.file_path.parent_path();
    const std::filesystem::path index_directory = IndexDirectory(dataset_directory, norm_order, target);
    if (!std::filesystem::exists(index_directory)) {
        Log::error("Missing disk QALSH index at {}. Run `index qalsh -p {} -d {} --target {}` first.",
                   index_directory.string(), NormDirectoryName(norm_order).substr(1), dataset_directory.string(),
                   TargetDirectoryName(target));
    }

    const IndexMetadata metadata = ReadMetadata(index_directory);
    ValidateApproximationRatio(metadata.approximation_ratio);
    if (expected_approximation_ratio.has_value() &&
        !NearlyEqual(metadata.approximation_ratio, *expected_approximation_ratio)) {
        Log::error("Disk QALSH approximation ratio mismatch: index has {}, command requested {}.",
                   metadata.approximation_ratio, *expected_approximation_ratio);
    }

    if (metadata.num_points != target_metadata.num_points || metadata.num_dimensions != target_metadata.num_dimensions) {
        Log::error(
            "Disk QALSH metadata mismatch for {}: index has {} points and {} dimensions, dataset has {} points and "
            "{} dimensions.",
            index_directory.string(), metadata.num_points, metadata.num_dimensions, target_metadata.num_points,
            target_metadata.num_dimensions);
    }
    if (metadata.num_hash_tables == 0 || metadata.collision_threshold == 0) {
        Log::error("Disk QALSH index has invalid table count {} or collision threshold {}.",
                   metadata.num_hash_tables, metadata.collision_threshold);
    }

    QalshConfig derived{.approximation_ratio = metadata.approximation_ratio};
    derived.Regularize(target_metadata.num_points, norm_order);
    if (metadata.num_hash_tables != derived.num_hash_tables ||
        metadata.collision_threshold != derived.collision_threshold || !NearlyEqual(metadata.bucket_width, derived.bucket_width)) {
        Log::error("Disk QALSH metadata does not match the derived QALSH configuration for {}.",
                   index_directory.string());
    }

    const std::uintmax_t expected_table_bytes = static_cast<std::uintmax_t>(metadata.num_hash_tables) *
                                                static_cast<std::uintmax_t>(metadata.num_points) *
                                                static_cast<std::uintmax_t>(sizeof(DiskQalshEntry));
    const std::uintmax_t expected_dot_bytes = static_cast<std::uintmax_t>(metadata.num_hash_tables) *
                                              static_cast<std::uintmax_t>(metadata.num_dimensions) *
                                              static_cast<std::uintmax_t>(sizeof(Coordinate));
    ValidateFileSize(TablesPath(index_directory), expected_table_bytes);
    ValidateFileSize(DotVectorsPath(index_directory), expected_dot_bytes);

    return metadata;
}

void BuildIndexes(const BuildRequest& request) {
    ValidateApproximationRatio(request.approximation_ratio);

    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(request.dataset_directory / "metadata.json");

    switch (request.target_selection) {
        case TargetSelection::kA:
            BuildOneIndex(request, Target::kA, dataset_metadata);
            return;
        case TargetSelection::kB:
            BuildOneIndex(request, Target::kB, dataset_metadata);
            return;
        case TargetSelection::kBoth:
            BuildOneIndex(request, Target::kA, dataset_metadata);
            BuildOneIndex(request, Target::kB, dataset_metadata);
            return;
    }

    Log::error("Unsupported disk QALSH target selection.");
}

}  // namespace DiskQalsh
