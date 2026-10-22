#include "disk_qalsh_index.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include <qalsh/qalsh.h>

#include "global.h"
#include "log.h"
#include "utils.h"

namespace {

constexpr float kMetadataFloatTolerance = 1e-5F;

bool NearlyEqual(float lhs, float rhs) {
    const float scale = std::max({1.0F, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= kMetadataFloatTolerance * scale;
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
    const std::uintmax_t expected_bytes = static_cast<std::uintmax_t>(metadata.num_points) * metadata.num_dimensions *
                                          sizeof(Coordinate);
    const std::uintmax_t actual_bytes = std::filesystem::file_size(metadata.file_path);
    if (actual_bytes != expected_bytes) {
        Log::error("Target point set file has unexpected size: expected {} bytes, got {} bytes from {}.",
                   expected_bytes, actual_bytes, metadata.file_path.string());
    }
}

bool MetadataMatches(const DiskQalsh::IndexMetadata& lhs, const DiskQalsh::IndexMetadata& rhs) {
    return NearlyEqual(lhs.approximation_ratio, rhs.approximation_ratio) && lhs.seed == rhs.seed &&
           lhs.num_points == rhs.num_points && lhs.num_dimensions == rhs.num_dimensions &&
           lhs.num_hash_tables == rhs.num_hash_tables && lhs.collision_threshold == rhs.collision_threshold &&
           NearlyEqual(lhs.bucket_width, rhs.bucket_width) && lhs.layout == rhs.layout;
}

qalsh::Metric ToMetric(float norm_order) {
    if (std::abs(norm_order - 1.0F) < Global::kEpsilon) {
        return qalsh::Metric::l1;
    }
    if (std::abs(norm_order - 2.0F) < Global::kEpsilon) {
        return qalsh::Metric::l2;
    }
    Log::error("Unsupported norm order: {}", norm_order);
}

void BuildOneIndex(const DiskQalsh::BuildRequest& request, DiskQalsh::Target target,
                   const DatasetMetadata& dataset_metadata) {
    const PointSetMetadata target_metadata =
        DiskQalsh::MakePointSetMetadata(request.dataset_directory, dataset_metadata, target);
    ValidateTargetMetadata(target_metadata);
    const DiskQalsh::IndexMetadata expected =
        DiskQalsh::ExpectedMetadata(target_metadata, request.norm_order, request.approximation_ratio, request.seed,
                                    request.layout);
    const std::filesystem::path index_directory =
        DiskQalsh::IndexDirectory(request.dataset_directory, request.norm_order, target);

    if (std::filesystem::exists(index_directory)) {
        if (!request.overwrite) {
            const DiskQalsh::IndexMetadata existing = DiskQalsh::ReadMetadata(index_directory);
            if (MetadataMatches(existing, expected)) {
                (void)DiskQalsh::ValidateIndex(target_metadata, request.norm_order, request.approximation_ratio);
                Log::info("Disk QALSH library index already exists at {}.", index_directory.string());
                return;
            }
            Log::error("Disk QALSH index already exists at {} but does not match requested parameters. Use --overwrite "
                       "to rebuild it.",
                       index_directory.string());
        }
        // Keep the existing immutable index in place until the new selected
        // layout has been fully written. PersistentIndex::Build publishes its
        // file atomically, so a failed rebuild cannot leave a data-loss window.
    }

    std::filesystem::create_directories(index_directory);
    Log::info("Building disk QALSH {} index at {}.", DiskQalsh::IndexLayoutName(request.layout),
              index_directory.string());
    PointSet points;
    points.Init(target_metadata);
    Point buffer(target_metadata.num_dimensions);
    const qalsh::PointAccessor accessor = [&](qalsh::PointId point_id) -> qalsh::PointView {
        points.ReadPointInto(point_id, buffer);
        return buffer;
    };
    const qalsh::IndexConfig config{
        .metric = ToMetric(request.norm_order),
        .num_points = expected.num_points,
        .num_dimensions = expected.num_dimensions,
        .qalsh = qalsh::QalshParameters{.approximation_ratio = expected.approximation_ratio,
                                         .bucket_width = expected.bucket_width,
                                         .error_probability = Global::kQalshDefaultErrorProbability,
                                         .num_hash_tables = expected.num_hash_tables,
                                         .collision_threshold = expected.collision_threshold,
                                         .initial_radius = 1.0F,
                                         .radius_growth = expected.approximation_ratio,
                                         .scan_quantum = Global::kScanSize * 2},
        .seed = expected.seed,
        .projection_vectors = {}};
    qalsh::PersistentIndex::Build(
        DiskQalsh::IndexPath(request.dataset_directory, request.norm_order, target).string(), config, accessor,
        qalsh::PersistentBuildOptions{.layout = request.layout,
                                      .page_size = qalsh::BPlusTreeIndex::kDefaultPageSize,
                                      .overwrite = true});
    DiskQalsh::WriteMetadata(index_directory, expected);
    Log::info("Built disk QALSH {} index at {}.", DiskQalsh::IndexLayoutName(request.layout),
              index_directory.string());
}

}  // namespace

namespace DiskQalsh {

void ValidateLibraryMetadata(const qalsh::IndexMetadata& library_metadata,
                             const IndexMetadata& metadata, float norm_order,
                             const std::filesystem::path& index_directory) {
    const qalsh::Metric expected_metric =
        std::abs(norm_order - 1.0F) < Global::kEpsilon ? qalsh::Metric::l1 : qalsh::Metric::l2;
    const auto& library_qalsh = library_metadata.qalsh;
    const bool matches =
        library_metadata.num_points == metadata.num_points &&
        library_metadata.num_dimensions == metadata.num_dimensions &&
        library_metadata.num_hash_tables == metadata.num_hash_tables &&
        library_metadata.layout == metadata.layout &&
        library_metadata.page_size == qalsh::BPlusTreeIndex::kDefaultPageSize &&
        library_metadata.seed == metadata.seed && library_metadata.metric == expected_metric &&
        NearlyEqual(library_qalsh.approximation_ratio, metadata.approximation_ratio) &&
        NearlyEqual(library_qalsh.bucket_width, metadata.bucket_width) &&
        NearlyEqual(library_qalsh.error_probability, Global::kQalshDefaultErrorProbability) &&
        library_qalsh.num_hash_tables == metadata.num_hash_tables &&
        library_qalsh.collision_threshold == metadata.collision_threshold &&
        library_qalsh.candidate_budget == Global::kNumCandidates &&
        NearlyEqual(library_qalsh.initial_radius, 1.0F) &&
        NearlyEqual(library_qalsh.radius_growth, metadata.approximation_ratio) &&
        library_qalsh.scan_quantum == Global::kScanSize * 2;
    if (!matches) {
        Log::error("Library QALSH index metadata does not match {}.", index_directory.string());
    }
}

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

qalsh::IndexLayout ParseIndexLayout(std::string_view value) {
    if (value == "b-plus-tree") {
        return qalsh::IndexLayout::b_plus_tree;
    }
    if (value == "sorted-array") {
        return qalsh::IndexLayout::sorted_array;
    }
    Log::error("Unsupported disk QALSH index layout: {}", value);
}

const char* IndexLayoutName(qalsh::IndexLayout layout) noexcept {
    switch (layout) {
        case qalsh::IndexLayout::b_plus_tree:
            return "B+ tree";
        case qalsh::IndexLayout::sorted_array:
            return "sorted-array";
        case qalsh::IndexLayout::in_memory:
            return "in-memory";
    }
    return "unknown";
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

std::filesystem::path IndexPath(const std::filesystem::path& dataset_directory, float norm_order, Target target) {
    return IndexDirectory(dataset_directory, norm_order, target) / "index.qalsh";
}

std::filesystem::path MetadataPath(const std::filesystem::path& index_directory) {
    return index_directory / "meta.json";
}

std::filesystem::path DotVectorsPath(const std::filesystem::path& index_directory) {
    return index_directory / "index.qalsh";
}

std::filesystem::path TablesPath(const std::filesystem::path& index_directory) {
    return index_directory / "index.qalsh";
}

PointSetMetadata MakePointSetMetadata(const std::filesystem::path& dataset_directory,
                                      const DatasetMetadata& dataset_metadata, Target target) {
    const bool is_a = target == Target::kA;
    return PointSetMetadata{.file_path = dataset_directory / (is_a ? "A.bin" : "B.bin"),
                            .num_points = is_a ? dataset_metadata.num_points_a : dataset_metadata.num_points_b,
                            .num_dimensions = dataset_metadata.num_dimensions};
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
    try {
        const nlohmann::json value = nlohmann::json::parse(input);
        return IndexMetadata{.approximation_ratio = value.at("approximation_ratio").get<float>(),
                             .seed = value.at("seed").get<unsigned int>(),
                             .num_points = value.at("num_points").get<unsigned int>(),
                             .num_dimensions = value.at("num_dimensions").get<unsigned int>(),
                             .num_hash_tables = value.at("num_hash_tables").get<unsigned int>(),
                             .collision_threshold = value.at("collision_threshold").get<unsigned int>(),
                             .bucket_width = value.at("bucket_width").get<float>(),
                             .layout = value.contains("layout")
                                           ? ParseIndexLayout(value.at("layout").get<std::string>())
                                           : qalsh::IndexLayout::b_plus_tree};
    } catch (const nlohmann::json::exception& error) {
        Log::error("Disk QALSH metadata JSON format error in {}: {}", path.string(), error.what());
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
    value["layout"] = metadata.layout == qalsh::IndexLayout::sorted_array ? "sorted-array" : "b-plus-tree";

    const std::filesystem::path destination = MetadataPath(index_directory);
    std::string temporary_template = destination.string() + ".tmp-XXXXXX";
    std::vector<char> temporary_name(temporary_template.begin(), temporary_template.end());
    temporary_name.push_back('\0');
    int fd = ::mkstemp(temporary_name.data());
    if (fd < 0) {
        Log::error("Failed to create disk QALSH metadata temporary file: {} ({})",
                   destination.string(), std::strerror(errno));
    }
    const std::string temporary_path(temporary_name.data());
    const std::string serialized = value.dump(4);
    bool installed = false;
    try {
        std::size_t written = 0;
        while (written < serialized.size()) {
            const ssize_t count = ::write(fd, serialized.data() + written, serialized.size() - written);
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count <= 0) {
                throw std::runtime_error("failed to write disk QALSH metadata");
            }
            written += static_cast<std::size_t>(count);
        }
        if (::fsync(fd) != 0) {
            throw std::runtime_error("failed to flush disk QALSH metadata");
        }
        if (::close(fd) != 0) {
            fd = -1;
            throw std::runtime_error("failed to close disk QALSH metadata");
        }
        fd = -1;
        if (::rename(temporary_path.c_str(), destination.c_str()) != 0) {
            throw std::runtime_error("failed to atomically install disk QALSH metadata");
        }
        installed = true;
    } catch (const std::exception& error) {
        if (fd >= 0) {
            (void)::close(fd);
        }
        (void)::unlink(temporary_path.c_str());
        Log::error("Failed to write disk QALSH metadata {}: {}", destination.string(), error.what());
    }
    if (!installed) {
        Log::error("Failed to install disk QALSH metadata: {}", destination.string());
    }
}

IndexMetadata ExpectedMetadata(const PointSetMetadata& target_metadata, float norm_order, float approximation_ratio,
                               unsigned int seed, qalsh::IndexLayout layout) {
    ValidateApproximationRatio(approximation_ratio);
    ValidateTargetMetadata(target_metadata);
    QalshConfig config{.approximation_ratio = approximation_ratio};
    config.Regularize(target_metadata.num_points, norm_order);
    return IndexMetadata{.approximation_ratio = approximation_ratio,
                         .seed = seed,
                         .num_points = target_metadata.num_points,
                         .num_dimensions = target_metadata.num_dimensions,
                         .num_hash_tables = config.num_hash_tables,
                         .collision_threshold = config.collision_threshold,
                         .bucket_width = config.bucket_width,
                         .layout = layout};
}

IndexMetadata ValidateIndex(const PointSetMetadata& target_metadata, float norm_order,
                            std::optional<float> expected_approximation_ratio,
                            bool validate_structure) {
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
        Log::error("Disk QALSH metadata mismatch for {}.", index_directory.string());
    }
    if (metadata.num_hash_tables == 0 || metadata.collision_threshold == 0) {
        Log::error("Disk QALSH index has invalid table count {} or collision threshold {}.", metadata.num_hash_tables,
                   metadata.collision_threshold);
    }
    QalshConfig derived{.approximation_ratio = metadata.approximation_ratio};
    derived.Regularize(target_metadata.num_points, norm_order);
    if (metadata.num_hash_tables != derived.num_hash_tables ||
        metadata.collision_threshold != derived.collision_threshold ||
        !NearlyEqual(metadata.bucket_width, derived.bucket_width)) {
        Log::error("Disk QALSH metadata does not match the derived QALSH configuration for {}.",
                   index_directory.string());
    }
    const std::filesystem::path index_path = IndexPath(dataset_directory, norm_order, target);
    if (!validate_structure) {
        if (!std::filesystem::is_regular_file(index_path)) {
            Log::error("Missing disk QALSH library index at {}.", index_path.string());
        }
        return metadata;
    }
    try {
        const auto index = qalsh::PersistentIndex::Open(index_path.string());
        ValidateLibraryMetadata(index->metadata(), metadata, norm_order, index_directory);
    } catch (const std::exception& error) {
        Log::error("Failed to open library QALSH index at {}: {}", index_path.string(), error.what());
    }
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
