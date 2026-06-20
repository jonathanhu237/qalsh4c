#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "config.h"
#include "point_set.h"

namespace DiskQalsh {

enum class Target {
    kA,
    kB,
};

enum class TargetSelection {
    kA,
    kB,
    kBoth,
};

struct DiskQalshEntry {
    float projection{0.0F};
    std::uint32_t point_id{0};
};

static_assert(sizeof(DiskQalshEntry) == 8);

struct IndexMetadata {
    float approximation_ratio{0.0F};
    unsigned int seed{0};
    unsigned int num_points{0};
    unsigned int num_dimensions{0};
    unsigned int num_hash_tables{0};
    unsigned int collision_threshold{0};
    float bucket_width{0.0F};

    [[nodiscard]] QalshConfig ToConfig() const;
};

struct BuildRequest {
    std::filesystem::path dataset_directory;
    float norm_order{0.0F};
    TargetSelection target_selection{TargetSelection::kB};
    unsigned int seed{0};
    float approximation_ratio{0.0F};
    bool overwrite{false};
};

[[nodiscard]] std::string NormDirectoryName(float norm_order);
[[nodiscard]] std::string TargetDirectoryName(Target target);
[[nodiscard]] TargetSelection ParseTargetSelection(std::string_view value);
[[nodiscard]] Target ParseTargetFromPointSetPath(const std::filesystem::path& file_path);
[[nodiscard]] std::filesystem::path IndexDirectory(const std::filesystem::path& dataset_directory, float norm_order,
                                                   Target target);
[[nodiscard]] std::filesystem::path MetadataPath(const std::filesystem::path& index_directory);
[[nodiscard]] std::filesystem::path DotVectorsPath(const std::filesystem::path& index_directory);
[[nodiscard]] std::filesystem::path TablesPath(const std::filesystem::path& index_directory);
[[nodiscard]] PointSetMetadata MakePointSetMetadata(const std::filesystem::path& dataset_directory,
                                                    const DatasetMetadata& dataset_metadata, Target target);

[[nodiscard]] IndexMetadata ReadMetadata(const std::filesystem::path& index_directory);
void WriteMetadata(const std::filesystem::path& index_directory, const IndexMetadata& metadata);
[[nodiscard]] IndexMetadata ExpectedMetadata(const PointSetMetadata& target_metadata, float norm_order,
                                             float approximation_ratio, unsigned int seed);
[[nodiscard]] IndexMetadata ValidateIndex(const PointSetMetadata& target_metadata, float norm_order,
                                          std::optional<float> expected_approximation_ratio = std::nullopt);

void BuildIndexes(const BuildRequest& request);

}  // namespace DiskQalsh
