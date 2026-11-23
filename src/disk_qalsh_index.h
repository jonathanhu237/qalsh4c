#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include <qalsh/qalsh.h>

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

struct IndexMetadata {
    float approximation_ratio{0.0F};
    unsigned int seed{0};
    unsigned int num_points{0};
    unsigned int num_dimensions{0};
    unsigned int num_hash_tables{0};
    unsigned int collision_threshold{0};
    float bucket_width{0.0F};
    qalsh::IndexLayout layout{qalsh::IndexLayout::b_plus_tree};

    [[nodiscard]] QalshConfig ToConfig() const;
};

struct BuildRequest {
    std::filesystem::path dataset_directory;
    float norm_order{0.0F};
    TargetSelection target_selection{TargetSelection::kB};
    unsigned int seed{0};
    float approximation_ratio{0.0F};
    bool overwrite{false};
    qalsh::IndexLayout layout{qalsh::IndexLayout::b_plus_tree};
};

[[nodiscard]] std::string NormDirectoryName(float norm_order);
[[nodiscard]] std::string TargetDirectoryName(Target target);
[[nodiscard]] TargetSelection ParseTargetSelection(std::string_view value);
[[nodiscard]] qalsh::IndexLayout ParseIndexLayout(std::string_view value);
[[nodiscard]] const char* IndexLayoutName(qalsh::IndexLayout layout) noexcept;
[[nodiscard]] Target ParseTargetFromPointSetPath(const std::filesystem::path& file_path);
[[nodiscard]] std::filesystem::path IndexDirectory(const std::filesystem::path& dataset_directory, float norm_order,
                                                   Target target);
[[nodiscard]] std::filesystem::path IndexPath(const std::filesystem::path& dataset_directory, float norm_order,
                                              Target target);
[[nodiscard]] std::filesystem::path MetadataPath(const std::filesystem::path& index_directory);
// Kept as a named compatibility helper for migration tooling. The library
// format stores vectors in the index file rather than in a flat sidecar.
[[nodiscard]] std::filesystem::path DotVectorsPath(const std::filesystem::path& index_directory);
[[nodiscard]] std::filesystem::path TablesPath(const std::filesystem::path& index_directory);
[[nodiscard]] PointSetMetadata MakePointSetMetadata(const std::filesystem::path& dataset_directory,
                                                    const DatasetMetadata& dataset_metadata, Target target);

[[nodiscard]] IndexMetadata ReadMetadata(const std::filesystem::path& index_directory);
void WriteMetadata(const std::filesystem::path& index_directory, const IndexMetadata& metadata);
[[nodiscard]] IndexMetadata ExpectedMetadata(const PointSetMetadata& target_metadata, float norm_order,
                                             float approximation_ratio, unsigned int seed,
                                             qalsh::IndexLayout layout = qalsh::IndexLayout::b_plus_tree);
[[nodiscard]] IndexMetadata ValidateIndex(const PointSetMetadata& target_metadata, float norm_order,
                                          std::optional<float> expected_approximation_ratio = std::nullopt,
                                          bool validate_structure = true);
void ValidateLibraryMetadata(const qalsh::IndexMetadata& library_metadata,
                             const IndexMetadata& metadata, float norm_order,
                             const std::filesystem::path& index_directory);

void BuildIndexes(const BuildRequest& request);

}  // namespace DiskQalsh
