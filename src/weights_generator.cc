#include "weights_generator.h"

#include <vector>

#include "types.h"

std::vector<double> UniformWeightsGenerator::Generate(const std::filesystem::path& dataset_directory) {
    // Load dataset metadata
    DatasetMetadata metadata;
    metadata.Load(dataset_directory / "metadata.toml");

    std::vector<double> weights(metadata.base_num_points, 1.0);
    return weights;
}