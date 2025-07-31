#include "weights_generator.h"

#include <spdlog/spdlog.h>

#include <vector>

// --------------------------------------------------
// UniformWeightsGenerator Implementation
// --------------------------------------------------
std::vector<double> UniformWeightsGenerator::Generate(const PointSetMetadata& from_metadata,
                                                      [[maybe_unused]] const PointSetMetadata& to_metadata,
                                                      [[maybe_unused]] bool in_memory,
                                                      [[maybe_unused]] bool use_cache) {
    std::vector<double> weights(from_metadata.num_points, 1.0);
    return weights;
}
