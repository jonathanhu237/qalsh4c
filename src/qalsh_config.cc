#include "qalsh_config.h"

#include <cmath>
#include <numbers>

// NOLINTBEGIN: readability-magic-numbers
void QalshConfig::Regularize(unsigned int num_points) {
    bucket_width = 2.0 * std::sqrt(approximation_ratio);
    beta = 100.0 / static_cast<double>(num_points);

    double term1 = std::sqrt(std::log(2.0 / beta));
    double term2 = std::sqrt(std::log(1.0 / error_probability));
    double p1 = 2.0 / std::numbers::pi_v<double> * atan(bucket_width / 2.0);
    double p2 = 2.0 / std::numbers::pi_v<double> * atan(bucket_width / (2.0 * approximation_ratio));
    double numerator = std::pow(term1 + term2, 2.0);
    double denominator = 2.0 * std::pow(p1 - p2, 2.0);
    num_hash_tables = static_cast<unsigned int>(std::ceil(numerator / denominator));

    double eta = term1 / term2;
    double alpha = (eta * p1 + p2) / (1 + eta);
    collision_threshold = static_cast<unsigned int>(std::ceil(alpha * num_hash_tables));
}
// NOLINTEND
