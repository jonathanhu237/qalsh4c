#include "utils.h"

#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

double Utils::L1Distance(const Point &pt1, const Point &pt2) {
    if (pt1.size() != pt2.size()) {
        spdlog::error("Vectors must be of the same size");
    }

    Eigen::Map<const Eigen::VectorXd> v1(pt1.data(), static_cast<Eigen::Index>(pt1.size()));
    Eigen::Map<const Eigen::VectorXd> v2(pt2.data(), static_cast<Eigen::Index>(pt2.size()));

    return static_cast<double>((v1 - v2).lpNorm<1>());
}

double Utils::DotProduct(const Point &pt1, const Point &pt2) {
    if (pt1.size() != pt2.size()) {
        spdlog::error("Points must be of the same size for dot product. vec1.size(): {}, vec2.size(): {}", pt1.size(),
                      pt2.size());
    }

    Eigen::Map<const Eigen::VectorXd> v1(pt1.data(), static_cast<Eigen::Index>(pt1.size()));
    Eigen::Map<const Eigen::VectorXd> v2(pt2.data(), static_cast<Eigen::Index>(pt2.size()));

    return v1.dot(v2);
}

unsigned int Utils::SampleFromWeights(const std::vector<double> &weights) {
    double total_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total_sum <= 0) {
        throw std::runtime_error("Total sum of weights must be positive.");
    }

    std::vector<double> cumulative_weights;
    cumulative_weights.reserve(weights.size());
    std::partial_sum(weights.begin(), weights.end(), std::back_inserter(cumulative_weights));

    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(0.0, total_sum);
    double random_value = dis(gen);

    auto it = std::ranges::upper_bound(cumulative_weights, random_value);

    return static_cast<unsigned int>(std::distance(cumulative_weights.begin(), it));
}

double Utils::GetMemoryUsage() {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    size_t memory_usage = 0;

    while (std::getline(status_file, line)) {
        // NOLINTNEXTLINE: readability-magic-numbers
        if (line.substr(0, 6) == "VmHWM:") {
            std::istringstream iss(line);
            std::string key;
            iss >> key >> memory_usage;
            break;
        }
    }

    // Convert from KB to MB
    return memory_usage / 1024.0;  // NOLINT: readability-magic-numbers
}
