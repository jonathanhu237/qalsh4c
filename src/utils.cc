#include "utils.h"

#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "global.h"

double Calc::LpDistance(const Point &pt1, const Point &pt2, double norm_order) {
    Eigen::Map<const Eigen::VectorXd> v1(pt1.data(), static_cast<Eigen::Index>(pt1.size()));
    Eigen::Map<const Eigen::VectorXd> v2(pt2.data(), static_cast<Eigen::Index>(pt2.size()));

    if (std::abs(norm_order - 1.0) < Global::kEpsilon) {
        return static_cast<double>((v1 - v2).lpNorm<1>());
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    if (std::abs(norm_order - 2.0) < Global::kEpsilon) {
        return static_cast<double>((v1 - v2).lpNorm<2>());
    }
    spdlog::error("Unsupported norm order: {}", norm_order);
    return 0.0;
}

double Calc::DotProduct(const Point &pt1, const Point &pt2) {
    Eigen::Map<const Eigen::VectorXd> v1(pt1.data(), static_cast<Eigen::Index>(pt1.size()));
    Eigen::Map<const Eigen::VectorXd> v2(pt2.data(), static_cast<Eigen::Index>(pt2.size()));

    return v1.dot(v2);
}

double Calc::L1Probability(double x) { return 2.0 / std::numbers::pi_v<double> * atan(x); }  // 2 / pi * atan(x)

double Calc::L2Probability(double x) { return std::erf(x * M_SQRT1_2); }  // 2 * Phi(x) - 1

double Memory::Usage() {
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
    return static_cast<double>(memory_usage) / 1024.0;  // NOLINT: readability-magic-numbers
}

std::mt19937 Random::SeededGenerator() {
    if (Global::kUseFixedSeed) {
        return std::mt19937(Global::kDefaultSeed);
    }
    return std::mt19937(std::random_device{}());
}

std::vector<Point> Utils::LoadPointsFromFile(const std::filesystem::path &file_path, unsigned int num_points,
                                             unsigned int num_dimensions) {
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) {
        spdlog::error("Could not open base points file: {}", file_path.string());
        return {};
    }

    unsigned int total_doubles = num_points * num_dimensions;
    std::vector<double> buffer(total_doubles);
    ifs.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamoff>(total_doubles * sizeof(Coordinate)));

    std::vector<Point> points;
    points.reserve(num_points);
    for (unsigned int i = 0; i < num_points; ++i) {
        auto start_offset = i * num_dimensions;
        auto end_offset = start_offset + num_dimensions;
        auto start_it = buffer.begin() + start_offset;
        auto end_it = buffer.begin() + end_offset;
        points.emplace_back(start_it, end_it);
    }

    return points;
}

Point Utils::ReadPoint(std::ifstream &ifs, unsigned int num_dimensions, unsigned int point_id) {
    ifs.seekg(static_cast<std::streamoff>(static_cast<unsigned long>(point_id * num_dimensions) * sizeof(Coordinate)),
              std::ios::beg);
    Point point(num_dimensions);
    ifs.read(reinterpret_cast<char *>(point.data()), static_cast<std::streamoff>(num_dimensions * sizeof(Coordinate)));
    return point;
}

unsigned int Utils::SampleFromWeights(const std::vector<double> &weights) {
    double total_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total_sum <= 0) {
        throw std::runtime_error("Total sum of weights must be positive.");
    }

    std::vector<double> cumulative_weights;
    cumulative_weights.reserve(weights.size());
    std::partial_sum(weights.begin(), weights.end(), std::back_inserter(cumulative_weights));

    static std::mt19937 gen = Random::SeededGenerator();
    std::uniform_real_distribution<> dis(0.0, total_sum);
    double random_value = dis(gen);

    auto it = std::ranges::upper_bound(cumulative_weights, random_value);

    return static_cast<unsigned int>(std::distance(cumulative_weights.begin(), it));
}
