#include "utils.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "global.h"

double Utils::L1Distance(const Point &pt1, const Point &pt2) {
    Eigen::Map<const Eigen::VectorXd> v1(pt1.data(), static_cast<Eigen::Index>(pt1.size()));
    Eigen::Map<const Eigen::VectorXd> v2(pt2.data(), static_cast<Eigen::Index>(pt2.size()));

    return static_cast<double>((v1 - v2).lpNorm<1>());
}

double Utils::DotProduct(const Point &pt1, const Point &pt2) {
    Eigen::Map<const Eigen::VectorXd> v1(pt1.data(), static_cast<Eigen::Index>(pt1.size()));
    Eigen::Map<const Eigen::VectorXd> v2(pt2.data(), static_cast<Eigen::Index>(pt2.size()));

    return v1.dot(v2);
}

DatasetMetadata Utils::LoadDatasetMetadata(const std::filesystem::path &file_path) {
    DatasetMetadata metadata;

    if (!std::filesystem::exists(file_path)) {
        spdlog::error("The dataset metadata file does not exists, file path: {}", file_path.string());
    }

    std::ifstream ifs(file_path);
    nlohmann::json json_metadata = nlohmann::json::parse(ifs);

    try {
        json_metadata.at("num_points_a").get_to(metadata.num_points_a);
        json_metadata.at("num_points_b").get_to(metadata.num_points_b);
        json_metadata.at("num_dimensions").get_to(metadata.num_dimensions);
        json_metadata.at("chamfer_distance").get_to(metadata.chamfer_distance);
    } catch (nlohmann::json::exception &e) {
        spdlog::error("JSON format error: {}", e.what());
    }

    return metadata;
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

// NOLINTBEGIN(readability-magic-numbers)
void Utils::RegularizeQalshConfig(QalshConfig &config, unsigned int num_points) {
    config.bucket_width = 2.0 * std::sqrt(config.approximation_ratio);
    double beta = Global::kNumCandidates / static_cast<double>(num_points);
    config.error_probability = Global::kQalshDefaultErrorProbability;

    double term1 = std::sqrt(std::log(2.0 / beta));
    double term2 = std::sqrt(std::log(1.0 / config.error_probability));
    double p1 = 2.0 / std::numbers::pi_v<double> * atan(config.bucket_width / 2.0);
    double p2 = 2.0 / std::numbers::pi_v<double> * atan(config.bucket_width / (2.0 * config.approximation_ratio));
    double numerator = std::pow(term1 + term2, 2.0);
    double denominator = 2.0 * std::pow(p1 - p2, 2.0);
    config.num_hash_tables = static_cast<unsigned int>(std::ceil(numerator / denominator));

    double eta = term1 / term2;
    double alpha = (eta * p1 + p2) / (1 + eta);
    config.collision_threshold = static_cast<unsigned int>(std::ceil(alpha * config.num_hash_tables));
}
// NOLINTEND(readability-magic-numbers)

void Utils::SaveQalshConfig(QalshConfig &config, const std::filesystem::path &file_path) {
    nlohmann::json metadata;
    metadata["approximation_ratio"] = config.approximation_ratio;
    metadata["bucket_width"] = config.bucket_width;
    metadata["error_probability"] = config.error_probability;
    metadata["num_hash_tables"] = config.num_hash_tables;
    metadata["collision_threshold"] = config.collision_threshold;
    metadata["page_size"] = config.page_size;

    std::ofstream ofs(file_path);
    if (!ofs.is_open()) {
        spdlog::error("Failed to open file for writing: {}", file_path.string());
        return;
    }

    ofs << metadata.dump(4);
}

QalshConfig Utils::LoadQalshConfig(const std::filesystem::path &file_path) {
    if (!std::filesystem::exists(file_path)) {
        spdlog::error("The QALSH config file does not exists, file path: {}", file_path.string());
    }

    std::ifstream ifs(file_path);
    nlohmann::json metadata = nlohmann::json::parse(ifs);
    QalshConfig config;

    try {
        metadata.at("approximation_ratio").get_to(config.approximation_ratio);
        metadata.at("bucket_width").get_to(config.bucket_width);
        metadata.at("error_probability").get_to(config.error_probability);
        metadata.at("num_hash_tables").get_to(config.num_hash_tables);
        metadata.at("collision_threshold").get_to(config.collision_threshold);
        metadata.at("page_size").get_to(config.page_size);
    } catch (nlohmann::json::exception &e) {
        spdlog::error("JSON format error: {}", e.what());
    }

    return config;
}

unsigned int Utils::SampleFromWeights(const std::vector<double> &weights) {
    double total_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total_sum <= 0) {
        throw std::runtime_error("Total sum of weights must be positive.");
    }

    std::vector<double> cumulative_weights;
    cumulative_weights.reserve(weights.size());
    std::partial_sum(weights.begin(), weights.end(), std::back_inserter(cumulative_weights));

    static std::mt19937 gen = CreateSeededGenerator();
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
    return static_cast<double>(memory_usage) / 1024.0;  // NOLINT: readability-magic-numbers
}

std::mt19937 Utils::CreateSeededGenerator() {
    if (Global::kUseFixedSeed) {
        return std::mt19937(Global::kDefaultSeed);
    }
    return std::mt19937(std::random_device{}());
}