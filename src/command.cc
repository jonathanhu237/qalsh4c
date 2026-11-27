#include "command.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <ratio>
#include <utility>

#include "ann_searcher.h"
#include "estimator.h"
#include "global.h"
#include "kappa.h"
#include "log.h"
#include "point_set.h"
#include "utils.h"

namespace {

PointSetMetadata MakePointSetMetadata(const std::filesystem::path& dataset_directory,
                                      const DatasetMetadata& dataset_metadata, const std::string& point_set_name) {
    const bool is_a = point_set_name == "A";
    return PointSetMetadata{
        .file_path = dataset_directory / std::format("{}.bin", point_set_name),
        .num_points = is_a ? dataset_metadata.num_points_a : dataset_metadata.num_points_b,
        .num_dimensions = dataset_metadata.num_dimensions,
    };
}

// |estimate - C| / C in percent, with no floor on C: only C = 0 is special.
float RelativeErrorPercentage(float estimation, float ground_truth) {
    if (ground_truth == 0.0F) {
        return estimation == 0.0F ? 0.0F : std::numeric_limits<float>::infinity();
    }
    const double error = std::fabs(static_cast<double>(estimation) - static_cast<double>(ground_truth)) /
                         std::fabs(static_cast<double>(ground_truth));
    return static_cast<float>(error * 100.0);  // NOLINT: readability-magic-numbers
}

}  // namespace

// --------------------------------------------------
// EstimateCommand Implementation
// --------------------------------------------------
EstimateCommand::EstimateCommand(std::unique_ptr<Estimator> estimator, float norm_order,
                                 std::filesystem::path dataset_directory, ChamferDirection direction)
    : estimator_(std::move(estimator)),
      norm_order_(norm_order),
      dataset_directory_(std::move(dataset_directory)),
      direction_(direction) {}

void EstimateCommand::Execute() {
    // Load dataset metadata
    Log::info("Loading dataset metadata...");
    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory_ / "metadata.json");

    // Construct point set metadata
    PointSetMetadata point_set_metadata_a = MakePointSetMetadata(dataset_directory_, dataset_metadata, "A");
    PointSetMetadata point_set_metadata_b = MakePointSetMetadata(dataset_directory_, dataset_metadata, "B");

    auto start = std::chrono::high_resolution_clock::now();
    float memory_before = Memory::Usage();
    float estimation{0.0};

    if (direction_ == ChamferDirection::kAb || direction_ == ChamferDirection::kBoth) {
        Log::info("Calculating the distance from A to B...");
        estimation += estimator_->EstimateDistance(point_set_metadata_a, point_set_metadata_b, norm_order_);
    }

    if (direction_ == ChamferDirection::kBa || direction_ == ChamferDirection::kBoth) {
        Log::info("Calculating the distance from B to A...");
        estimation += estimator_->EstimateDistance(point_set_metadata_b, point_set_metadata_a, norm_order_);
    }

    auto end = std::chrono::high_resolution_clock::now();
    float memory_after = Memory::Usage();

    // Output the result.
    std::optional<float> ground_truth = dataset_metadata.GroundTruth(norm_order_, direction_);
    float elapsed_time = std::chrono::duration<float, std::milli>(end - start).count();
    float memory_usage = memory_after - memory_before;

    if (ground_truth.has_value()) {
        float relative_error_percentage = RelativeErrorPercentage(estimation, *ground_truth);
        std::cout << std::format(
            "Direction: {}\n"
            "Estimated Distance: {:.6f}\n"
            "Ground Truth: {:.6f}\n"
            "Time Consumed: {:.3f} ms\n"
            "Memory Usage: {:.2f} MB\n"
            "Relative Error: {:.2f}%\n",
            ChamferDirectionLabel(direction_), estimation, *ground_truth, elapsed_time, memory_usage,
            relative_error_percentage);
        return;
    }

    std::cout << std::format(
        "Direction: {}\n"
        "Estimated Distance: {:.6f}\n"
        "Ground Truth: unavailable\n"
        "Time Consumed: {:.3f} ms\n"
        "Memory Usage: {:.2f} MB\n"
        "Relative Error: unavailable\n",
        ChamferDirectionLabel(direction_), estimation, elapsed_time, memory_usage);
}

KappaCommand::KappaCommand(float norm_order, float approximation_ratio, std::vector<double> epsilons, double delta)
    : norm_order_(norm_order), approximation_ratio_(approximation_ratio), epsilons_(std::move(epsilons)),
      delta_(delta) {}

void KappaCommand::Execute() {
    QalshConfig config{.approximation_ratio = approximation_ratio_};
    // With beta = 1, m and l do not depend on the number of target points.
    config.Regularize(1, norm_order_);
    const kappa::Parameters parameters = MakeKappaParameters(config, norm_order_);
    try {
        const auto start = std::chrono::steady_clock::now();
        const kappa::Bound bound = kappa::CertifyKappa(parameters);
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout << std::format(
            "kappa norm=l{} c={} gamma={} w={:.9g} m={} l={} h={} lower={:.9f} upper={:.9f} kappa_up={}.{:06d} "
            "grid_points={} grid_end={:.4f} ms={:.1f}\n",
            parameters.norm, approximation_ratio_, parameters.gamma, parameters.bucket_width,
            parameters.num_hash_tables, parameters.collision_threshold,
            parameters.num_hash_tables - parameters.collision_threshold + 1, bound.lower, bound.upper,
            bound.millionths / 1'000'000, bound.millionths % 1'000'000, bound.grid_points, bound.grid_end, elapsed);
        for (const double epsilon : epsilons_) {
            std::cout << std::format("T epsilon={} delta={} T={}\n", epsilon, delta_,
                                     kappa::SampleSize(bound.millionths, epsilon, delta_));
        }
    } catch (const std::exception& error) {
        Log::error("Certifying kappa failed: {}", error.what());
    }
}
