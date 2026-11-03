#include "estimator.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <vector>

#include "global.h"
#include "utils.h"

namespace {

void WritePointFile(const std::filesystem::path& path, const std::vector<Point>& points) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    for (const Point& point : points) {
        output.write(reinterpret_cast<const char*>(point.data()),
                     static_cast<std::streamsize>(point.size() * sizeof(Coordinate)));
    }
}

PointSetMetadata Metadata(const std::filesystem::path& path, size_t num_points, unsigned int num_dimensions) {
    return PointSetMetadata{
        .file_path = path,
        .num_points = static_cast<unsigned int>(num_points),
        .num_dimensions = num_dimensions,
    };
}

std::vector<Point> RandomPoints(std::mt19937& gen, unsigned int num_points, unsigned int num_dimensions) {
    std::uniform_real_distribution<Coordinate> dis(-10.0F, 10.0F);
    std::vector<Point> points(num_points, Point(num_dimensions));
    for (Point& point : points) {
        for (Coordinate& coordinate : point) {
            coordinate = dis(gen);
        }
    }
    return points;
}

double ExactChamfer(const std::vector<Point>& sources, const std::vector<Point>& targets, float norm_order) {
    double total = 0.0;
    for (const Point& source : sources) {
        Distance best = std::numeric_limits<Distance>::infinity();
        for (const Point& target : targets) {
            best = std::min(best, Calc::LpDistance(source, target, norm_order));
        }
        total += best;
    }
    return total;
}

bool RelativelyClose(double estimate, double truth, double tolerance) {
    return std::abs(estimate - truth) <= tolerance * truth;
}

}  // namespace

int main() {
    Global::kUseFixedSeed = true;
    const std::filesystem::path temp_dir = std::filesystem::current_path() / "pool_sampling_estimator_test_tmp";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    constexpr unsigned int kNumDimensions = 8;
    std::mt19937 gen(7);
    const std::vector<Point> targets = RandomPoints(gen, 60, kNumDimensions);
    std::vector<Point> sources = RandomPoints(gen, 40, kNumDimensions);
    // Sources that coincide with targets contribute zero and must not bias the estimate.
    sources.push_back(targets[3]);
    sources.push_back(targets[17]);
    WritePointFile(temp_dir / "sources.bin", sources);
    WritePointFile(temp_dir / "targets.bin", targets);
    const PointSetMetadata source_metadata = Metadata(temp_dir / "sources.bin", sources.size(), kNumDimensions);
    const PointSetMetadata target_metadata = Metadata(temp_dir / "targets.bin", targets.size(), kNumDimensions);

    int status = 0;
    for (const float norm_order : {1.0F, 2.0F}) {
        const double truth = ExactChamfer(sources, targets, norm_order);

        // With the whole target set as the subsample, U_a = v_a and every ratio D_a / U_a is one, so the
        // two-stage estimate is exact for any pool and sample sizes.
        PoolSamplingEstimator full_subsample(7, 5, 1000, 2.0F, 1.0F, 0.1F, 0);
        if (!RelativelyClose(full_subsample.EstimateDistance(source_metadata, target_metadata, norm_order), truth,
                             1e-4)) {
            status = 1;
            break;
        }

        // The direct baseline skips QALSH; with U_a = v_a its importance weights are exact as well.
        PoolSamplingEstimator direct(0, 5, 1000, 2.0F, 1.0F, 0.1F, 0, false, Global::kDefaultDiskBlockSize,
                                     Global::kDefaultDiskQueryBatchSize, std::nullopt, {}, true);
        if (!RelativelyClose(direct.EstimateDistance(source_metadata, target_metadata, norm_order), truth, 1e-4)) {
            status = 4;
            break;
        }

        // The full ablation queries every source point; with U_a = v_a its fused weights are exact too.
        PoolSamplingEstimator full(0, 5, 1000, 2.0F, 1.0F, 0.1F, 0, false, Global::kDefaultDiskBlockSize,
                                   Global::kDefaultDiskQueryBatchSize, std::nullopt, {}, false, true, true);
        if (!RelativelyClose(full.EstimateDistance(source_metadata, target_metadata, norm_order), truth, 1e-4)) {
            status = 5;
            break;
        }

        // A five-point subsample gives loose bounds; large first and second stages still land near the truth.
        PoolSamplingEstimator small_subsample(20000, 20000, 5, 2.0F, 1.0F, 0.1F, 0);
        if (!RelativelyClose(small_subsample.EstimateDistance(source_metadata, target_metadata, norm_order), truth,
                             0.05)) {
            status = 2;
            break;
        }

        // Plans with exact bounds see no variance in the pilot (the certified one still pays its radius), and every
        // design they pick is exact.
        for (const PoolPlanMode mode : {PoolPlanMode::kModel, PoolPlanMode::kCertified}) {
            for (const bool full_request : {false, true}) {
                PoolSamplingEstimator planned(0, 0, 1000, 2.0F, 1.0F, 0.1F, 0, false, Global::kDefaultDiskBlockSize,
                                              Global::kDefaultDiskQueryBatchSize, std::nullopt, {}, false,
                                              full_request, true, mode, 0.05F, 64);
                if (!RelativelyClose(planned.EstimateDistance(source_metadata, target_metadata, norm_order), truth,
                                     1e-4)) {
                    status = 6;
                }
            }
        }

        // A certified one-stage design on this small set draws more often than there are source points before its
        // confidence sequence could certify 5%, so it scans the rest and returns the exact sum.
        PoolSamplingEstimator certified_direct(0, 0, 5, 2.0F, 1.0F, 0.1F, 0, false, Global::kDefaultDiskBlockSize,
                                               Global::kDefaultDiskQueryBatchSize, std::nullopt, {}, true, false, true,
                                               PoolPlanMode::kCertified, 0.05F, 64);
        if (!RelativelyClose(certified_direct.EstimateDistance(source_metadata, target_metadata, norm_order), truth,
                             1e-4)) {
            status = 12;
        }

        // With loose bounds, plans for a 2% RMS error land near the truth for the pooled, direct and uniform designs.
        for (const bool direct_request : {false, true}) {
            PoolSamplingEstimator planned(0, 0, 5, 2.0F, 1.0F, 0.1F, 0, false, Global::kDefaultDiskBlockSize,
                                          Global::kDefaultDiskQueryBatchSize, std::nullopt, {}, direct_request, false,
                                          true, PoolPlanMode::kModel, 0.02F, 256);
            if (!RelativelyClose(planned.EstimateDistance(source_metadata, target_metadata, norm_order), truth,
                                 0.1)) {
                status = 7;
            }
        }
        PoolSamplingEstimator planned_uniform(0, 0, 5, 2.0F, 1.0F, 0.1F, 0, false, Global::kDefaultDiskBlockSize,
                                              Global::kDefaultDiskQueryBatchSize, std::nullopt, {}, false, false,
                                              true, PoolPlanMode::kModel, 0.02F, 256, true);
        if (!RelativelyClose(planned_uniform.EstimateDistance(source_metadata, target_metadata, norm_order), truth,
                             0.1)) {
            status = 8;
        }

        // Stratified plans: with exact bounds every stratum estimate is exact, and with loose bounds a plan for a 2%
        // RMS error lands near the truth. The certified plan needs a longer pilot, split across four strata.
        for (const PoolPlanMode mode : {PoolPlanMode::kModel, PoolPlanMode::kCertified}) {
            PoolSamplingEstimator stratified(0, 0, 1000, 2.0F, 1.0F, 0.1F, 0, false, Global::kDefaultDiskBlockSize,
                                             Global::kDefaultDiskQueryBatchSize, std::nullopt, {}, false, false, true,
                                             mode, 0.05F, 512, false, PoolSamplingEstimator::kDefaultPilotRelativeError,
                                             4);
            if (!RelativelyClose(stratified.EstimateDistance(source_metadata, target_metadata, norm_order), truth,
                                 1e-4)) {
                status = 10;
            }
        }
        PoolSamplingEstimator stratified_loose(0, 0, 5, 2.0F, 1.0F, 0.1F, 0, false, Global::kDefaultDiskBlockSize,
                                               Global::kDefaultDiskQueryBatchSize, std::nullopt, {}, false, false, true,
                                               PoolPlanMode::kModel, 0.02F, 256, false,
                                               PoolSamplingEstimator::kDefaultPilotRelativeError, 4);
        if (!RelativelyClose(stratified_loose.EstimateDistance(source_metadata, target_metadata, norm_order), truth,
                             0.1)) {
            status = 11;
        }

        // The uniform baseline scans from scratch, so many uniform draws land near the truth whatever the bounds.
        PoolSamplingEstimator uniform(0, 20000, 5, 2.0F, 1.0F, 0.1F, 0, false, Global::kDefaultDiskBlockSize,
                                      Global::kDefaultDiskQueryBatchSize, std::nullopt, {}, false, false, true,
                                      PoolPlanMode::kFixed, 0.0F, PoolSamplingEstimator::kDefaultPilotSize, true);
        if (!RelativelyClose(uniform.EstimateDistance(source_metadata, target_metadata, norm_order), truth, 0.05)) {
            status = 9;
        }
        if (status != 0) {
            break;
        }
    }

    if (status == 0) {
        // Every source coincides with a target, so the distance is exactly zero.
        const std::vector<Point> copies{targets[0], targets[5], targets[59]};
        WritePointFile(temp_dir / "copies.bin", copies);
        PoolSamplingEstimator estimator(10, 10, 5, 2.0F, 1.0F, 0.1F, 0);
        if (estimator.EstimateDistance(Metadata(temp_dir / "copies.bin", copies.size(), kNumDimensions),
                                       target_metadata, 2.0F) != 0.0F) {
            status = 3;
        }
    }

    std::filesystem::remove_all(temp_dir);
    return status;
}
