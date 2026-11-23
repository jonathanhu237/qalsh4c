#include "estimator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ann_searcher.h"
#include "config.h"
#include "global.h"
#include "kappa.h"
#include "log.h"
#include "point_set.h"
#include "utils.h"
#include "weights_generator.h"

namespace {

std::vector<Point> LoadSampledQueryPoints(const PointSetMetadata& metadata,
                                          const std::vector<std::pair<unsigned int, unsigned int>>& sample_counts) {
    PointSet point_set;
    point_set.Init(metadata);

    std::vector<Point> query_points;
    query_points.reserve(sample_counts.size());
    for (const auto& [point_id, count] : sample_counts) {
        (void)count;
        query_points.push_back(point_set.ReadPoint(point_id));
    }

    return query_points;
}

void InitExactScanSearcher(LinearScanAnnSearcher& ann_searcher, const PointSetMetadata& metadata, float norm_order,
                           const std::shared_ptr<const std::vector<Point>>& loaded_target_points) {
    if (loaded_target_points) {
        if (loaded_target_points->size() != metadata.num_points) {
            Log::error("Loaded target point count does not match exact scan metadata.");
        }
        ann_searcher.InitLoaded(loaded_target_points, metadata.num_dimensions, norm_order);
        return;
    }
    ann_searcher.Init(metadata, norm_order);
}

// d(a, B) of every source point, written by an exact computation, for experiments that repeat the estimator.
std::vector<double> LoadExactDistances(const std::filesystem::path& path, unsigned int num_points) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        Log::error("Failed to open the exact distances file: {}", path.string());
    }
    const auto expected_bytes = static_cast<std::streamoff>(num_points) * static_cast<std::streamoff>(sizeof(double));
    if (input.tellg() != expected_bytes) {
        Log::error("The exact distances file {} should hold {} float64 values.", path.string(), num_points);
    }
    input.seekg(0);
    std::vector<double> distances(num_points);
    input.read(reinterpret_cast<char*>(distances.data()), expected_bytes);
    if (!input) {
        Log::error("Failed to read the exact distances file: {}", path.string());
    }
    return distances;
}

// T from the certified bound on kappa for the stopping rule the QALSH search uses (Theorems 1 and 2).
unsigned int AutomaticQalshSampleCount(const QalshConfig& qalsh_config, float norm_order, double epsilon,
                                       double delta) {
    try {
        const auto start = std::chrono::steady_clock::now();
        const kappa::Bound bound = kappa::CertifyKappa(MakeKappaParameters(qalsh_config, norm_order));
        const std::uint64_t samples = kappa::SampleSize(bound.millionths, epsilon, delta);
        Log::info("Certified kappa <= {:.6f} (gamma = {}, {} grid points, {:.1f} ms): T = {} for epsilon = {}, delta = {}",
                  static_cast<double>(bound.millionths) / 1e6, QalshStoppingFactor(qalsh_config.approximation_ratio),
                  bound.grid_points,
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(), samples,
                  epsilon, delta);
        if (samples > std::numeric_limits<unsigned int>::max()) {
            Log::error("Automatic QALSH sampling selected too many samples: {}", samples);
        }
        return static_cast<unsigned int>(samples);
    } catch (const std::exception& error) {
        Log::error("Automatic QALSH sampling failed: {}", error.what());
    }
}

}  // namespace

// --------------------------------------------------
// AnnEstimator Implementation
// --------------------------------------------------
AnnEstimator::AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher, std::filesystem::path save_distances)
    : ann_searcher_(std::move(ann_searcher)), save_distances_(std::move(save_distances)) {}

float AnnEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, float norm_order) {
    if (!ann_searcher_) {
        Log::error("The ANN searcher is not set.");
    }

    ann_searcher_->Init(to, norm_order);
    std::vector<Point> query_points = PointSet::LoadPointsFromFile(from.file_path, from.num_points, from.num_dimensions);
    std::vector<Distance> distances(from.num_points, 0.0F);

    const auto* linear_scan = dynamic_cast<const LinearScanAnnSearcher*>(ann_searcher_.get());
    if (linear_scan == nullptr || !linear_scan->UsesBatchedScan()) {
        Log::error("The exact computation needs the batched linear scan.");
    }
    const std::vector<AnnResult> results = linear_scan->SearchBatched(
        query_points, std::vector<Distance>(query_points.size(), std::numeric_limits<Distance>::infinity()));
    std::ranges::transform(results, distances.begin(), &AnnResult::distance);

    std::vector<double> exact(distances.begin(), distances.end());
    if (!save_distances_.empty()) {
        std::ofstream output(save_distances_, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(exact.data()),
                     static_cast<std::streamsize>(exact.size() * sizeof(double)));
        if (!output) {
            Log::error("Failed to write the exact distances file: {}", save_distances_.string());
        }
        Log::info("Saved d(a, B) of {} source points to {}", exact.size(), save_distances_.string());
    }
    double total = 0.0;
    for (const double distance : exact) {
        total += distance;
    }
    return static_cast<float>(total);
}

// --------------------------------------------------
// SamplingEstimator Implementation
// --------------------------------------------------
SamplingEstimator::SamplingEstimator(std::unique_ptr<WeightsGenerator> weights_generator, SamplingOptions options)
    : weights_generator_(std::move(weights_generator)), options_(std::move(options)) {
    if (options_.repeats == 0) {
        Log::error("The number of repeats must be positive.");
    }
}

unsigned int SamplingEstimator::SampleCount(const PointSetMetadata& from, const PointSetMetadata& to,
                                            float norm_order) const {
    const std::string from_name = from.file_path.stem().string();
    const std::string to_name = to.file_path.stem().string();
    const bool is_ab_direction = from_name == "A" && to_name == "B";
    const bool is_ba_direction = from_name == "B" && to_name == "A";

    if (is_ab_direction && options_.num_samples_ab != 0) {
        Log::info("Using specified A->B sample count: {}", options_.num_samples_ab);
        return options_.num_samples_ab;
    }
    if (is_ba_direction && options_.num_samples_ba != 0) {
        Log::info("Using specified B->A sample count: {}", options_.num_samples_ba);
        return options_.num_samples_ba;
    }
    if (options_.num_samples_default != 0) {
        Log::info("Using default sample count: {}", options_.num_samples_default);
        return options_.num_samples_default;
    }

    // Without a sample count, QALSH weights take T from the certified bound on kappa (Theorems 1 and 2).
    auto* qalsh_weights_generator = dynamic_cast<QalshWeightsGenerator*>(weights_generator_.get());
    if (qalsh_weights_generator == nullptr) {
        Log::error("Only QALSH weights choose the sample count themselves; specify -n for other weights.");
    }
    if (!(options_.epsilon > 0.0F)) {
        Log::error("Epsilon must be positive for automatic QALSH sampling.");
    }
    if (!(options_.delta > 0.0F) || !(options_.delta < 1.0F)) {
        Log::error("Delta must be in (0, 1) for automatic QALSH sampling.");
    }
    const QalshConfig qalsh_config = qalsh_weights_generator->BuildConfig(from, to, norm_order, options_.use_cache);
    const unsigned int samples = AutomaticQalshSampleCount(qalsh_config, norm_order, options_.epsilon, options_.delta);
    Log::info("SamplingEstimator automatically selected {} samples.", samples);
    return samples;
}

float SamplingEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to,
                                           float norm_order) {
    if (!weights_generator_) {
        Log::error("The weights generator is not set.");
    }
    auto* qalsh_weights_generator = dynamic_cast<QalshWeightsGenerator*>(weights_generator_.get());
    // QALSH crude distances are distances to target points, so they bound the exact scan of a sampled point.
    const bool weights_are_upper_bounds = qalsh_weights_generator != nullptr;

    const unsigned int num_samples = SampleCount(from, to, norm_order);

    Log::info("Generating weights...");
    const auto weights_start = std::chrono::steady_clock::now();
    const std::vector<Weight> weights = weights_generator_->Generate(from, to, norm_order, options_.use_cache);
    const auto weights_end = std::chrono::steady_clock::now();
    Log::info("Stage time: weights_ms={:.3f}",
              std::chrono::duration<double, std::milli>(weights_end - weights_start).count());
    if (weights.size() != from.num_points) {
        Log::error("Weights size does not match the number of query points in the dataset");
    }

    WeightedSampler sampler(weights);
    const double total = sampler.Total();
    Log::info("Total sum of weights: {:.17g}", total);

    const bool look_up = !options_.exact_distances.empty();
    std::vector<double> exact_distances;
    if (look_up) {
        exact_distances = LoadExactDistances(options_.exact_distances, from.num_points);
    }

    double estimate = 0.0;
    for (unsigned int repeat = 1; repeat <= options_.repeats; ++repeat) {
        // Draw T points with probability D_a / D and count repeated draws once.
        std::unordered_map<unsigned int, unsigned int> sample_count_map;
        sample_count_map.reserve(num_samples);
        for (unsigned int sample_index = 0; sample_index < num_samples; ++sample_index) {
            ++sample_count_map[sampler.Sample()];
        }
        std::vector<std::pair<unsigned int, unsigned int>> sample_counts(sample_count_map.begin(),
                                                                         sample_count_map.end());
        std::ranges::sort(sample_counts);

        // d(x, B) of every distinct sampled point x.
        std::vector<double> nearest(sample_counts.size());
        if (look_up) {
            for (size_t index = 0; index < sample_counts.size(); ++index) {
                nearest[index] = exact_distances[sample_counts[index].first];
            }
        } else {
            nearest = ExactDistancesOfSamples(from, to, norm_order, sample_counts, weights, weights_are_upper_bounds);
        }

        // eta = (1/T) sum_l D d(x_l, B) / D_{x_l}; the sampler never draws a point of weight zero.
        double sum = 0.0;
        for (size_t index = 0; index < sample_counts.size(); ++index) {
            const auto& [point_id, count] = sample_counts[index];
            sum += static_cast<double>(count) * total * nearest[index] / static_cast<double>(weights[point_id]);
        }
        estimate = sum / static_cast<double>(num_samples);
        Log::info("Sample estimate: repeat={} T={} estimate={:.17g}", repeat, num_samples, estimate);
    }
    return static_cast<float>(estimate);
}

std::vector<double> SamplingEstimator::ExactDistancesOfSamples(
    const PointSetMetadata& from, const PointSetMetadata& to, float norm_order,
    const std::vector<std::pair<unsigned int, unsigned int>>& sample_counts, const std::vector<Weight>& weights,
    bool weights_are_upper_bounds) const {
    auto* qalsh_weights_generator = dynamic_cast<QalshWeightsGenerator*>(weights_generator_.get());
    const std::shared_ptr<const std::vector<Point>> loaded_target_points =
        qalsh_weights_generator == nullptr ? nullptr : qalsh_weights_generator->TargetPoints();

    const std::vector<Point> query_points = LoadSampledQueryPoints(from, sample_counts);
    LinearScanAnnSearcher shared_ann_searcher;
    InitExactScanSearcher(shared_ann_searcher, to, norm_order, loaded_target_points);

    // A crude distance bounds the nearest-neighbor distance from above, which lets the scan prune.
    std::vector<Distance> upper_bounds(sample_counts.size());
    for (size_t index = 0; index < sample_counts.size(); ++index) {
        upper_bounds[index] = weights_are_upper_bounds ? weights[sample_counts[index].first]
                                                       : std::numeric_limits<Distance>::infinity();
    }

    if (!shared_ann_searcher.UsesBatchedScan()) {
        Log::error("The exact stage needs the batched linear scan.");
    }
    const std::vector<AnnResult> results = shared_ann_searcher.SearchBatched(query_points, upper_bounds);
    std::vector<double> nearest(sample_counts.size());
    for (size_t index = 0; index < results.size(); ++index) {
        nearest[index] = static_cast<double>(results[index].distance);
    }
    return nearest;
}
