#include "estimator.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "ann_searcher.h"
#include "point_set.h"
#include "utils.h"

// --------------------------------------------------
// AnnEstimator Implementation
// --------------------------------------------------
AnnEstimator::AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher, bool use_cache)
    : ann_searcher_(std::move(ann_searcher)), use_cache_(use_cache) {}

double AnnEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, bool in_memory) {
    // Check the ANN searcher
    if (!ann_searcher_) {
        spdlog::error("The ANN searcher is not set.");
    }

    std::filesystem::path dataset_directory = from.file_path.parent_path();
    std::string stem = from.file_path.stem();
    std::filesystem::path cache_file = dataset_directory / std::format("qalsh_weights_{}.bin", stem);
    std::vector<double> distances(from.num_points, 0.0);

    if (use_cache_) {
        if ([[maybe_unused]] auto* linear_searcher = dynamic_cast<LinearScanAnnSearcher*>(ann_searcher_.get())) {
            spdlog::warn("Use cache is not supported for Linear Scan, the setting will be ignored.");
        } else if ([[maybe_unused]] auto* qalsh_searcher = dynamic_cast<QalshAnnSearcher*>(ann_searcher_.get())) {
            // Read the cache files
            std::ifstream ifs(cache_file, std::ios::binary);
            if (ifs.is_open()) {
                spdlog::info("Loading cached weights from {}", cache_file.string());
                ifs.read(reinterpret_cast<char*>(distances.data()),
                         static_cast<std::streamoff>(distances.size() * sizeof(Coordinate)));

                return std::accumulate(distances.begin(), distances.end(), 0.0);
            }
            spdlog::warn("Cache file {} does not exist, generating new ones", cache_file.string());
        } else {
            spdlog::error("Unknown ANN searcher type.");
        }
    }

    ann_searcher_->Reset();
    ann_searcher_->Init(to, in_memory);

    std::unique_ptr<PointSet> query_set;
    if (in_memory) {
        query_set = std::make_unique<InMemoryPointSet>(from);
    } else {
        query_set = std::make_unique<DiskPointSet>(from);
    }

    for (unsigned int point_id = 0; point_id < query_set->get_num_points(); point_id++) {
        Point query_point = query_set->GetPoint(point_id);
        distances[point_id] = ann_searcher_->Search(query_point).distance;
    }

    if (use_cache_) {
        // Write the distances to the cache file
        std::ofstream ofs(cache_file, std::ios::binary);
        if (ofs.is_open()) {
            spdlog::info("Saving cached weights to {}", cache_file.string());
            ofs.write(reinterpret_cast<const char*>(distances.data()),
                      static_cast<std::streamoff>(distances.size() * sizeof(Coordinate)));
        } else {
            spdlog::error("Failed to open cache file for writing: {}", cache_file.string());
        }
    }

    return std::accumulate(distances.begin(), distances.end(), 0.0);
}

// --------------------------------------------------
// SamplingEstimator Implementation
// --------------------------------------------------
SamplingEstimator::SamplingEstimator(std::unique_ptr<WeightsGenerator> weights_generator, unsigned int num_samples,
                                     bool use_cache)
    : weights_generator_(std::move(weights_generator)), num_samples_(num_samples), use_cache_(use_cache) {}

double SamplingEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, bool in_memory) {
    // Check the ANN searcher
    if (!weights_generator_) {
        spdlog::error("The ANN searcher is not set.");
    }

    // Generate weights.
    spdlog::info("Generating weights...");
    std::vector<double> weights = weights_generator_->Generate(from, to, in_memory, use_cache_);

    // Check the size of weights.
    if (weights.size() != from.num_points) {
        spdlog::error("Weights size does not match the number of query points in the dataset");
    }

    // Sample the points using the generated weights.
    double approximation = 0.0;
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);

    std::unique_ptr<PointSet> from_set;
    if (in_memory) {
        from_set = std::make_unique<InMemoryPointSet>(from);
    } else {
        from_set = std::make_unique<DiskPointSet>(from);
    }

    if (num_samples_ == 0) {
        num_samples_ = static_cast<unsigned int>(std::log(from_set->get_num_points()));
        spdlog::info("Number of samples set to log(n): {}", num_samples_);
    }

    LinearScanAnnSearcher linear_scan_ann_searcher;
    linear_scan_ann_searcher.Init(to, in_memory);
    for (unsigned int i = 0; i < num_samples_; i++) {
        unsigned int point_id = Utils::SampleFromWeights(weights);
        Point query = from_set->GetPoint(point_id);
        AnnResult result = linear_scan_ann_searcher.Search(query);
        approximation += sum * result.distance / weights[point_id];
    }

    return approximation / num_samples_;
}