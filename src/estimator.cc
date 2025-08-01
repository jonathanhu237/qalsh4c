#include "estimator.h"

#include <spdlog/spdlog.h>

#include <memory>
#include <numeric>
#include <utility>

#include "ann_searcher.h"
#include "point_set.h"
#include "utils.h"

// --------------------------------------------------
// AnnEstimator Implementation
// --------------------------------------------------
AnnEstimator::AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher) : ann_searcher_(std::move(ann_searcher)) {}

double AnnEstimator::EstimateDistance(const PointSetMetadata& from, const PointSetMetadata& to, bool in_memory) {
    // Check the ANN searcher
    if (!ann_searcher_) {
        spdlog::error("The ANN searcher is not set.");
    }

    ann_searcher_->Reset();
    ann_searcher_->Init(to, in_memory);
    double distance = 0;

    std::unique_ptr<PointSet> query_set;
    if (in_memory) {
        query_set = std::make_unique<InMemoryPointSet>(from);
    } else {
        query_set = std::make_unique<DiskPointSet>(from);
    }

    for (unsigned int point_id = 0; point_id < query_set->get_num_points(); point_id++) {
        Point query_point = query_set->GetPoint(point_id);
        distance += ann_searcher_->Search(query_point).distance;
    }

    return distance;
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