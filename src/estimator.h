#ifndef ESTIMATOR_H_
#define ESTIMATOR_H_

#include <filesystem>
#include <memory>

#include "ann_searcher.h"
#include "point_set.h"

struct EstimateResult {
    double chamfer_distance{0.0};
    double relative_error{0.0};
};

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class Estimator {
   public:
    virtual ~Estimator() = default;

    virtual void set_in_memory(bool in_memory) = 0;

    virtual EstimateResult Estimate(const std::filesystem::path& dataset_directory) = 0;
};

class AnnEstimator : public Estimator {
   public:
    AnnEstimator(std::unique_ptr<AnnSearcher> ann_searcher);

    void set_in_memory(bool in_memory) override;

    EstimateResult Estimate(const std::filesystem::path& dataset_directory) override;

   private:
    double CalculateDistance(const PointSetMetadata& from, const PointSetMetadata& to);

    std::unique_ptr<AnnSearcher> ann_searcher_;
    bool in_memory_{false};
};

#endif