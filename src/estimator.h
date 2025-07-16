#ifndef ESTIMATOR_H_
#define ESTIMATOR_H_

#include <filesystem>

#include "types.h"

class Estimator {
   public:
    virtual ~Estimator() = default;
    virtual double Estimate(const std::filesystem::path& dataset_directory) = 0;
};

class LinearScanEstimator : public Estimator {
   public:
    LinearScanEstimator() = default;
    double Estimate(const std::filesystem::path& dataset_directory) override;

   private:
    void PrintConfiguration() const;

    std::filesystem::path dataset_directory_;
    DatasetMetadata dataset_metadata_;
};

#endif