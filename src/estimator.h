#ifndef ESTIMATOR_H_
#define ESTIMATOR_H_

#include <filesystem>

class Estimator {
   public:
    virtual ~Estimator() = default;
    virtual double Estimate(const std::filesystem::path& dataset_directory) = 0;
};

class AnnEstimator : public Estimator {
   public:
    AnnEstimator(std::string searcher_type);
    double Estimate(const std::filesystem::path& dataset_directory) override;

   private:
    std::string searcher_type_;
};

#endif