#ifndef ESTIMATOR_H_
#define ESTIMATOR_H_

#include <filesystem>

class Estimator {
   public:
    virtual ~Estimator() = default;
    virtual double Estimate(std::filesystem::path dataset_directory) = 0;
};

#endif