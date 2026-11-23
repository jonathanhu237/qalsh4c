#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "config.h"
#include "estimator.h"

class Command {
   public:
    virtual ~Command() = default;
    virtual void Execute() = 0;
};

class EstimateCommand : public Command {
   public:
    EstimateCommand(std::unique_ptr<Estimator> estimator, float norm_order, std::filesystem::path dataset_directory,
                    ChamferDirection direction);
    void Execute() override;

   private:
    std::unique_ptr<Estimator> estimator_;
    float norm_order_;
    std::filesystem::path dataset_directory_;
    ChamferDirection direction_;
};

// Prints the certified bound on kappa and the sample sizes T it gives (Theorems 1 and 2) for the QALSH parameters that
// a norm and an approximation ratio determine, under the current stopping rule.
class KappaCommand : public Command {
   public:
    KappaCommand(float norm_order, float approximation_ratio, std::vector<double> epsilons, double delta);
    void Execute() override;

   private:
    float norm_order_;
    float approximation_ratio_;
    std::vector<double> epsilons_;
    double delta_;
};
