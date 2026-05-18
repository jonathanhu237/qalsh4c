#pragma once

#include <filesystem>
#include <memory>

#include "config.h"
#include "dataset.h"
#include "estimator.h"

class Command {
   public:
    virtual ~Command() = default;
    virtual void Execute() = 0;
};

class EstimateCommand : public Command {
   public:
    EstimateCommand(std::unique_ptr<Estimator> estimator, double norm_order, std::filesystem::path dataset_directory,
                    ChamferDirection direction);
    void Execute() override;

   private:
    std::unique_ptr<Estimator> estimator_;
    double norm_order_;
    std::filesystem::path dataset_directory_;
    ChamferDirection direction_;
};

class DatasetGenerateGmmCommand : public Command {
   public:
    explicit DatasetGenerateGmmCommand(DatasetPrep::GmmOptions options);
    void Execute() override;

   private:
    DatasetPrep::GmmOptions options_;
};

class DatasetConvertCommand : public Command {
   public:
    explicit DatasetConvertCommand(DatasetPrep::ConvertOptions options);
    void Execute() override;

   private:
    DatasetPrep::ConvertOptions options_;
};

class DatasetGroundTruthCommand : public Command {
   public:
    explicit DatasetGroundTruthCommand(DatasetPrep::GroundTruthOptions options);
    void Execute() override;

   private:
    DatasetPrep::GroundTruthOptions options_;
};
