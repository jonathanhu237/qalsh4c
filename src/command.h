#pragma once

#include <filesystem>
#include <memory>

#include "config.h"
#include "dataset.h"
#include "disk_qalsh_index.h"
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

class QalshIndexCommand : public Command {
   public:
    explicit QalshIndexCommand(DiskQalsh::BuildRequest request);
    void Execute() override;

   private:
    DiskQalsh::BuildRequest request_;
};

struct StaticIndexRequest {
    std::filesystem::path dataset_directory;
    float norm_order{0.0F};
    DiskQalsh::TargetSelection target_selection{DiskQalsh::TargetSelection::kB};
    unsigned int num_projections{0};  // 0 selects the default m for the indexed set's size
    unsigned int num_repetitions{0};
    unsigned int seed{0};
    bool overwrite{false};
};

class StaticIndexCommand : public Command {
   public:
    explicit StaticIndexCommand(StaticIndexRequest request);
    void Execute() override;

   private:
    StaticIndexRequest request_;
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
