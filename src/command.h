#ifndef COMMAND_H_
#define COMMAND_H_

#include <filesystem>
#include <memory>
#include <random>

#include "dataset_metadata.h"
#include "estimator.h"
#include "point_set.h"
#include "qalsh_config.h"

class Command {
   public:
    virtual ~Command() = default;
    virtual void Execute() = 0;
};

class GenerateDatasetCommand : public Command {
   public:
    GenerateDatasetCommand(DatasetMetadata dataset_metadata, double left_boundary, double right_boundary,
                           std::filesystem::path output_directory, bool in_memory);
    void Execute() override;

   private:
    void GeneratePointSet(const std::string& point_set_file_path, unsigned int num_points);

    DatasetMetadata dataset_metadata_;
    double left_boundary_;
    double right_boundary_;
    std::filesystem::path output_directory_;
    bool in_memory_;

    std::mt19937 gen_;
};

class IndexCommand : public Command {
   public:
    IndexCommand(double approximation_ratio, unsigned int page_size, std::filesystem::path dataset_directory);
    void Execute() override;

   private:
    void BuildIndex(const PointSetMetadata& point_set_metadata, const std::filesystem::path& index_directory);

    QalshConfig qalsh_config_;
    std::filesystem::path dataset_directory_;

    std::mt19937 gen_;
};

class EstimateCommand : public Command {
   public:
    EstimateCommand(std::unique_ptr<Estimator> estimator, std::filesystem::path dataset_directory, bool in_memory);
    void Execute() override;

   private:
    std::unique_ptr<Estimator> estimator_;
    std::filesystem::path dataset_directory_;
    bool in_memory_;
};

#endif