#ifndef COMMAND_H_
#define COMMAND_H_

#include <filesystem>
#include <memory>
#include <random>

#include "estimator.h"

class Command {
   public:
    virtual ~Command() = default;
    virtual void Execute() = 0;
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