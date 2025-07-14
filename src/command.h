#ifndef COMMAND_H_
#define COMMAND_H_

#include <filesystem>
#include <memory>
#include <random>

#include "indexer.h"
#include "types.h"

class ICommand {
   public:
    virtual ~ICommand() = default;
    virtual auto Execute() -> void = 0;
};

class GenerateDatasetCommand : public ICommand {
   public:
    GenerateDatasetCommand(std::string data_type_, std::filesystem::path dataset_directory,
                           unsigned int base_num_points, unsigned int query_num_points, unsigned int num_dimensions,
                           double left_boundary, double right_boundary);
    auto Execute() -> void override;

   private:
    auto PrintConfiguration() -> void;
    auto GeneratePointSet(const std::filesystem::path& dataset_directory, const std::string& point_set_name,
                          unsigned int num_points) -> void;

    std::filesystem::path dataset_directory_;
    DatasetMetadata dataset_metadata_;

    double left_boundary_{0.0};
    double right_boundary_{0.0};

    std::mt19937 gen_;
};

class IndexCommand : public ICommand {
   public:
    explicit IndexCommand(std::unique_ptr<Indexer> indexer);
    auto Execute() -> void override;

   private:
    std::unique_ptr<Indexer> indexer_;
};

#endif