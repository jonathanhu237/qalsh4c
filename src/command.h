#ifndef COMMAND_H_
#define COMMAND_H_

#include <filesystem>
#include <memory>

#include "dataset_generator.h"
#include "estimator.h"
#include "indexer.h"

class Command {
   public:
    virtual ~Command() = default;
    virtual void Execute() = 0;
};

class GenerateDatasetCommand : public Command {
   public:
    GenerateDatasetCommand(std::unique_ptr<DatasetGenerator> dataset_generator,
                           std::filesystem::path dataset_directory);
    void Execute() override;

   private:
    std::unique_ptr<DatasetGenerator> dataset_generator_;
    std::filesystem::path dataset_directory_;
};

class IndexCommand : public Command {
   public:
    explicit IndexCommand(std::unique_ptr<Indexer> indexer);
    void Execute() override;

   private:
    std::unique_ptr<Indexer> indexer_;
};

class EstimateCommand : public Command {
   public:
    explicit EstimateCommand(std::unique_ptr<Estimator> estimator, std::filesystem::path dataset_directory);
    void Execute() override;

   private:
    std::unique_ptr<Estimator> estimator_;
    std::filesystem::path dataset_directory_;
};

#endif