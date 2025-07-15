#ifndef COMMAND_H_
#define COMMAND_H_

#include <memory>

#include "dataset_generator.h"
#include "estimator.h"
#include "indexer.h"

class Command {
   public:
    virtual ~Command() = default;
    virtual auto Execute() -> void = 0;
};

class GenerateDatasetCommand : public Command {
   public:
    GenerateDatasetCommand(std::unique_ptr<DatasetGenerator> dataset_generator);
    auto Execute() -> void override;

   private:
    std::unique_ptr<DatasetGenerator> dataset_generator_;
};

class IndexCommand : public Command {
   public:
    explicit IndexCommand(std::unique_ptr<Indexer> indexer);
    auto Execute() -> void override;

   private:
    std::unique_ptr<Indexer> indexer_;
};

class EstimateCommand : public Command {
   public:
    explicit EstimateCommand(std::unique_ptr<Estimator> estimator);
    auto Execute() -> void override;

   private:
    std::unique_ptr<Estimator> estimator_;
};

#endif