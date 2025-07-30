#ifndef COMMAND_H_
#define COMMAND_H_

#include <filesystem>
#include <memory>

#include "estimator.h"

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class Command {
   public:
    virtual ~Command() = default;
    virtual void Execute() = 0;
};

class EstimateCommand : public Command {
   public:
    explicit EstimateCommand(std::unique_ptr<Estimator> estimator, std::filesystem::path dataset_directory,
                             bool in_memory);
    void Execute() override;

   private:
    std::unique_ptr<Estimator> estimator_;
    std::filesystem::path dataset_directory_;
    bool in_memory_;
};

#endif