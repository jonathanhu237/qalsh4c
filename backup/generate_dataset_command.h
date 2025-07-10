#ifndef GENERATE_DATASET_COMMAND_H_
#define GENERATE_DATASET_COMMAND_H_

#include <filesystem>

#include "command.h"

namespace fs = std::filesystem;

namespace qalsh_chamfer {

class GenerateDatasetCommand : public Command {
   public:
    GenerateDatasetCommand(std::string dataset_name, fs::path parent_directory, unsigned int num_points,
                           unsigned int num_dimensions, int left_boundary, int right_boundary, bool verbose);
    auto Execute() -> void override;

   private:
    auto PrintConfiguration() -> void;

    std::string dataset_name_;
    fs::path parent_directory_;
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    int left_boundary_{0};
    int right_boundary_{0};
    bool verbose_{false};
};

}  // namespace qalsh_chamfer

#endif