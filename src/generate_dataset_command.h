#ifndef GENERATE_DATASET_COMMAND_H_
#define GENERATE_DATASET_COMMAND_H_

#include <spdlog/spdlog.h>

#include <filesystem>
#include <string>

#include "command.h"
#include "utils.h"

template <typename T>
class GenerateDatasetCommand : public Command {
   public:
    GenerateDatasetCommand(std::filesystem::path parent_directory, std::string dataset_name, unsigned int num_points,
                           unsigned int num_dimensions, double left_boundary, double right_boundary);
    auto Execute() -> void override;

   private:
    auto PrintConfiguration() -> void;

    std::filesystem::path parent_directory_;
    std::string dataset_name_;
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    double left_boundary_{0.0};
    double right_boundary_{0.0};
};

template <typename T>
GenerateDatasetCommand<T>::GenerateDatasetCommand(std::filesystem::path parent_directory, std::string dataset_name,
                                                  unsigned int num_points, unsigned int num_dimensions,
                                                  double left_boundary, double right_boundary)
    : parent_directory_(std::move(parent_directory)),
      dataset_name_(std::move(dataset_name)),
      num_points_(num_points),
      num_dimensions_(num_dimensions),
      left_boundary_(left_boundary),
      right_boundary_(right_boundary) {}

template <typename T>
auto GenerateDatasetCommand<T>::Execute() -> void {
    PrintConfiguration();
    spdlog::info("Begin to generate dataset...");
}

template <typename T>
auto GenerateDatasetCommand<T>::PrintConfiguration() -> void {
    spdlog::debug(R"(The configuration is as follows:
---------- Dataset Generator Configuration ----------
Data Type: {}
Data Directory: {}
Dataset Name: {}
Number of Points: {}
Number of Dimensions: {}
Left Boundary: {}
Right Boundary: {}
-----------------------------------------------------)",
                  Utils::to_string<T>(), parent_directory_.string(), dataset_name_, num_points_, num_dimensions_,
                  left_boundary_, right_boundary_);
}

#endif