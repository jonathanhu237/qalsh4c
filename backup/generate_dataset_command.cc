#include "generate_dataset_command.h"

#include <format>
#include <iostream>

namespace qalsh_chamfer {

GenerateDatasetCommand::GenerateDatasetCommand(std::string dataset_name, fs::path parent_directory,
                                               unsigned int num_points, unsigned int num_dimensions, int left_boundary,
                                               int right_boundary, bool verbose)
    : dataset_name_(std::move(dataset_name)),
      parent_directory_(std::move(parent_directory)),
      num_points_(num_points),
      num_dimensions_(num_dimensions),
      left_boundary_(left_boundary),
      right_boundary_(right_boundary),
      verbose_(verbose) {}

auto GenerateDatasetCommand::Execute() -> void {
    if (verbose_) {
        PrintConfiguration();
    }
}

auto GenerateDatasetCommand::PrintConfiguration() -> void {
    std::cout << std::format("---------- Dataset Generator Configuration ----------\n");
    std::cout << std::format("Dataset Name: {}\n", dataset_name_);
    std::cout << std::format("Parent Directory: {}\n", parent_directory_.string());
    std::cout << std::format("Number of Points: {}\n", num_points_);
    std::cout << std::format("Number of Dimensions: {}\n", num_dimensions_);
    std::cout << std::format("Left Boundary: {}\n", left_boundary_);
    std::cout << std::format("Right Boundary: {}\n", right_boundary_);
    std::cout << std::format("-----------------------------------------------------\n");
}

}  // namespace qalsh_chamfer