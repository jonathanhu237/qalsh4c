#include <CLI/CLI.hpp>
#include <exception>
#include <format>
#include <iostream>
#include <memory>
#include <string>

#include "command.h"
#include "constants.h"
#include "data_type.h"
#include "generate_dataset_command.h"

namespace fs = std::filesystem;

auto main(int argc, char** argv) -> int {
    CLI::App app{"QALSH for Chamfer Distance Approximation"};
    app.require_subcommand(1);

    std::unique_ptr<qalsh_chamfer::Command> command{nullptr};

    // Generate dataset
    CLI::App* generate_dataset_cmd =
        app.add_subcommand("generate", "Generate a Dataset for Chamfer Distance Approximation");

    std::string dataset_name;
    generate_dataset_cmd->add_option("-n,--dataset-name", dataset_name, "Name of the dataset")->required();

    fs::path data_directory;
    generate_dataset_cmd->add_option("-d,--data-directory", data_directory, "Data directory")
        ->default_val(fs::path("data"));

    unsigned int num_points{0};
    generate_dataset_cmd->add_option("-N,--num-points", num_points, "Number of points within each set in the dataset")
        ->required();

    unsigned int num_dimensions{0};
    generate_dataset_cmd->add_option("-D,--num-dimensions", num_dimensions, "Number of dimensions for the dataset")
        ->required();

    double left_boundary{0.0};
    generate_dataset_cmd->add_option("-l,--left-boundary", left_boundary, "Left boundary for generated points")
        ->default_val(qalsh_chamfer::kDefaultLeftBoundary);

    double right_boundary{0.0};
    generate_dataset_cmd->add_option("-r,--right-boundary", right_boundary, "Right boundary for generated points")
        ->default_val(qalsh_chamfer::kDefaultLeftBoundary);

    std::string data_type;
    generate_dataset_cmd->add_option("-t,--data-type", data_type, "The type of the dataset")
        ->required()
        ->transform(CLI::CheckedTransformer(qalsh_chamfer::DataTypeMap, CLI::ignore_case));

    bool verbose{false};
    generate_dataset_cmd->add_flag("-v,--verbose", verbose, "Enable verbose output");

    generate_dataset_cmd->callback([&]() {
        command = std::make_unique<qalsh_chamfer::GenerateDatasetCommand>(
            dataset_name, data_directory, num_points, num_dimensions, left_boundary, right_boundary, verbose);
    });

    try {
        CLI11_PARSE(app, argc, argv);

        if (command) {
            command->Execute();
        }
    } catch (std::exception& e) {
        std::cerr << std::format("Error: {}\n", e.what());
        return 1;
    }

    return 0;
}