#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <cstdint>
#include <exception>
#include <format>
#include <memory>

#include "command.h"
#include "constants.h"
#include "generate_dataset_command.h"

auto main(int argc, char** argv) -> int {
    CLI::App app{"Fast Chamfer Distance Approximation via Query-Aware Locality-Sensitive Hashing (QALSH)."};

    app.add_flag_callback(
        "-v,--verbose", []() { spdlog::set_level(spdlog::level::debug); }, "Enable verbose (debug) logging");
    app.require_subcommand(1);

    std::unique_ptr<Command> command;
    CLI::App* generate_cmd = app.add_subcommand("generate", "Generate a dataset for Chamfer Distance Approximation");

    std::string parent_directory;
    generate_cmd->add_option("-p,--parent-directory", parent_directory, "Parent directory for the dataset")
        ->default_val("data");

    std::string dataset_name;
    generate_cmd->add_option("-n,--dataset-name", dataset_name, "Name of the dataset")->required();

    std::string data_type;
    generate_cmd->add_option("-t,--data-type", data_type, "Data type for the dataset (uint8, int, double)")
        ->default_val("double");

    unsigned int base_num_points{0};
    generate_cmd->add_option("-B,--base_num_points", base_num_points, "Number of points within base set in the dataset")
        ->required();

    unsigned int query_num_points{0};
    generate_cmd
        ->add_option("-Q,--query_num_points", query_num_points, "Number of points within query set in the dataset")
        ->required();

    unsigned int num_dimensions{0};
    generate_cmd->add_option("-D,--num_dimensions", num_dimensions, "Number of dimensions for the dataset")->required();

    double left_boundary{0.0};
    generate_cmd->add_option("-l,--left_boundary", left_boundary, "Left boundary for generated points")
        ->default_val(Constants::kDefaultLeftBoundary);

    double right_boundary{0.0};
    generate_cmd->add_option("-r,--right_boundary", right_boundary, "Right boundary for generated points")
        ->default_val(Constants::kDefaultRightBoundary);

    generate_cmd->callback([&]() {
        if (data_type == "uint8") {
            command = std::make_unique<GenerateDatasetCommand<uint8_t>>(parent_directory, dataset_name, base_num_points,
                                                                        query_num_points, num_dimensions, left_boundary,
                                                                        right_boundary);
        } else if (data_type == "int") {
            command = std::make_unique<GenerateDatasetCommand<int>>(parent_directory, dataset_name, base_num_points,
                                                                    query_num_points, num_dimensions, left_boundary,
                                                                    right_boundary);
        } else if (data_type == "double") {
            command = std::make_unique<GenerateDatasetCommand<double>>(parent_directory, dataset_name, base_num_points,
                                                                       query_num_points, num_dimensions, left_boundary,
                                                                       right_boundary);
        } else {
            throw std::invalid_argument("Unsupported data type specified.");
        }
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