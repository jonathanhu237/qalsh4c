#include <CLI11/CLI11.hpp>
#include <exception>
#include <format>
#include <iostream>

#include "constants.h"
#include "qalsh_chamfer.h"

auto main(int argc, char **argv) -> int {
    CLI::App app{"Approximate Chamfer Distance using QALSH"};
    argv = app.ensure_utf8(argv);

    std::string dataset_name;
    app.add_option("-n,--name", dataset_name, "Name of the dataset")->required();

    std::string parent_directory;
    app.add_option("-p,--parent_directory", parent_directory, "Parent directory for the dataset")
        ->default_val(qalsh_chamfer::kDefaultParentDirectory);

    unsigned int num_points{0};
    app.add_option("-N,--num_points", num_points, "Number of points within each set in the dataset")->required();

    unsigned int num_dimensions{0};
    app.add_option("-D,--num_dimensions", num_dimensions, "Number of dimensions for the dataset")->required();

    bool verbose{false};
    app.add_flag("-v,--verbose", verbose, "Enable verbose output");

    try {
        CLI11_PARSE(app, argc, argv);

        auto qalsh_chamfer = qalsh_chamfer::QalshChamferBuilder()
                                 .set_dataset_name(dataset_name)
                                 .set_parent_directory(parent_directory)
                                 .set_num_points(num_points)
                                 .set_num_dimensions(num_dimensions)
                                 .set_verbose(verbose)
                                 .Build();

        if (verbose) {
            qalsh_chamfer->PrintConfiguration();
        }
        qalsh_chamfer->Execute();

    } catch (const std::exception &e) {
        std::cerr << std::format("Error: {}\n", e.what());
        return 1;
    }

    return 0;
}