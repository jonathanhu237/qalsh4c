#include <CLI11/CLI11.hpp>
#include <format>

#include "crude_nn.h"

namespace fs = std::filesystem;

auto main(int argc, char** argv) -> int {
    CLI::App app("Generate Crude Nearest Neighbors for Sets");
    argv = app.ensure_utf8(argv);

    std::string dataset_name;
    app.add_option("-n,--name", dataset_name, "Name of the dataset")->required();

    fs::path parent_directory;
    app.add_option("-p,--parent_directory", parent_directory, "Parent directory for the dataset")
        ->default_val(fs::path("data"));

    unsigned int num_points{0};
    app.add_option("-N,--num_points", num_points, "Number of points within each set in the dataset")->required();

    unsigned int num_dimensions{0};
    app.add_option("-D,--num_dimensions", num_dimensions, "Number of dimensions for the dataset")->required();

    bool verbose{false};
    app.add_flag("-v,--verbose", verbose, "Enable verbose output");

    try {
        CLI11_PARSE(app, argc, argv);

        auto crude_nn = qalsh_chamfer::CrudeNnBuilder()
                            .set_dataset_name(dataset_name)
                            .set_parent_directory(parent_directory)
                            .set_num_points(num_points)
                            .set_num_dimensions(num_dimensions)
                            .set_verbose(verbose)
                            .Build();

        if (verbose) {
            crude_nn->PrintConfiguration();
        }
        crude_nn->Execute();

    } catch (const std::exception& e) {
        std::cerr << std::format("Error: {}\n", e.what());
        return 1;
    }

    return 0;
}