#include <CLI11/CLI11.hpp>
#include <exception>
#include <format>

#include "const.h"
#include "dataset_generator.h"

auto main(int argc, char** argv) -> int {
    CLI::App app("Dataset Generator for Chamfer Distance Approximation");
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

    int left_boundary{0};
    app.add_option("-l,--left_boundary", left_boundary, "Left boundary for generated points")
        ->default_val(qalsh_chamfer::kDefaultLeftBoundary);

    int right_boundary{0};
    app.add_option("-r,--right_boundary", right_boundary, "Right boundary for generated points")
        ->default_val(qalsh_chamfer::kDefaultRightBoundary);

    bool verbose{false};
    app.add_flag("-v,--verbose", verbose, "Enable verbose output");

    try {
        CLI11_PARSE(app, argc, argv);

        auto dataset_generator = qalsh_chamfer::DatasetGeneratorBuilder()
                                     .set_dataset_name(dataset_name)
                                     .set_parent_directory(parent_directory)
                                     .set_num_points_(num_points)
                                     .set_num_dimensions(num_dimensions)
                                     .set_left_boundary(left_boundary)
                                     .set_right_boundary(right_boundary)
                                     .set_verbose(verbose)
                                     .Build();

        if (verbose) {
            dataset_generator->PrintConfiguration();
        }
        dataset_generator->Execute();

    } catch (const std::exception& e) {
        std::cerr << std::format("Error: {}\n", e.what());
        return 1;
    }

    return 0;
}