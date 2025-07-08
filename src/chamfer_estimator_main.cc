#include <CLI/CLI.hpp>
#include <format>

#include "chamfer_estimator.h"

namespace fs = std::filesystem;

auto main(int argc, char** argv) -> int {
    CLI::App app("Chamfer Approximation for sets");
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

    unsigned int num_samples{0};
    app.add_option("-s,--num_samples", num_samples,
                   "Number of samples to use for approximation (default: 0, which means log(N) samples will be used)");

    bool verbose{false};
    app.add_flag("-v,--verbose", verbose, "Enable verbose output");

    try {
        CLI11_PARSE(app, argc, argv);

        auto chamfer_estimator = qalsh_chamfer::ChamferEstimatorBuilder()
                                     .set_dataset_name(dataset_name)
                                     .set_parent_directory(parent_directory)
                                     .set_num_points(num_points)
                                     .set_num_dimensions(num_dimensions)
                                     .ReadParamFromBinaryFile()
                                     .set_num_samples(num_samples)
                                     .set_verbose(verbose)
                                     .Build();

        if (verbose) {
            chamfer_estimator->PrintConfiguration();
        }
        chamfer_estimator->Execute();

    } catch (const std::exception& e) {
        std::cerr << std::format("Error: {}\n", e.what());
        return 1;
    }

    return 0;
}