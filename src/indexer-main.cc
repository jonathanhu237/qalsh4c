#include <CLI11/CLI11.hpp>
#include <cmath>
#include <exception>
#include <format>
#include <string>

#include "const.h"
#include "indexer.h"

auto main(int argc, char** argv) -> int {
    CLI::App app{"Indexer for Approximating Chamfer Distance using QALSH"};
    argv = app.ensure_utf8(argv);

    std::string dataset_name;
    app.add_option("-n,--dataset-name", dataset_name, "Name of the dataset to be indexed")->required();

    std::string parent_directory;
    app.add_option("-p,--parent-directory", parent_directory, "Parent directory for the dataset")
        ->default_val(qalsh_chamfer::kDefaultParentDirectory);

    unsigned int num_points{0};
    app.add_option("-N,--num_points", num_points, "Number of points within each set in the dataset")->required();

    unsigned int num_dimensions{0};
    app.add_option("-D,--num_dimensions", num_dimensions, "Number of dimensions for the dataset")->required();

    float beta{0.0};
    app.add_option("-b, --beta", beta,
                   "Beta parameter for the indexer (default: 0.0, which means the beta will be 100/num_points)")
        ->default_val(0.0);

    float error_probability{0.0};
    app.add_option("-e, --error_probability", error_probability, "Error probability for the indexer (default: 1/e)")
        ->default_val(qalsh_chamfer::kDefaultErrorProbability);

    unsigned int num_hash_tables{0};
    app.add_option("-m,--num_hash_tables", num_hash_tables,
                   "Number of hash tables to use for indexing (default: 0, which means the number will be determined "
                   "automatically)")
        ->default_val(0);

    unsigned int page_size{0};
    const unsigned int kDefaultPageSize = 4096;
    app.add_option("-B,--page_size", page_size, "Page size for the indexer (default: 4096 bytes)")
        ->default_val(kDefaultPageSize);

    bool verbose{false};
    app.add_flag("-v,--verbose", verbose, "Enable verbose output");

    try {
        CLI11_PARSE(app, argc, argv);

        auto indexer = qalsh_chamfer::IndexerBuilder()
                           .set_dataset_name(dataset_name)
                           .set_parent_directory(parent_directory)
                           .set_num_points(num_points)
                           .set_num_dimensions(num_dimensions)
                           .set_beta(beta)
                           .set_error_probability(error_probability)
                           .set_num_hash_tables(num_hash_tables)
                           .set_page_size(page_size)
                           .set_verbose(verbose)
                           .Build();

        if (verbose) {
            indexer->PrintConfiguration();
        }
        indexer->Execute();

    } catch (const std::exception& e) {
        std::cerr << std::format("Error: {}\n", e.what());
        return 1;
    }

    return 0;
}