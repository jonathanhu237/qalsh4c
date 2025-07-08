#include <CLI/CLI.hpp>
#include <cmath>
#include <exception>
#include <filesystem>
#include <format>
#include <string>

#include "constants.h"
#include "indexer.h"

namespace fs = std::filesystem;

auto main(int argc, char** argv) -> int {
    CLI::App app{"Indexer for Approximating Chamfer Distance using QALSH"};
    argv = app.ensure_utf8(argv);

    std::string dataset_name;
    app.add_option("-n,--dataset-name", dataset_name, "Name of the dataset to be indexed")->required();

    fs::path parent_directory;
    app.add_option("-p,--parent-directory", parent_directory, "Parent directory for the dataset")
        ->default_val(fs::path("data"));

    unsigned int num_points{0};
    app.add_option("-N,--num_points", num_points, "Number of points within each set in the dataset")->required();

    unsigned int num_dimensions{0};
    app.add_option("-D,--num_dimensions", num_dimensions, "Number of dimensions for the dataset")->required();

    double approximation_ratio{0.0};
    app.add_option("-c, --approximation_ratio", approximation_ratio, "Approximation ratio for QALSH")->default_val(2.0);

    double bucket_width{0.0};
    app.add_option("-w, --bucket_width", bucket_width,
                   "Bucket width for the indexer (default: 0, which means the bucket width will be 2\\sqrt(c))")
        ->default_val(0.0);

    double beta{0.0};
    app.add_option("-b, --beta", beta,
                   "Beta parameter for the indexer (default: 0, which means the beta will be 100/num_points)")
        ->default_val(0.0);

    double error_probability{0.0};
    app.add_option("-e, --error_probability", error_probability, "Error probability for the indexer (default: 1/e)")
        ->default_val(qalsh_chamfer::kDefaultErrorProbability)
        ->default_str("1/e");

    unsigned int num_hash_tables{0};
    app.add_option("-m,--num_hash_tables", num_hash_tables,
                   "Number of hash tables to use for indexing (default: 0, which means the number will be determined "
                   "automatically)")
        ->default_val(0);

    unsigned int collision_threshold{0};
    app.add_option("-l,--collision_threshold", collision_threshold,
                   "Collision threshold for the indexer (default: 0, which means the collision threshold will be "
                   "determined automatically)")
        ->default_val(0);

    unsigned int page_size{0};
    app.add_option("-B,--page_size", page_size,
                   std::format("Page size for the indexer (default: {} bytes)", qalsh_chamfer::kDefaultPageSize))
        ->default_val(qalsh_chamfer::kDefaultPageSize);

    bool verbose{false};
    app.add_flag("-v,--verbose", verbose, "Enable verbose output");

    try {
        CLI11_PARSE(app, argc, argv);

        auto indexer = qalsh_chamfer::IndexerBuilder()
                           .set_dataset_name(dataset_name)
                           .set_parent_directory(parent_directory)
                           .set_num_points(num_points)
                           .set_num_dimensions(num_dimensions)
                           .set_approximation_ratio(approximation_ratio)
                           .set_bucket_width(bucket_width)
                           .set_beta(beta)
                           .set_error_probability(error_probability)
                           .set_collision_schema_param(num_hash_tables, collision_threshold)
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