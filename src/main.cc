#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <exception>
#include <format>
#include <memory>

#include "command.h"
#include "constants.h"
#include "indexer.h"
#include "types.h"

auto main(int argc, char** argv) -> int {
    CLI::App app{"Fast Chamfer Distance Approximation via Query-Aware Locality-Sensitive Hashing (QALSH)."};

    app.add_flag_callback(
        "-v,--verbose", []() { spdlog::set_level(spdlog::level::debug); }, "Enable verbose (debug) logging");
    app.require_subcommand(1);

    std::unique_ptr<ICommand> command;

    // ------------------------------
    // generate dataset
    // ------------------------------

    CLI::App* generate_cmd =
        app.add_subcommand("generate_dataset", "Generate a dataset for Chamfer Distance Approximation");

    std::filesystem::path dataset_directory;
    generate_cmd->add_option("-d,--dataset_directory", dataset_directory, "Directory for the dataset")->required();

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

    bool in_memory{false};
    generate_cmd->add_flag("-i,--in_memory", in_memory, "Generate the dataset in memory (default: false)")
        ->default_val(false);

    generate_cmd->callback([&]() {
        command = std::unique_ptr<ICommand>(new GenerateDatasetCommand(data_type, dataset_directory, base_num_points,
                                                                       query_num_points, num_dimensions, left_boundary,
                                                                       right_boundary, in_memory));
    });

    // ------------------------------
    // index
    // ------------------------------

    CLI::App* index_cmd = app.add_subcommand("index", "Index a dataset for Chamfer Distance Approximation");
    index_cmd->require_subcommand(1);

    std::unique_ptr<Indexer> indexer;

    // ------------------------------
    // index qalsh
    // ------------------------------

    CLI::App* index_qalsh_cmd = index_cmd->add_subcommand("qalsh", "Index a dataset using QALSH algorithm");

    index_cmd->add_option("-d,--dataset_directory", dataset_directory, "Directory for the dataset")->required();

    double approximation_ratio{0.0};
    index_qalsh_cmd->add_option("-c, --approximation_ratio", approximation_ratio, "Approximation ratio for QALSH")
        ->default_val(2.0);

    double bucket_width{0.0};
    index_qalsh_cmd
        ->add_option("-w, --bucket_width", bucket_width,
                     "Bucket width for the indexer (default: 0, which means the bucket width will be 2\\sqrt(c))")
        ->default_val(0.0);

    double beta{0.0};
    index_qalsh_cmd
        ->add_option("-b, --beta", beta,
                     "Beta parameter for the indexer (default: 0, which means the beta will be 100/num_points)")
        ->default_val(0.0);

    double error_probability{0.0};
    index_qalsh_cmd
        ->add_option("-e, --error_probability", error_probability, "Error probability for the indexer (default: 1/e)")
        ->default_val(Constants::kDefaultErrorProbability)
        ->default_str("1/e");

    unsigned int num_hash_tables{0};
    index_qalsh_cmd
        ->add_option("-m,--num_hash_tables", num_hash_tables,
                     "Number of hash tables to use for indexing (default: 0, which means the number will be determined "
                     "automatically)")
        ->default_val(0);

    unsigned int collision_threshold{0};
    index_qalsh_cmd
        ->add_option("-l,--collision_threshold", collision_threshold,
                     "Collision threshold for the indexer (default: 0, which means the collision threshold will be "
                     "determined automatically)")
        ->default_val(0);

    unsigned int page_size{0};
    index_qalsh_cmd
        ->add_option("-B,--page_size", page_size,
                     std::format("Page size for the indexer (default: {} bytes)", Constants::kDefaultPageSize))
        ->default_val(Constants::kDefaultPageSize);

    index_qalsh_cmd->add_flag("-i,--in_memory", in_memory, "Index the dataset in memory (default: false)")
        ->default_val(false);

    index_qalsh_cmd->callback([&]() {
        QalshConfiguration qalsh_config = {.approximation_ratio = approximation_ratio,
                                           .bucket_width = bucket_width,
                                           .beta = beta,
                                           .error_probability = error_probability,
                                           .num_hash_tables = num_hash_tables,
                                           .collision_threshold = collision_threshold,
                                           .page_size = page_size};
        indexer = std::make_unique<QalshIndexer>(dataset_directory, qalsh_config, in_memory);
    });

    index_cmd->callback([&]() {
        if (!indexer) {
            throw std::runtime_error("Indexer is not set. Please specify an indexer.");
        }
        command = std::unique_ptr<ICommand>(new IndexCommand(std::move(indexer)));
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