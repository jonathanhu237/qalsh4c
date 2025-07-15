#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <format>
#include <memory>

#include "command.h"
#include "constants.h"
#include "dataset_generator.h"
#include "indexer.h"
#include "types.h"

int main(int argc, char** argv) {
    CLI::App app{"Fast Chamfer Distance Approximation via Query-Aware Locality-Sensitive Hashing (QALSH)."};

    std::string log_level;
    app.add_option("-l,--log_level", log_level, "Set the logging level (default: info)")
        ->default_val("info")
        ->check(CLI::IsMember({"debug", "info"}))
        ->each([&](const std::string& level) { spdlog::set_level(spdlog::level::from_str(level)); });

    app.require_subcommand(1);
    std::unique_ptr<Command> command;

    // ------------------------------
    // generate dataset command
    // ------------------------------

    CLI::App* generate_dataset_command =
        app.add_subcommand("generate_dataset", "Generate a dataset for Chamfer Distance Approximation");

    std::filesystem::path dataset_directory;
    generate_dataset_command->add_option("-d,--dataset_directory", dataset_directory, "Directory for the dataset")
        ->required();

    std::unique_ptr<DatasetGenerator> dataset_generator;
    generate_dataset_command->require_subcommand(1);

    // ------------------------------
    // synthesize dataset command
    // ------------------------------

    CLI::App* synthesize_dataset_command = generate_dataset_command->add_subcommand(
        "synthesize", "Generate a dataset by synthesizing points within a specified range");

    std::string data_type;
    synthesize_dataset_command
        ->add_option("-t,--data-type", data_type, "Data type for the dataset (uint8, int, double)")
        ->default_val("double")
        ->check(CLI::IsMember({"uint8", "int", "double"}));

    unsigned int base_num_points{0};
    synthesize_dataset_command
        ->add_option("-B,--base_num_points", base_num_points, "Number of points within base set in the dataset")
        ->required();

    unsigned int query_num_points{0};
    synthesize_dataset_command
        ->add_option("-Q,--query_num_points", query_num_points, "Number of points within query set in the dataset")
        ->required();

    unsigned int num_dimensions{0};
    synthesize_dataset_command
        ->add_option("-D,--num_dimensions", num_dimensions, "Number of dimensions for the dataset")
        ->required();

    double left_boundary{0.0};
    synthesize_dataset_command->add_option("-l,--left_boundary", left_boundary, "Left boundary for generated points")
        ->default_val(Constants::kDefaultLeftBoundary);

    double right_boundary{0.0};
    synthesize_dataset_command->add_option("-r,--right_boundary", right_boundary, "Right boundary for generated points")
        ->default_val(Constants::kDefaultRightBoundary);

    bool in_memory{false};
    synthesize_dataset_command->add_flag("-i,--in_memory", in_memory, "Generate the dataset in memory (default: false)")
        ->default_val(false)
        ->default_str("false");

    synthesize_dataset_command->callback([&]() {
        DatasetMetadata dataset_metadata = DatasetMetadata{
            .data_type = data_type,
            .base_num_points = base_num_points,
            .query_num_points = query_num_points,
            .num_dimensions = num_dimensions,
            .chamfer_distance = 0.0  // Will be calculated later
        };
        dataset_generator =
            std::make_unique<DatasetSynthesizer>(dataset_metadata, left_boundary, right_boundary, in_memory);
    });

    generate_dataset_command->callback([&]() {
        if (!dataset_generator) {
            spdlog::critical("Dataset generator is not set. Please specify a dataset generator.");
        }
        command = std::unique_ptr<Command>(new GenerateDatasetCommand(std::move(dataset_generator), dataset_directory));
    });

    // ------------------------------
    // index command
    // ------------------------------

    CLI::App* index_command = app.add_subcommand("index", "Index a dataset for Chamfer Distance Approximation");
    index_command->add_option("-d,--dataset_directory", dataset_directory, "Directory for the dataset")->required();

    std::unique_ptr<Indexer> indexer;
    index_command->require_subcommand(1);

    // ------------------------------
    // qalsh index command
    // ------------------------------

    CLI::App* qalsh_index_command = index_command->add_subcommand("qalsh", "Index a dataset using QALSH algorithm");

    double approximation_ratio{0.0};
    qalsh_index_command->add_option("-c, --approximation_ratio", approximation_ratio, "Approximation ratio for QALSH")
        ->default_val(2.0);

    double bucket_width{0.0};
    qalsh_index_command
        ->add_option("-w, --bucket_width", bucket_width,
                     "Bucket width for the indexer (default: 0, which means the bucket width will be 2\\sqrt(c))")
        ->default_val(0.0);

    double beta{0.0};
    qalsh_index_command
        ->add_option("-b, --beta", beta,
                     "Beta parameter for the indexer (default: 0, which means the beta will be 100/num_points)")
        ->default_val(0.0);

    double error_probability{0.0};
    qalsh_index_command
        ->add_option("-e, --error_probability", error_probability, "Error probability for the indexer (default: 1/e)")
        ->default_val(Constants::kDefaultErrorProbability)
        ->default_str("1/e");

    unsigned int num_hash_tables{0};
    qalsh_index_command
        ->add_option("-m,--num_hash_tables", num_hash_tables,
                     "Number of hash tables to use for indexing (default: 0, which means the number will be determined "
                     "automatically)")
        ->default_val(0);

    unsigned int collision_threshold{0};
    qalsh_index_command
        ->add_option("-l,--collision_threshold", collision_threshold,
                     "Collision threshold for the indexer (default: 0, which means the collision threshold will be "
                     "determined automatically)")
        ->default_val(0);

    unsigned int page_size{0};
    qalsh_index_command
        ->add_option("-B,--page_size", page_size,
                     std::format("Page size for the indexer (default: {} bytes)", Constants::kDefaultPageSize))
        ->default_val(Constants::kDefaultPageSize);

    qalsh_index_command->add_flag("-i,--in_memory", in_memory, "Index the dataset in memory (default: false)")
        ->default_val(false)
        ->default_str("false");

    qalsh_index_command->callback([&]() {
        QalshConfiguration qalsh_config = {.approximation_ratio = approximation_ratio,
                                           .bucket_width = bucket_width,
                                           .beta = beta,
                                           .error_probability = error_probability,
                                           .num_hash_tables = num_hash_tables,
                                           .collision_threshold = collision_threshold,
                                           .page_size = page_size};
        indexer = std::make_unique<QalshIndexer>(dataset_directory, qalsh_config, in_memory);
    });

    index_command->callback([&]() {
        if (!indexer) {
            spdlog::critical("Indexer is not set. Please specify an indexer.");
        }
        command = std::unique_ptr<Command>(new IndexCommand(std::move(indexer)));
    });

    CLI11_PARSE(app, argc, argv);

    if (!command) {
        spdlog::critical("Command is not set. Please specify a command.");
    } else {
        command->Execute();
    }

    return 0;
}