
#include <spdlog/common.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <format>
#include <memory>
#include <mutex>

#include "command.h"
#include "constants.h"
#include "dataset_generator.h"
#include "estimator.h"
#include "global.h"
#include "indexer.h"
#include "sink.h"
#include "types.h"

int main(int argc, char** argv) {
    CLI::App app{"Fast Chamfer Distance Approximation via Query-Aware Locality-Sensitive Hashing (QALSH)."};

    std::string log_level;
    app.add_option("-l,--log_level", log_level, "Set the logging level (default: warn)")
        ->default_val("warn")
        ->check(CLI::IsMember({"debug", "info", "warn", "error"}))
        ->each([&](const std::string& level) {
            auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
            auto terminating_sink = std::make_shared<TerminatingSink<std::mutex>>(console_sink);
            auto logger = std::make_shared<spdlog::logger>("qalsh_chamfer", terminating_sink);
            logger->set_level(spdlog::level::from_str(level));
            spdlog::set_default_logger(logger);
        });

    app.add_flag("-m,--high_memory", Global::high_memory_mode, "Enable high memory mode")
        ->default_str(Global::high_memory_mode ? "true" : "false");

    app.add_flag("-t,--measure_time", Global::measure_time, "Enable time measurement")
        ->default_str(Global::measure_time ? "true" : "false");

    app.require_subcommand(1);
    std::unique_ptr<Command> command;

    app.callback([&]() {
        if (Global::high_memory_mode) {
            spdlog::warn("High memory mode is enabled.");
        }

        if (Global::measure_time) {
            spdlog::warn("Time measurement is enabled.");
        }

        if (!command) {
            spdlog::error("Command is not set. Please specify a command.");
            std::exit(EXIT_FAILURE);
        }
        command->Execute();
    });

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

    generate_dataset_command->callback([&]() {
        if (!dataset_generator) {
            spdlog::error("Dataset generator is not set. Please specify a dataset generator.");
        }
        command = std::unique_ptr<Command>(new GenerateDatasetCommand(std::move(dataset_generator), dataset_directory));
    });

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

    synthesize_dataset_command->callback([&]() {
        DatasetMetadata dataset_metadata = DatasetMetadata{
            .data_type = data_type,
            .base_num_points = base_num_points,
            .query_num_points = query_num_points,
            .num_dimensions = num_dimensions,
            .chamfer_distance = 0.0  // Will be calculated later
        };
        dataset_generator = std::make_unique<DatasetSynthesizer>(dataset_metadata, left_boundary, right_boundary);
    });

    // ------------------------------
    // index command
    // ------------------------------

    CLI::App* index_command = app.add_subcommand("index", "Index a dataset for Chamfer Distance Approximation");
    index_command->add_option("-d,--dataset_directory", dataset_directory, "Directory for the dataset")->required();

    std::unique_ptr<Indexer> indexer;
    index_command->require_subcommand(1);

    index_command->callback([&]() {
        if (!indexer) {
            spdlog::error("Indexer is not set. Please specify an indexer.");
        }
        command = std::unique_ptr<Command>(new IndexCommand(std::move(indexer), dataset_directory));
    });

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

    qalsh_index_command->callback([&]() {
        QalshConfiguration qalsh_config = {.approximation_ratio = approximation_ratio,
                                           .bucket_width = bucket_width,
                                           .beta = beta,
                                           .error_probability = error_probability,
                                           .num_hash_tables = num_hash_tables,
                                           .collision_threshold = collision_threshold,
                                           .page_size = page_size};
        indexer = std::make_unique<QalshIndexer>(qalsh_config);
    });

    // ------------------------------
    // estimate command
    // ------------------------------

    CLI::App* estimate_command = app.add_subcommand("estimate", "Estimate Chamfer distance");

    estimate_command->add_option("-d,--dataset_directory", dataset_directory, "Directory for the dataset")->required();

    std::unique_ptr<Estimator> estimator;
    estimate_command->require_subcommand(1);

    estimate_command->callback([&]() {
        if (!estimator) {
            spdlog::error("Estimator is not set. Please specify an estimator.");
        }
        command = std::unique_ptr<Command>(new EstimateCommand(std::move(estimator), dataset_directory));
    });

    // ------------------------------
    // ann estimate command
    // ------------------------------

    CLI::App* ann_estimate_command = estimate_command->add_subcommand("ann", "Estimate Chamfer distance using ANN");

    std::string ann_searcher_type;
    ann_estimate_command
        ->add_option("-s,--searcher_type", ann_searcher_type, "Type of ANN searcher (linear_scan, qalsh)")
        ->required()
        ->check(CLI::IsMember({"linear_scan", "qalsh"}));

    ann_estimate_command->callback([&]() { estimator = std::make_unique<AnnEstimator>(ann_searcher_type); });

    // ------------------------------
    // sample estimate command
    // ------------------------------

    CLI::App* sampling = estimate_command->add_subcommand("sampling", "Estimate Chamfer distance using sampling");

    std::string sampling_searcher_type;
    sampling->add_option("-s,--searcher_type", sampling_searcher_type, "Type of sampling searcher (uniform, qalsh)")
        ->required()
        ->check(CLI::IsMember({"uniform", "qalsh"}));

    unsigned int num_samples{0};
    sampling->add_option("-n,--num_samples", num_samples, "Number of samples to use for estimation")->required();

    sampling->callback([&]() { estimator = std::make_unique<SamplingEstimator>(sampling_searcher_type, num_samples); });

    CLI11_PARSE(app, argc, argv);

    return 0;
}