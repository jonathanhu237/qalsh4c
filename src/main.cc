#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <filesystem>
#include <memory>

#include "ann_searcher.h"
#include "command.h"
#include "dataset_metadata.h"
#include "estimator.h"
#include "global.h"
#include "sink.h"

int main(int argc, char** argv) {
    // Setup logger.
    auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    auto terminating_sink = std::make_shared<TerminatingSink<std::mutex>>(console_sink);
    auto logger = std::make_shared<spdlog::logger>("qalsh_chamfer", terminating_sink);
    spdlog::set_default_logger(logger);

    // Parse command-line arguments.
    CLI::App app{"Fast Chamfer Distance Approximation via Query-Aware Locality-Sensitive Hashing (QALSH)."};

    std::string log_level;
    app.add_option("-l,--log_level", log_level, "Set the logging level (default: warn)")
        ->default_val("warn")
        ->check(CLI::IsMember({"debug", "info", "warn", "error"}));

    std::unique_ptr<Command> command;
    app.require_subcommand(1);
    app.callback([&]() {
        if (!command) {
            spdlog::error("Command is not set. Please specify a command.");
        }
        spdlog::set_level(spdlog::level::from_str(log_level));
        command->Execute();
    });

    // ------------------------------
    // generate dataset
    // ------------------------------
    CLI::App* generate_dataset =
        app.add_subcommand("generate_dataset", "Generate a dataset for Chamfer distance estimation.");

    unsigned int num_points_a{0};
    generate_dataset->add_option("-a,--num_points_a", num_points_a, "Number of points in the first point set (A).")
        ->default_val(Global::kDefaultNumPointsA);

    unsigned int num_points_b{0};
    generate_dataset->add_option("-b,--num_points_b", num_points_b, "Number of points in the second point set (B).")
        ->default_val(Global::kDefaultNumPointsB);

    unsigned int num_dimensions{0};
    generate_dataset->add_option("-d,--num_dimensions", num_dimensions, "Number of dimension for the point sets.")
        ->default_val(Global::kDefaultNumDimensions);

    double left_boundary{0.0};
    generate_dataset
        ->add_option("-l,--left_boundary", left_boundary,
                     "The left boundary of uniform distribution used in dataset generation.")
        ->default_val(Global::kDefaultLeftBoundary);

    double right_boundary{0.0};
    generate_dataset
        ->add_option("-r,--right_boundary", right_boundary,
                     "The right boundary of uniform distribution used in dataset generation")
        ->default_val(Global::kDefaultRightBoundary);

    std::filesystem::path output_directory;
    generate_dataset->add_option("-o,--output_directory", output_directory, "The directory to save dataset.")
        ->required();

    bool in_memory{false};
    generate_dataset->add_flag("--in-memory", in_memory, "Run the algorithm in memory")
        ->default_str(in_memory ? "True" : "False");

    generate_dataset->callback([&]() {
        DatasetMetadata dataset_metadata = {
            .num_points_a = num_points_a,
            .num_points_b = num_points_b,
            .num_dimensions = num_dimensions,
        };
        command = std::make_unique<GenerateDatasetCommand>(dataset_metadata, left_boundary, right_boundary,
                                                           output_directory, in_memory);
    });

    // ------------------------------
    // index
    // ------------------------------
    CLI::App* index = app.add_subcommand("qalsh", "Index a dataset using QALSH algorithm");

    double approximation_ratio{0.0};
    index->add_option("-c, --approximation_ratio", approximation_ratio, "Approximation ratio for QALSH")
        ->default_val(Global::kDefaultApproximationRatio);

    unsigned int page_size{0};
    index
        ->add_option("-B,--page_size", page_size,
                     std::format("Page size for the indexer (default: {} bytes)", Global::kDefaultPageSize))
        ->default_val(Global::kDefaultPageSize);

    std::filesystem::path dataset_directory;
    index->add_option("-d,--dataset_directory", dataset_directory, "Directory for the dataset")->required();

    index->callback(
        [&]() { command = std::make_unique<IndexCommand>(approximation_ratio, page_size, dataset_directory); });

    // ------------------------------
    // estimate
    // ------------------------------
    CLI::App* estimate = app.add_subcommand("estimate", "Estimate Chamfer distance.");

    estimate->add_option("-d,--dataset_directory", dataset_directory, "Directory for the dataset")->required();

    estimate->add_flag("--in-memory", in_memory, "Run the algorithm in memory")
        ->default_str(in_memory ? "True" : "False");

    std::unique_ptr<Estimator> estimator;
    estimate->require_subcommand(1);
    estimate->callback([&]() {
        if (!estimator) {
            spdlog::error("Estimator is not set. Please specify a estimator.");
        }
        command = std::make_unique<EstimateCommand>(std::move(estimator), dataset_directory, in_memory);
    });

    // ------------------------------
    // ann estimate
    // ------------------------------
    CLI::App* ann = estimate->add_subcommand("ann", "Estimate Chamfer distance using ANN.");

    std::unique_ptr<AnnSearcher> ann_searcher;
    ann->require_subcommand(1);
    ann->callback([&]() {
        if (!ann_searcher) {
            spdlog::error("Ann searcher is not set. Please specify a Ann searcher.");
        }
        estimator = std::make_unique<AnnEstimator>(std::move(ann_searcher));
    });

    // ------------------------------
    // linear scan ann estimate
    // ------------------------------
    CLI::App* linear_scan = ann->add_subcommand("linear_scan", "Use linear scan for ANN.");

    linear_scan->callback([&] { ann_searcher = std::make_unique<LinearScanAnnSearcher>(); });

    // ------------------------------
    // qalsh ann estimate
    // ------------------------------
    CLI::App* qalsh_ann = ann->add_subcommand("qalsh", "Use QALSH for ANN.");

    qalsh_ann->add_option("-c, --approximation_ratio", approximation_ratio, "Approximation ratio for QALSH")
        ->default_val(Global::kDefaultApproximationRatio);

    qalsh_ann->callback([&] {
        if (in_memory) {
            ann_searcher = std::make_unique<QalshAnnSearcher>(approximation_ratio);
        } else {
            spdlog::error("The disk version of QalshAnnSearcher has not been implemented yet.");
        }
    });

    CLI11_PARSE(app, argc, argv);

    return 0;
}