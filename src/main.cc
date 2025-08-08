#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <filesystem>
#include <memory>

#include "ann_searcher.h"
#include "command.h"
#include "estimator.h"
#include "global.h"
#include "sink.h"
#include "weights_generator.h"

int main(int argc, char** argv) {
    // Setup logger.
    auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    auto terminating_sink = std::make_shared<TerminatingSink<std::mutex>>(console_sink);
    auto logger = std::make_shared<spdlog::logger>("qalsh_chamfer", terminating_sink);
    spdlog::set_default_logger(logger);

    // Parse command-line arguments.
    CLI::App app{"Fast Chamfer Distance Approximation via Query-Aware Locality-Sensitive Hashing (QALSH)."};

    std::string log_level;
    app.add_option("-l,--log-level", log_level, "Set the logging level")
        ->default_val("info")
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
    // index
    // ------------------------------
    CLI::App* index = app.add_subcommand("index", "Index a dataset using QALSH algorithm");

    double approximation_ratio{0.0};
    index->add_option("-c, --approximation-ratio", approximation_ratio, "Approximation ratio for QALSH")
        ->default_val(Global::kDefaultApproximationRatio);

    unsigned int page_size{0};
    index
        ->add_option("-B,--page-size", page_size,
                     std::format("Page size for the indexer (default: {} bytes)", Global::kDefaultPageSize))
        ->default_val(Global::kDefaultPageSize);

    std::filesystem::path dataset_directory;
    index->add_option("-d,--dataset-directory", dataset_directory, "Directory for the dataset")->required();

    index->callback(
        [&]() { command = std::make_unique<IndexCommand>(approximation_ratio, page_size, dataset_directory); });

    // ------------------------------
    // estimate
    // ------------------------------
    CLI::App* estimate = app.add_subcommand("estimate", "Estimate Chamfer distance.");

    estimate->add_option("-d,--dataset-directory", dataset_directory, "Directory for the dataset")->required();

    bool in_memory{false};
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

    linear_scan->callback([&] {
        if (in_memory) {
            ann_searcher = std::make_unique<InMemoryLinearScanAnnSearcher>();
        } else {
            ann_searcher = std::make_unique<DiskLinearScanAnnSearcher>();
        }
    });

    // ------------------------------
    // qalsh ann estimate
    // ------------------------------
    CLI::App* qalsh_ann = ann->add_subcommand("qalsh", "Use QALSH for ANN.");

    qalsh_ann->add_option("-c, --approximation-ratio", approximation_ratio, "Approximation ratio for QALSH")
        ->default_val(Global::kDefaultApproximationRatio);

    // If in_memory = false, the setting of approximation_ratio would not have any effect.
    qalsh_ann->callback([&] {
        if (in_memory) {
            ann_searcher = std::make_unique<InMemoryQalshAnnSearcher>(approximation_ratio);
        } else {
            ann_searcher = std::make_unique<DiskQalshAnnSearcher>();
        }
    });

    // ------------------------------
    // sampling estimate
    // ------------------------------
    CLI::App* sampling = estimate->add_subcommand("sampling", "Estimate Chamfer distance using sampling.");

    unsigned int num_samples{0};
    sampling->add_option("-n,--num-samples", num_samples, "Number of samples to use for estimation");

    sampling
        ->add_option("-c,--approximation-ratio", approximation_ratio, "Approximation ratio for the Chamfer distance")
        ->default_val(Global::kDefaultApproximationRatio);

    double error_probability{0.0};
    sampling->add_option("-e,--error-probability", error_probability, "Error probability for the algorithm")
        ->default_val(Global::kDefaultErrorProbability);

    bool use_cache{false};
    sampling->add_flag("--use-cache", use_cache, "Use cached files if available")
        ->default_val(false)
        ->default_str(use_cache ? "True" : "False");

    std::unique_ptr<WeightsGenerator> weights_generator;
    sampling->require_subcommand(1);
    sampling->callback([&]() {
        if (!weights_generator) {
            spdlog::error("Weights generator is not set. Please specify a weights generator.");
        }
        estimator = std::make_unique<SamplingEstimator>(std::move(weights_generator), num_samples, approximation_ratio,
                                                        error_probability, use_cache);
    });

    // ------------------------------
    // uniform sampling estimate
    // ------------------------------
    CLI::App* uniform = sampling->add_subcommand("uniform", "Generate samples using uniform distribution.");
    uniform->callback([&]() { weights_generator = std::make_unique<UniformWeightsGenerator>(); });

    // ------------------------------
    // qalsh sampling estimate
    // ------------------------------
    CLI::App* qalsh_sampling = sampling->add_subcommand("qalsh", "Generate samples using QALSH.");

    qalsh_sampling->add_option("-c, --approximation-ratio", approximation_ratio, "Approximation ratio for QALSH")
        ->default_val(Global::kDefaultApproximationRatio);

    qalsh_sampling->callback([&]() {
        if (in_memory) {
            weights_generator = std::make_unique<InMemoryQalshWeightsGenerator>(approximation_ratio);
        } else {
            weights_generator = std::make_unique<DiskQalshWeightsGenerator>();
        }
    });

    CLI11_PARSE(app, argc, argv);

    return 0;
}