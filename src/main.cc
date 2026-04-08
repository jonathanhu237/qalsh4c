#include <CLI/CLI.hpp>
#include <filesystem>
#include <memory>

#include "ann_searcher.h"
#include "command.h"
#include "estimator.h"
#include "global.h"
#include "log.h"
#include "weights_generator.h"

int main(int argc, char** argv) {
    // Parse command-line arguments.
    CLI::App app{"Fast Chamfer Distance Approximation via Query-Aware Locality-Sensitive Hashing (QALSH)."};

    app.add_flag("--use-fixed-seed", Global::kUseFixedSeed, "Use a fixed seed for randomness")
        ->default_str(Global::kUseFixedSeed ? "True" : "False");

    std::unique_ptr<Command> command;
    app.require_subcommand(1);
    app.callback([&]() {
        if (!command) {
            Log::error("Command is not set. Please specify a command.");
        }
        command->Execute();
    });

    // ------------------------------
    // index
    // ------------------------------
    CLI::App* index = app.add_subcommand("index", "Index a dataset for QALSH algorithm");

    double norm_order{0.0};
    index->add_option("-p, --norm-order", norm_order, "Norm order")
        ->required()
        ->check(CLI::IsMember({1.0, 2.0}));  // NOLINT(readability-magic-numbers)

    double approximation_ratio{0.0};
    index->add_option("-c, --approximation-ratio", approximation_ratio, "Approximation ratio for QALSH")
        ->default_val(Global::kDefaultApproximationRatio);

    unsigned int page_size{0};
    index
        ->add_option("-B,--page-size", page_size,
                     std::format("Page size for the indexer (default: {} bytes)", Global::kDefaultPageSize))
        ->default_val(Global::kDefaultPageSize);

    bool fast{false};
    index->add_flag("--fast", fast, "Use in-memory sorting (faster but uses more memory)");

    std::filesystem::path dataset_directory;
    index->add_option("-d,--dataset-directory", dataset_directory, "Directory for the dataset")->required();

    index->callback([&]() {
        command = std::make_unique<IndexCommand>(norm_order, approximation_ratio, page_size, dataset_directory, fast);
    });

    // ------------------------------
    // estimate
    // ------------------------------
    CLI::App* estimate = app.add_subcommand("estimate", "Estimate Chamfer distance.");

    estimate->add_option("-p, --norm-order", norm_order, "Norm order")
        ->required()
        ->check(CLI::IsMember({1.0, 2.0}));  // NOLINT(readability-magic-numbers)

    estimate->add_option("-d,--dataset-directory", dataset_directory, "Directory for the dataset")->required();

    std::unique_ptr<Estimator> estimator;
    estimate->require_subcommand(1);
    estimate->callback([&]() {
        if (!estimator) {
            Log::error("Estimator is not set. Please specify a estimator.");
        }
        command = std::make_unique<EstimateCommand>(std::move(estimator), norm_order, dataset_directory);
    });

    // ------------------------------
    // ann estimate
    // ------------------------------
    CLI::App* ann = estimate->add_subcommand("ann", "Estimate Chamfer distance using ANN.");

    std::unique_ptr<AnnSearcher> ann_searcher;
    ann->require_subcommand(1);
    ann->callback([&]() {
        if (!ann_searcher) {
            Log::error("Ann searcher is not set. Please specify a Ann searcher.");
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

    qalsh_ann->add_option("-c, --approximation-ratio", approximation_ratio, "Approximation ratio for QALSH")
        ->default_val(Global::kDefaultApproximationRatio);

    bool candidate_limit{false};
    qalsh_ann->add_flag("--candidate-limit", candidate_limit,
                        "Enable early termination when candidate count reaches the limit");

    qalsh_ann->callback([&] { ann_searcher = std::make_unique<QalshAnnSearcher>(candidate_limit); });

    // ------------------------------
    // sampling estimate
    // ------------------------------
    CLI::App* sampling = estimate->add_subcommand("sampling", "Estimate Chamfer distance using sampling.");

    unsigned int num_samples{0};
    unsigned int num_samples_ab{0};
    unsigned int num_samples_ba{0};

    sampling->add_option("-n,--num-samples", num_samples, "Default number of samples for both directions");
    sampling->add_option("--num-samples-ab", num_samples_ab, "Number of samples for A->B direction (overrides -n)");
    sampling->add_option("--num-samples-ba", num_samples_ba, "Number of samples for B->A direction (overrides -n)");

    double epsilon{0.0};
    sampling->add_option("-e,--epsilon", epsilon, "Epsilon for the algorithm")->default_val(Global::kQalsh4cEpsilon);

    bool use_cache{false};
    sampling->add_flag("--use-cache", use_cache, "Use cached files if available")
        ->default_val(false)
        ->default_str(use_cache ? "True" : "False");

    std::unique_ptr<WeightsGenerator> weights_generator;
    sampling->require_subcommand(1);
    sampling->callback([&]() {
        if (!weights_generator) {
            Log::error("Weights generator is not set. Please specify a weights generator.");
        }
        estimator = std::make_unique<SamplingEstimator>(std::move(weights_generator), num_samples, num_samples_ab,
                                                        num_samples_ba, epsilon, use_cache);
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
    qalsh_sampling->callback([&]() { weights_generator = std::make_unique<QalshWeightsGenerator>(); });

    CLI11_PARSE(app, argc, argv);

    return 0;
}
