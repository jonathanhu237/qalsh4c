#include <CLI/CLI.hpp>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "ann_searcher.h"
#include "command.h"
#include "estimator.h"
#include "global.h"
#include "log.h"
#include "weights_generator.h"

#ifdef QALSH4C_USE_OPENMP
#include <omp.h>
#endif

namespace {

unsigned int DefaultThreadCount() {
#ifdef QALSH4C_USE_OPENMP
    const int num_processors = omp_get_num_procs();
    if (num_processors > 0) {
        return static_cast<unsigned int>(num_processors);
    }
#endif

    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    return hardware_threads == 0 ? 1 : hardware_threads;
}

}  // namespace

int main(int argc, char** argv) {
    // Parse command-line arguments.
    CLI::App app{"Fast Chamfer Distance Approximation via Query-Aware Locality-Sensitive Hashing (QALSH)."};

    unsigned int thread_count{DefaultThreadCount()};
    app.add_option("-T,--threads", thread_count, "Number of OpenMP threads")
        ->default_val(thread_count)
        ->check(CLI::PositiveNumber);

    app.add_flag("--use-fixed-seed", Global::kUseFixedSeed, "Use a fixed seed for randomness")
        ->default_str(Global::kUseFixedSeed ? "True" : "False");
    app.add_option("--initial-radius", Global::kInitialRadius,
                   "Starting radius R_0 of the round rules (2^-149 is valid for any float32 data and any c > 1)")
        ->default_val(Global::kInitialRadius)
        ->check(CLI::PositiveNumber);
    app.add_flag("--qalsh-stop-within-round{false}", Global::kQalshStopAtRoundBoundary,
                 "With the round rule, check QALSH's stopping condition after every new candidate instead of after each "
                 "complete round (QALSH's own test; gamma = c^2 instead of c)");
    app.add_option("--qalsh-scan-order", Global::kQalshScanOrder,
                   "Order of the window entries inside a round of an in-memory QALSH query: quanta (a fixed number of "
                   "entries per table visit), distance (all tables advance together in distance from the query's "
                   "projection) or distance-reference (the distance order one entry at a time, for checks)")
        ->default_val(Global::kQalshScanOrder)
        ->check(CLI::IsMember({"quanta", "distance", "distance-reference"}));
    app.add_option("--qalsh-advance-check", Global::kQalshCheckAfterRadiusAdvance,
                   "With round-end checks, also check the stopping condition right after the radius grows, against the "
                   "new radius (true) or only at the end of each round (false)")
        ->default_val(Global::kQalshCheckAfterRadiusAdvance);
    app.add_option("--qalsh-stop-rule", Global::kQalshStopRule,
                   "Stopping rule of in-memory QALSH queries: round (the round rules) or continuous (with the distance "
                   "scan order, stop once the best distance is at most c times the radius of the scan front)")
        ->default_val(Global::kQalshStopRule)
        ->check(CLI::IsMember({"round", "continuous"}));

    std::unique_ptr<Command> command;
    app.require_subcommand(1);
    app.callback([&]() {
        if (!command) {
            Log::error("Command is not set. Please specify a command.");
        }
#ifdef QALSH4C_USE_OPENMP
        omp_set_num_threads(static_cast<int>(thread_count));
#endif
        command->Execute();
    });

    float norm_order{0.0};
    std::string direction_value{"ab"};
    std::filesystem::path dataset_directory;

    // ------------------------------
    // kappa
    // ------------------------------
    CLI::App* kappa_app = app.add_subcommand(
        "kappa", "Print the certified bound on kappa and the sample sizes T it gives for a norm and an approximation "
                 "ratio, under the stopping rule the global options select.");
    float kappa_norm_order{2.0F};
    float kappa_approximation_ratio{Global::kDefaultApproximationRatio};
    std::vector<double> kappa_epsilons{0.1};  // NOLINT(readability-magic-numbers)
    double kappa_delta{0.1};                  // NOLINT(readability-magic-numbers)
    kappa_app->add_option("-p,--norm-order", kappa_norm_order, "Norm order")
        ->required()
        ->check(CLI::IsMember({1.0, 2.0}));  // NOLINT(readability-magic-numbers)
    kappa_app->add_option("-c,--approximation-ratio", kappa_approximation_ratio, "Approximation ratio c of QALSH")
        ->default_val(Global::kDefaultApproximationRatio);
    kappa_app->add_option("-e,--epsilon", kappa_epsilons, "Error tolerances epsilon, one T per value (default 0.1)");
    kappa_app->add_option("--delta", kappa_delta, "Failure probability delta")->default_val(0.1);
    kappa_app->callback([&]() {
        command = std::make_unique<KappaCommand>(kappa_norm_order, kappa_approximation_ratio, kappa_epsilons,
                                                 kappa_delta);
    });

    // ------------------------------
    // estimate
    // ------------------------------
    CLI::App* estimate = app.add_subcommand("estimate", "Estimate Chamfer distance.");

    estimate->add_option("-p, --norm-order", norm_order, "Norm order")
        ->required()
        ->check(CLI::IsMember({1.0, 2.0}));  // NOLINT(readability-magic-numbers)

    estimate->add_option("-d,--dataset-directory", dataset_directory, "Directory for the dataset")->required();

    estimate->add_option("--direction", direction_value, "Chamfer direction: ab, ba, or both")
        ->default_val("ab")
        ->check(CLI::IsMember({"ab", "ba", "both"}));

    std::unique_ptr<Estimator> estimator;
    estimate->require_subcommand(1);
    estimate->callback([&]() {
        if (!estimator) {
            Log::error("Estimator is not set. Please specify a estimator.");
        }
        command = std::make_unique<EstimateCommand>(std::move(estimator), norm_order, dataset_directory,
                                                    ParseChamferDirection(direction_value));
    });

    // ------------------------------
    // ann estimate
    // ------------------------------
    CLI::App* ann = estimate->add_subcommand("ann", "Estimate Chamfer distance using ANN.");

    std::unique_ptr<AnnSearcher> ann_searcher;
    std::filesystem::path save_distances;
    ann->add_option("--save-distances", save_distances,
                    "Also write d(a, B) of every source point as float64 (input of sampling --exact-distances)");
    ann->require_subcommand(1);
    ann->callback([&]() {
        if (!ann_searcher) {
            Log::error("Ann searcher is not set. Please specify a Ann searcher.");
        }
        if (!save_distances.empty() && direction_value == "both") {
            Log::error("--save-distances needs a single direction (ab or ba).");
        }
        estimator = std::make_unique<AnnEstimator>(std::move(ann_searcher), save_distances);
    });

    // ------------------------------
    // linear scan ann estimate
    // ------------------------------
    CLI::App* linear_scan = ann->add_subcommand("linear_scan", "Use linear scan for ANN.");

    linear_scan->callback([&] { ann_searcher = std::make_unique<LinearScanAnnSearcher>(); });

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

    float epsilon{0.0};
    sampling->add_option("-e,--epsilon", epsilon, "Epsilon for the algorithm")->default_val(Global::kQalsh4cEpsilon);

    float delta{0.0};
    sampling->add_option("--delta", delta, "Failure probability for automatic QALSH sample count")
        ->default_val(Global::kQalsh4cDelta);

    bool use_cache{false};
    sampling->add_flag("--use-cache", use_cache, "Use cached files if available")
        ->default_val(false)
        ->default_str(use_cache ? "True" : "False");

    unsigned int repeats{1};
    sampling->add_option("--repeats", repeats, "Independent samples drawn from the same weights; each is logged")
        ->default_val(1)
        ->check(CLI::PositiveNumber);

    std::filesystem::path exact_distances;
    sampling->add_option("--exact-distances", exact_distances,
                         "For experiments: a float64 file of d(a, B) for every source point; the sampled ones are "
                         "looked up instead of computed, which gives the same estimates")
        ->check(CLI::ExistingFile);

    std::unique_ptr<WeightsGenerator> weights_generator;
    sampling->require_subcommand(1);
    sampling->callback([&]() {
        if (!weights_generator) {
            Log::error("Weights generator is not set. Please specify a weights generator.");
        }
        estimator = std::make_unique<SamplingEstimator>(
            std::move(weights_generator), SamplingOptions{.num_samples_default = num_samples,
                                                          .num_samples_ab = num_samples_ab,
                                                          .num_samples_ba = num_samples_ba,
                                                          .epsilon = epsilon,
                                                          .delta = delta,
                                                          .use_cache = use_cache,
                                                          .repeats = repeats,
                                                          .exact_distances = exact_distances});
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
    float qalsh_sampling_approximation_ratio{Global::kDefaultApproximationRatio};
    qalsh_sampling
        ->add_option("-c,--approximation-ratio", qalsh_sampling_approximation_ratio,
                     "Approximation ratio for QALSH")
        ->default_val(Global::kDefaultApproximationRatio);
    qalsh_sampling->callback([&]() {
        weights_generator = std::make_unique<QalshWeightsGenerator>(qalsh_sampling_approximation_ratio);
    });

    // ------------------------------
    // precomputed weights
    // ------------------------------
    CLI::App* precomputed_sampling = sampling->add_subcommand(
        "precomputed", "Use weights from a float32 file, one per source point (for example saved crude distances).");
    std::filesystem::path precomputed_weights_path;
    precomputed_sampling->add_option("-f,--file", precomputed_weights_path, "Weights file")
        ->required()
        ->check(CLI::ExistingFile);
    precomputed_sampling->callback([&]() {
        weights_generator = std::make_unique<PrecomputedWeightsGenerator>(precomputed_weights_path);
    });

    // ------------------------------
    // quadtree sampling estimate
    // ------------------------------
    CLI::App* quadtree_sampling =
        sampling->add_subcommand("quadtree", "Generate samples using quadtree-derived importance weights.");
    unsigned int quadtree_max_levels{QuadtreeWeightsGenerator::kDefaultMaxLevels};
    quadtree_sampling
        ->add_option("--max-levels", quadtree_max_levels,
                     "Maximum quadtree levels; 0 splits until singleton")
        ->default_val(QuadtreeWeightsGenerator::kDefaultMaxLevels)
        ->check(CLI::Range(0, std::numeric_limits<int>::max()));
    quadtree_sampling->callback([&]() {
        weights_generator = std::make_unique<QuadtreeWeightsGenerator>(quadtree_max_levels);
    });

    CLI11_PARSE(app, argc, argv);

    return 0;
}
