#include <CLI/CLI.hpp>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>

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

    std::string direction_value{"ab"};
    index->add_option("--direction", direction_value, "Chamfer direction: ab, ba, or both")
        ->default_val("ab")
        ->check(CLI::IsMember({"ab", "ba", "both"}));

    std::filesystem::path dataset_directory;
    index->add_option("-d,--dataset-directory", dataset_directory, "Directory for the dataset")->required();

    index->callback([&]() {
        command = std::make_unique<IndexCommand>(norm_order, approximation_ratio, page_size, dataset_directory, fast,
                                                 ParseChamferDirection(direction_value));
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

    // ------------------------------
    // dataset
    // ------------------------------
    CLI::App* dataset = app.add_subcommand("dataset", "Prepare datasets for QALSH4C.");
    dataset->require_subcommand(1);

    CLI::App* dataset_generate = dataset->add_subcommand("generate", "Generate synthetic datasets.");
    dataset_generate->require_subcommand(1);

    CLI::App* dataset_gmm = dataset_generate->add_subcommand("gmm", "Generate a Gaussian mixture dataset.");
    unsigned int gmm_num_points_a{1000};
    unsigned int gmm_num_points_b{1000};
    unsigned int gmm_num_dimensions{256};
    double gmm_outlier_fraction{0.1};
    double gmm_outlier_scale{4096.0};
    std::filesystem::path gmm_output_directory;
    std::string gmm_ground_truth_specs;
    std::string gmm_ground_truth_device{"auto"};
    size_t gmm_cuda_batch_size{0};
    bool gmm_overwrite{false};
    bool gmm_no_progress{false};
    unsigned int gmm_seed{0};
    dataset_gmm->add_option("-a,--num-points-a", gmm_num_points_a, "Number of points in point set A")
        ->default_val(gmm_num_points_a);
    dataset_gmm->add_option("-b,--num-points-b", gmm_num_points_b, "Number of points in point set B")
        ->default_val(gmm_num_points_b);
    dataset_gmm->add_option("--num-dimensions", gmm_num_dimensions, "Number of dimensions")
        ->default_val(gmm_num_dimensions);
    dataset_gmm->add_option("--outlier-fraction", gmm_outlier_fraction, "Fraction of outlier points")
        ->default_val(gmm_outlier_fraction);
    dataset_gmm->add_option("--outlier-scale", gmm_outlier_scale, "Outlier covariance scale")
        ->default_val(gmm_outlier_scale);
    dataset_gmm->add_option("-o,--output-directory", gmm_output_directory, "Dataset output directory")->required();
    dataset_gmm->add_option("--ground-truth", gmm_ground_truth_specs,
                            "Comma-separated selectors: l1:ab,l1:ba,l2:ab,l2:ba");
    dataset_gmm->add_option("--ground-truth-device", gmm_ground_truth_device,
                            "Ground-truth compute device: auto, cpu, or cuda")
        ->default_val("auto")
        ->check(CLI::IsMember({"auto", "cpu", "cuda"}));
    dataset_gmm->add_option("--cuda-batch-size", gmm_cuda_batch_size,
                            "Maximum CUDA source batch points for ground-truth computation (0 = auto)")
        ->default_val(gmm_cuda_batch_size);
    dataset_gmm->add_flag("--overwrite", gmm_overwrite, "Overwrite existing dataset files and remove old indexes");
    dataset_gmm->add_flag("--no-progress", gmm_no_progress, "Disable progress reporting");
    CLI::Option* gmm_seed_option = dataset_gmm->add_option("--seed", gmm_seed, "Seed for reproducible generation");
    dataset_gmm->callback([&]() {
        DatasetPrep::GmmOptions options{
            .output = {.output_directory = gmm_output_directory,
                       .ground_truth_specs = gmm_ground_truth_specs,
                       .overwrite = gmm_overwrite,
                       .show_progress = !gmm_no_progress,
                       .ground_truth_device = DatasetPrep::ParseGroundTruthDevice(gmm_ground_truth_device),
                       .cuda_batch_size = gmm_cuda_batch_size},
            .num_points_a = gmm_num_points_a,
            .num_points_b = gmm_num_points_b,
            .num_dimensions = gmm_num_dimensions,
            .outlier_fraction = gmm_outlier_fraction,
            .outlier_scale = gmm_outlier_scale,
            .seed = gmm_seed_option->count() > 0 ? std::optional<unsigned int>(gmm_seed) : std::nullopt,
        };
        command = std::make_unique<DatasetGenerateGmmCommand>(std::move(options));
    });

    CLI::App* dataset_convert = dataset->add_subcommand("convert", "Convert known raw datasets.");
    dataset_convert->require_subcommand(1);

    std::filesystem::path convert_output_directory;
    std::filesystem::path convert_raw_directory;
    std::filesystem::path convert_images_file;
    std::string convert_ground_truth_specs;
    std::string convert_ground_truth_device{"auto"};
    size_t convert_cuda_batch_size{0};
    bool convert_overwrite{false};
    bool convert_no_progress{false};
    bool convert_no_shuffle{false};
    unsigned int convert_num_points_a{0};
    unsigned int convert_seed{0};

    auto add_common_convert_options = [&](CLI::App* convert_app, bool needs_raw_dir, bool needs_images) {
        convert_app->add_option("-o,--output-directory", convert_output_directory, "Dataset output directory")
            ->required();
        convert_app->add_option("--ground-truth", convert_ground_truth_specs,
                                "Comma-separated selectors: l1:ab,l1:ba,l2:ab,l2:ba");
        convert_app->add_option("--ground-truth-device", convert_ground_truth_device,
                                "Ground-truth compute device: auto, cpu, or cuda")
            ->default_val("auto")
            ->check(CLI::IsMember({"auto", "cpu", "cuda"}));
        convert_app->add_option("--cuda-batch-size", convert_cuda_batch_size,
                                "Maximum CUDA source batch points for ground-truth computation (0 = auto)")
            ->default_val(convert_cuda_batch_size);
        convert_app->add_flag("--overwrite", convert_overwrite,
                              "Overwrite existing dataset files and remove old indexes");
        convert_app->add_flag("--no-progress", convert_no_progress, "Disable progress reporting");
        convert_app->add_flag("--no-shuffle", convert_no_shuffle, "Split points in loaded order");
        convert_app->add_option("--num-points-a", convert_num_points_a, "Number of points to place in point set A");
        convert_app->add_option("--seed", convert_seed, "Seed for reproducible shuffle");
        if (needs_raw_dir) {
            convert_app->add_option("--raw-dir", convert_raw_directory, "Raw dataset directory")->required();
        }
        if (needs_images) {
            convert_app->add_option("--images", convert_images_file, "MNIST IDX image file")->required();
        }
    };

    auto make_convert_options = [&](DatasetPrep::DatasetKind kind, const CLI::App* convert_app) {
        DatasetPrep::ConvertOptions options{
            .output = {.output_directory = convert_output_directory,
                       .ground_truth_specs = convert_ground_truth_specs,
                       .overwrite = convert_overwrite,
                       .show_progress = !convert_no_progress,
                       .ground_truth_device = DatasetPrep::ParseGroundTruthDevice(convert_ground_truth_device),
                       .cuda_batch_size = convert_cuda_batch_size},
            .kind = kind,
            .raw_directory = convert_raw_directory,
            .images_file = convert_images_file,
            .num_points_a = (*convert_app)["--num-points-a"]->count() > 0
                                ? std::optional<unsigned int>(convert_num_points_a)
                                : std::nullopt,
            .seed = (*convert_app)["--seed"]->count() > 0 ? std::optional<unsigned int>(convert_seed) : std::nullopt,
            .shuffle = !convert_no_shuffle,
        };
        return options;
    };

    CLI::App* convert_mnist = dataset_convert->add_subcommand("mnist", "Convert local full MNIST IDX image set.");
    add_common_convert_options(convert_mnist, true, false);
    convert_mnist->callback([&]() {
        command =
            std::make_unique<DatasetConvertCommand>(make_convert_options(DatasetPrep::DatasetKind::kMnist, convert_mnist));
    });

    CLI::App* convert_p53 = dataset_convert->add_subcommand("p53", "Convert local P53 K9.data.");
    add_common_convert_options(convert_p53, true, false);
    convert_p53->callback([&]() {
        command =
            std::make_unique<DatasetConvertCommand>(make_convert_options(DatasetPrep::DatasetKind::kP53, convert_p53));
    });

    CLI::App* convert_sift = dataset_convert->add_subcommand("sift", "Convert local SIFT fvecs files.");
    add_common_convert_options(convert_sift, true, false);
    convert_sift->callback([&]() {
        command =
            std::make_unique<DatasetConvertCommand>(make_convert_options(DatasetPrep::DatasetKind::kSift, convert_sift));
    });

    CLI::App* convert_gist = dataset_convert->add_subcommand("gist", "Convert local GIST fvecs files.");
    add_common_convert_options(convert_gist, true, false);
    convert_gist->callback([&]() {
        command =
            std::make_unique<DatasetConvertCommand>(make_convert_options(DatasetPrep::DatasetKind::kGist, convert_gist));
    });

    CLI::App* convert_trevi = dataset_convert->add_subcommand("trevi", "Convert local Trevi BMP patches.");
    add_common_convert_options(convert_trevi, true, false);
    convert_trevi->callback([&]() {
        command =
            std::make_unique<DatasetConvertCommand>(make_convert_options(DatasetPrep::DatasetKind::kTrevi, convert_trevi));
    });

    CLI::App* dataset_ground_truth =
        dataset->add_subcommand("ground-truth", "Compute selected ground-truth fields for an existing dataset.");
    std::filesystem::path ground_truth_dataset_directory;
    std::string ground_truth_specs;
    std::string ground_truth_device{"auto"};
    size_t ground_truth_cuda_batch_size{0};
    bool ground_truth_overwrite{false};
    bool ground_truth_no_progress{false};
    dataset_ground_truth->add_option("-d,--dataset-directory", ground_truth_dataset_directory, "Dataset directory")
        ->required();
    dataset_ground_truth->add_option("--ground-truth", ground_truth_specs,
                                     "Comma-separated selectors: l1:ab,l1:ba,l2:ab,l2:ba")
        ->required();
    dataset_ground_truth->add_option("--ground-truth-device", ground_truth_device,
                                     "Ground-truth compute device: auto, cpu, or cuda")
        ->default_val("auto")
        ->check(CLI::IsMember({"auto", "cpu", "cuda"}));
    dataset_ground_truth->add_option("--cuda-batch-size", ground_truth_cuda_batch_size,
                                     "Maximum CUDA source batch points for ground-truth computation (0 = auto)")
        ->default_val(ground_truth_cuda_batch_size);
    dataset_ground_truth->add_flag("--overwrite", ground_truth_overwrite, "Overwrite existing ground-truth values");
    dataset_ground_truth->add_flag("--no-progress", ground_truth_no_progress, "Disable progress reporting");
    dataset_ground_truth->callback([&]() {
        DatasetPrep::GroundTruthOptions options{.dataset_directory = ground_truth_dataset_directory,
                                                .ground_truth_specs = ground_truth_specs,
                                                .overwrite = ground_truth_overwrite,
                                                .show_progress = !ground_truth_no_progress,
                                                .ground_truth_device =
                                                    DatasetPrep::ParseGroundTruthDevice(ground_truth_device),
                                                .cuda_batch_size = ground_truth_cuda_batch_size};
        command = std::make_unique<DatasetGroundTruthCommand>(std::move(options));
    });

    CLI11_PARSE(app, argc, argv);

    return 0;
}
