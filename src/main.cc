#include <CLI/CLI.hpp>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <thread>

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
    app.add_flag("--scalar-exact-scan", Global::kUseScalarExactScan,
                 "Ablation: find exact nearest neighbors with the scalar kernel instead of the batched L1/L2 kernels")
        ->default_str(Global::kUseScalarExactScan ? "True" : "False");
    app.add_option("--initial-radius", Global::kInitialRadius,
                   "Starting search radius R_0 of QALSH queries (2^-149 is valid for any float32 data and any c > 1)")
        ->default_val(Global::kInitialRadius)
        ->check(CLI::PositiveNumber);
    app.add_flag("--qalsh-stop-within-round{false}", Global::kQalshStopAtRoundBoundary,
                 "Check QALSH's stopping condition after every candidate instead of after each complete radius round "
                 "(legacy behaviour; the sample-size constant then uses c^2 instead of c)");
    app.add_option("--qalsh-mid-round-ratio", Global::kQalshMidRoundRatio,
                   "With round-end checks, also stop inside a round once a candidate lies within this multiple of the "
                   "radius (1 keeps the constant c; 0 waits for the end of every round)")
        ->default_val(Global::kQalshMidRoundRatio)
        ->check(CLI::NonNegativeNumber);

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
    // index
    // ------------------------------
    CLI::App* index = app.add_subcommand("index", "Build persistent indexes.");
    index->require_subcommand(1);

    CLI::App* qalsh_index = index->add_subcommand("qalsh", "Build a disk QALSH index.");
    float qalsh_index_norm_order{0.0F};
    std::filesystem::path qalsh_index_dataset_directory;
    std::string qalsh_index_target{"b"};
    unsigned int qalsh_index_seed{Global::kDefaultSeed};
    float qalsh_index_approximation_ratio{Global::kDefaultApproximationRatio};
    bool qalsh_index_overwrite{false};
    std::string qalsh_index_layout{"b-plus-tree"};
    qalsh_index->add_option("-p,--norm-order", qalsh_index_norm_order, "Norm order")
        ->required()
        ->check(CLI::IsMember({1.0, 2.0}));  // NOLINT(readability-magic-numbers)
    qalsh_index->add_option("-d,--dataset-directory", qalsh_index_dataset_directory, "Directory for the dataset")
        ->required();
    qalsh_index->add_option("--target", qalsh_index_target, "Target point set to index: a, b, or both")
        ->default_val("b")
        ->check(CLI::IsMember({"a", "b", "both"}));
    qalsh_index->add_option("--seed", qalsh_index_seed, "Seed for deterministic QALSH index vectors")
        ->default_val(Global::kDefaultSeed);
    qalsh_index
        ->add_option("-c,--approximation-ratio", qalsh_index_approximation_ratio,
                     "Approximation ratio to store in the disk QALSH index")
        ->default_val(Global::kDefaultApproximationRatio);
    qalsh_index
        ->add_option("--layout,--index-layout", qalsh_index_layout,
                     "Persistent index layout: b-plus-tree or sorted-array")
        ->default_val("b-plus-tree")
        ->check(CLI::IsMember({"b-plus-tree", "sorted-array"}));
    qalsh_index->add_flag("--overwrite", qalsh_index_overwrite, "Overwrite an existing selected target index");
    qalsh_index->callback([&]() {
        command = std::make_unique<QalshIndexCommand>(
            DiskQalsh::BuildRequest{.dataset_directory = qalsh_index_dataset_directory,
                                    .norm_order = qalsh_index_norm_order,
                                    .target_selection = DiskQalsh::ParseTargetSelection(qalsh_index_target),
                                    .seed = qalsh_index_seed,
                                    .approximation_ratio = qalsh_index_approximation_ratio,
                                    .overwrite = qalsh_index_overwrite,
                                    .layout = DiskQalsh::ParseIndexLayout(qalsh_index_layout)});
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

    bool disk_based{false};
    estimate->add_flag("--disk-based", disk_based, "Use disk-based external-memory implementations when available")
        ->default_val(false)
        ->default_str(disk_based ? "True" : "False");

    unsigned int disk_block_size{Global::kDefaultDiskBlockSize};
    estimate
        ->add_option("--disk-block-size", disk_block_size,
                     "Number of target points read per disk block in disk-based mode")
        ->default_val(Global::kDefaultDiskBlockSize)
        ->check(CLI::PositiveNumber);

    unsigned int disk_query_batch_size{Global::kDefaultDiskQueryBatchSize};
    estimate
        ->add_option("--disk-query-batch-size", disk_query_batch_size,
                     "Number of query points read per disk batch in disk-based mode")
        ->default_val(Global::kDefaultDiskQueryBatchSize)
        ->check(CLI::PositiveNumber);

    DiskQalshSearchOptions disk_qalsh_options;
    estimate
        ->add_option("--disk-qalsh-table-block-size", disk_qalsh_options.table_block_size,
                     "Legacy flat-table block size (ignored by the library persistent backend)")
        ->default_val(Global::kDefaultDiskQalshTableBlockSize)
        ->check(CLI::PositiveNumber);
    estimate
        ->add_option("--disk-qalsh-table-cache-blocks", disk_qalsh_options.table_cache_blocks,
                     "Legacy flat-table cache capacity (ignored by the library persistent backend)")
        ->default_val(Global::kDefaultDiskQalshTableCacheBlocks)
        ->check(CLI::PositiveNumber);
    estimate->add_flag("--disk-qalsh-profile", disk_qalsh_options.profile,
                       "Print disk QALSH profile; legacy flat-table cache counters are reported as unsupported")
        ->default_val(false)
        ->default_str(disk_qalsh_options.profile ? "True" : "False");

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
        estimator = std::make_unique<AnnEstimator>(std::move(ann_searcher), disk_based, disk_block_size,
                                                   disk_query_batch_size);
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
    float qalsh_ann_approximation_ratio{Global::kDefaultApproximationRatio};
    qalsh_ann
        ->add_option("-c,--approximation-ratio", qalsh_ann_approximation_ratio,
                     "Approximation ratio for QALSH")
        ->default_val(Global::kDefaultApproximationRatio);

    const CLI::Option* qalsh_ann_approximation_ratio_option = (*qalsh_ann)["--approximation-ratio"];
    qalsh_ann->callback([&] {
        if (disk_based) {
            const std::optional<float> expected_approximation_ratio =
                qalsh_ann_approximation_ratio_option->count() > 0
                    ? std::optional<float>(qalsh_ann_approximation_ratio)
                    : std::nullopt;
            ann_searcher = std::make_unique<DiskBasedQalshAnnSearcher>(expected_approximation_ratio,
                                                                       disk_qalsh_options);
            return;
        }
        ann_searcher = std::make_unique<QalshAnnSearcher>(qalsh_ann_approximation_ratio);
    });

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

    unsigned int exact_batch_size{0};
    sampling->add_option("--exact-batch-size", exact_batch_size,
                         "Batch size for stage 2 exact linear scans (0: auto)")
        ->default_val(0);

    std::unique_ptr<WeightsGenerator> weights_generator;
    sampling->require_subcommand(1);
    sampling->callback([&]() {
        if (!weights_generator) {
            Log::error("Weights generator is not set. Please specify a weights generator.");
        }
        estimator = std::make_unique<SamplingEstimator>(std::move(weights_generator), num_samples, num_samples_ab,
                                                        num_samples_ba, epsilon, delta, use_cache, exact_batch_size,
                                                        disk_based, disk_block_size, disk_query_batch_size);
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
    const CLI::Option* qalsh_sampling_approximation_ratio_option = (*qalsh_sampling)["--approximation-ratio"];
    qalsh_sampling->callback([&]() {
        if (disk_based) {
            const std::optional<float> expected_approximation_ratio =
                qalsh_sampling_approximation_ratio_option->count() > 0
                    ? std::optional<float>(qalsh_sampling_approximation_ratio)
                    : std::nullopt;
            weights_generator = std::make_unique<QalshWeightsGenerator>(expected_approximation_ratio,
                                                                        disk_query_batch_size,
                                                                        disk_qalsh_options);
            return;
        }
        weights_generator = std::make_unique<QalshWeightsGenerator>(qalsh_sampling_approximation_ratio);
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
    float gmm_outlier_fraction{0.1F};
    float gmm_outlier_scale{4096.0F};
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
