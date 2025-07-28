#include "dataset_generator.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <memory>
#include <utility>
#include <vector>

#include "ann_searcher.h"
#include "estimator.h"
#include "global.h"
#include "point_set.h"
#include "types.h"
#include "utils.h"

// ---------------------------------------------
// DatasetSynthesizer Implementation
// ---------------------------------------------
DatasetSynthesizer::DatasetSynthesizer(DatasetMetadata dataset_metadata, double left_boundary, double right_boundary)
    : dataset_metadata_(std::move(dataset_metadata)),
      left_boundary_(left_boundary),
      right_boundary_(right_boundary),
      gen_(std::random_device{}()) {}

void DatasetSynthesizer::Generate(const std::filesystem::path &dataset_directory) {
    // Create the directory if it does not exist
    if (!std::filesystem::exists(dataset_directory)) {
        spdlog::info("Creating dataset directory: {}", dataset_directory.string());
        std::filesystem::create_directory(dataset_directory);
    }

    std::mt19937 gen(std::random_device{}());

    // We must save the metadata first since it will be used by the estimator (update the chamfer distance later)
    dataset_metadata_.Save(dataset_directory / "metadata.toml");

    // Generate the base point set and the query point set
    spdlog::info("Generating base point set...");
    GeneratePointSet(dataset_directory, "base", dataset_metadata_.base_num_points);

    spdlog::info("Generating query point set...");
    GeneratePointSet(dataset_directory, "query", dataset_metadata_.query_num_points);

    // Calculate the Chamfer distance between the base and query sets
    spdlog::info("Calculating Chamfer distance between base and query sets...");
    auto base_set_reader =
        PointSetReaderFactory::Create(dataset_directory / "base.bin", dataset_metadata_.data_type,
                                      dataset_metadata_.base_num_points, dataset_metadata_.num_dimensions);
    auto query_set_reader =
        PointSetReaderFactory::Create(dataset_directory / "query.bin", dataset_metadata_.data_type,
                                      dataset_metadata_.query_num_points, dataset_metadata_.num_dimensions);

    auto start = std::chrono::high_resolution_clock::now();
    AnnEstimator ann_estimator(std::make_unique<LinearScanAnnSearcher>());
    dataset_metadata_.chamfer_distance = ann_estimator.Estimate(dataset_directory);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    spdlog::info("Chamfer distance calculated: {}, took {:.2f} ms", dataset_metadata_.chamfer_distance,
                 elapsed.count());

    // Save the updated metadata
    dataset_metadata_.Save(dataset_directory / "metadata.toml");
}

void DatasetSynthesizer::GeneratePointSet(const std::filesystem::path &dataset_directory,
                                          const std::string &point_set_name, unsigned int num_points) {
    std::filesystem::path point_set_file_path = dataset_directory / std::format("{}.bin", point_set_name);
    auto point_set_writer = PointSetWriterFactory::Create(point_set_file_path, dataset_metadata_.data_type,
                                                          dataset_metadata_.num_dimensions);

    if (dataset_metadata_.data_type == "double") {
        std::uniform_real_distribution<double> dist(left_boundary_, right_boundary_);
        for (unsigned int i = 0; i < num_points; i++) {
            Point<double> point(dataset_metadata_.num_dimensions);
            std::ranges::generate(point, [&]() { return dist(gen_); });
            point_set_writer->AddPoint(point);
        }
    } else if (dataset_metadata_.data_type == "int") {
        std::uniform_int_distribution<int> dist(static_cast<int>(left_boundary_), static_cast<int>(right_boundary_));
        for (unsigned int i = 0; i < num_points; i++) {
            Point<int> point(dataset_metadata_.num_dimensions);
            std::ranges::generate(point, [&]() { return static_cast<int>(dist(gen_)); });
            point_set_writer->AddPoint(point);
        }
    } else if (dataset_metadata_.data_type == "float") {
        std::uniform_real_distribution<float> dist(static_cast<float>(left_boundary_),
                                                   static_cast<float>(right_boundary_));
        for (unsigned int i = 0; i < num_points; i++) {
            Point<float> point(dataset_metadata_.num_dimensions);
            std::ranges::generate(point, [&]() { return dist(gen_); });
            point_set_writer->AddPoint(point);
        }
    } else if (dataset_metadata_.data_type == "uint8") {
        std::uniform_int_distribution<uint8_t> dist(static_cast<uint8_t>(left_boundary_),
                                                    static_cast<uint8_t>(right_boundary_));
        for (unsigned int i = 0; i < num_points; i++) {
            Point<uint8_t> point(dataset_metadata_.num_dimensions);
            std::ranges::generate(point, [&]() { return dist(gen_); });
            point_set_writer->AddPoint(point);
        }
    } else {
        spdlog::error(std::format("Unsupported data type: {}", dataset_metadata_.data_type));
    }

    point_set_writer->Flush();
}

// ---------------------------------------------
// DatasetConverter Implementation
// ---------------------------------------------
DatasetConverter::DatasetConverter(std::string dataset_name, std::filesystem::path raw_dataset_directory,
                                   unsigned int query_num_points)
    : dataset_name_(std::move(dataset_name)),
      raw_dataset_directory_(std::move(raw_dataset_directory)),
      query_num_points_(query_num_points) {}

void DatasetConverter::Generate(const std::filesystem::path &dataset_directory) {
    output_dataset_directory_ = dataset_directory;

    if (dataset_name_ == "sift") {
        spdlog::info("Converting SIFT dataset ....");
        query_num_points_ = query_num_points_ == 0 ? Global::kSiftNumPoints / 2 : query_num_points_;
        dataset_metadata_ = DatasetMetadata{
            .data_type = std::string(Global::kSiftDataType),
            .base_num_points = Global::kSiftNumPoints - query_num_points_,
            .query_num_points = query_num_points_,
            .num_dimensions = Global::kSiftNumDimensions,
            .chamfer_distance = 0  // Update later
        };
        ConvertSift();
    } else if (dataset_name_ == "gist") {
        spdlog::info("Converting GIST dataset ....");
        query_num_points_ = query_num_points_ == 0 ? Global::kGistNumPoints / 2 : query_num_points_;
        dataset_metadata_ = DatasetMetadata{
            .data_type = std::string(Global::kGistDataType),
            .base_num_points = Global::kGistNumPoints - query_num_points_,
            .query_num_points = query_num_points_,
            .num_dimensions = Global::kGistNumDimensions,
            .chamfer_distance = 0  // Update later
        };
        ConvertGist();
    } else if (dataset_name_ == "trevi") {
        spdlog::info("Converting TREVI dataset ....");
        query_num_points_ = query_num_points_ == 0 ? Global::kTreviNumPoints / 2 : query_num_points_;
        dataset_metadata_ = DatasetMetadata{
            .data_type = std::string(Global::kTreviDataType),
            .base_num_points = Global::kTreviNumPoints - query_num_points_,
            .query_num_points = query_num_points_,
            .num_dimensions = Global::kTreviNumDimensions,
            .chamfer_distance = 0  // Update later
        };
        ConvertTrevi();
    } else if (dataset_name_ == "p53") {
        spdlog::info("Converting P53 dataset ....");
        query_num_points_ = query_num_points_ == 0 ? Global::kP53NumPoints / 2 : query_num_points_;
        dataset_metadata_ = DatasetMetadata{
            .data_type = std::string(Global::kP53DataType),
            .base_num_points = Global::kP53NumPoints - query_num_points_,
            .query_num_points = query_num_points_,
            .num_dimensions = Global::kP53NumDimensions,
            .chamfer_distance = 0  // Update later
        };
        ConvertP53();
    } else {
        spdlog::error(std::format("Unsupported dataset: {}", dataset_name_));
    }

    // We must save the metadata first since it will be used by the estimator (update the chamfer distance later)
    dataset_metadata_.Save(output_dataset_directory_ / "metadata.toml");

    // Calculate the Chamfer distance between the base and query sets
    spdlog::info("Calculating Chamfer distance between base and query sets...");
    auto base_set_reader =
        PointSetReaderFactory::Create(output_dataset_directory_ / "base.bin", dataset_metadata_.data_type,
                                      dataset_metadata_.base_num_points, dataset_metadata_.num_dimensions);
    auto query_set_reader =
        PointSetReaderFactory::Create(output_dataset_directory_ / "query.bin", dataset_metadata_.data_type,
                                      dataset_metadata_.query_num_points, dataset_metadata_.num_dimensions);

    auto start = std::chrono::high_resolution_clock::now();
    AnnEstimator ann_estimator(std::make_unique<LinearScanAnnSearcher>());
    dataset_metadata_.chamfer_distance = ann_estimator.Estimate(output_dataset_directory_);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    spdlog::info("Chamfer distance calculated: {}, took {:.2f} ms", dataset_metadata_.chamfer_distance,
                 elapsed.count());

    // Save the updated metadata
    dataset_metadata_.Save(output_dataset_directory_ / "metadata.toml");
}

void DatasetConverter::ConvertTexmexDataset() {
    const std::filesystem::path original_base_path =
        raw_dataset_directory_ / std::format("{}_base.fvecs", dataset_name_);
    const std::filesystem::path output_base_path = output_dataset_directory_ / "base.bin";
    const std::filesystem::path output_query_path = output_dataset_directory_ / "query.bin";

    std::vector<std::vector<float>> original_base = Utils::ReadFvecs(original_base_path);
    if (!CheckPoints(original_base, dataset_metadata_.base_num_points + dataset_metadata_.query_num_points,
                     dataset_metadata_.num_dimensions)) {
        spdlog::error("{} is invalid", original_base_path.string());
    }

    // Sample queries from the dataset.
    spdlog::info("Spliting dataset...");
    auto [base_set, query_set] = SplitDataset(original_base, dataset_metadata_.query_num_points);

    // We need to convert the float points to uint8 points in sift dataset
    if (dataset_name_ == "sift") {
        spdlog::info("Converting SIFT dataset into uint8 format...");
        std::vector<std::vector<uint8_t>> base_set_uint8;
        std::vector<std::vector<uint8_t>> query_set_uint8;

        for (auto &point : base_set) {
            std::vector<uint8_t> point_uint8;
            point_uint8.reserve(point.size());
            for (auto &value : point) {
                point_uint8.push_back(static_cast<uint8_t>(value));
            }
            base_set_uint8.emplace_back(std::move(point_uint8));
        }

        for (auto &point : query_set) {
            std::vector<uint8_t> point_uint8;
            point_uint8.reserve(point.size());
            for (auto &value : point) {
                point_uint8.push_back(static_cast<uint8_t>(value));
            }
            query_set_uint8.emplace_back(std::move(point_uint8));
        }

        spdlog::info("Writing points...");
        Utils::WritePoints<uint8_t>(base_set_uint8, output_base_path);
        Utils::WritePoints<uint8_t>(query_set_uint8, output_query_path);
    } else {
        spdlog::info("Writing points...");
        Utils::WritePoints<float>(base_set, output_base_path);
        Utils::WritePoints<float>(query_set, output_query_path);
    }
}

void DatasetConverter::ConvertSift() { ConvertTexmexDataset(); }

void DatasetConverter::ConvertGist() { ConvertTexmexDataset(); }

void DatasetConverter::ConvertTrevi() {
    const std::filesystem::path raw_dir_path = raw_dataset_directory_;
    const std::filesystem::path output_base_path = output_dataset_directory_ / "base.bin";
    const std::filesystem::path output_query_path = output_dataset_directory_ / "query.bin";

    constexpr size_t kPatchDim = 64;
    constexpr size_t kPatchSize = kPatchDim * kPatchDim;

    std::vector<std::vector<uint8_t>> all_patches;

    for (const auto &entry : std::filesystem::directory_iterator(raw_dir_path)) {
        if (entry.path().extension() == ".bmp") {
            const std::string bmp_file = entry.path().string();
            std::vector<std::vector<uint8_t>> pixel_data = Utils::ReadBmpGrayscale(bmp_file);

            const size_t img_height = pixel_data.size();
            const size_t img_width = pixel_data[0].size();

            for (size_t i = 0; (i + kPatchDim) <= img_height; i += kPatchDim) {
                for (size_t j = 0; (j + kPatchDim) <= img_width; j += kPatchDim) {
                    std::vector<uint8_t> patch;
                    patch.reserve(kPatchSize);
                    for (size_t row = 0; row < kPatchDim; ++row) {
                        const auto &pixel_row = pixel_data[i + row];
                        for (size_t col = 0; col < kPatchDim; ++col) {
                            patch.push_back(pixel_row[j + col]);
                        }
                    }
                    all_patches.emplace_back(std::move(patch));
                }
            }
        }
    }

    auto [dataset, queries] = SplitDataset(all_patches, dataset_metadata_.query_num_points);

    if (!CheckPoints(dataset, dataset_metadata_.base_num_points, dataset_metadata_.num_dimensions)) {
        throw std::runtime_error("The converted trevi dataset is invalid.");
    }

    if (!CheckPoints(queries, dataset_metadata_.query_num_points, dataset_metadata_.num_dimensions)) {
        throw std::runtime_error("The converted trevi queries are invalid.");
    }

    Utils::WritePoints(dataset, output_base_path.string());
    Utils::WritePoints(queries, output_query_path.string());
}

void DatasetConverter::ConvertP53() {
    const std::filesystem::path raw_dir_path = raw_dataset_directory_;
    const std::filesystem::path output_dir_path = output_dataset_directory_;
    const std::filesystem::path raw_dataset_path = raw_dir_path / "K9.data";
    const std::filesystem::path output_ds_path = output_dir_path / "base.bin";
    const std::filesystem::path output_q_path = output_dir_path / "query.bin";

    std::ifstream ifs(raw_dataset_path);
    if (!ifs.is_open()) {
        throw std::runtime_error(std::format("Failed to open {}", raw_dataset_path.string()));
    }

    std::string line;
    std::vector<std::vector<float>> original_data;

    while (std::getline(ifs, line)) {
        std::vector<float> point;
        std::stringstream ss(line);
        std::string token;
        bool skip_row = false;
        uint32_t col = 0;

        while (std::getline(ss, token, ',')) {
            if (token == "?") {
                skip_row = true;
                break;
            }
            if (col == Global::kP53NumDimensions) {
                // The last token is the class label and should be ignored
                break;
            }
            point.push_back(std::stof(token));
            ++col;
        }

        if (skip_row) {
            continue;
        }

        original_data.emplace_back(std::move(point));
    }

    ifs.close();

    // Use the utility function to split the dataset
    auto [dataset, queries] = SplitDataset(original_data, dataset_metadata_.query_num_points);

    if (!CheckPoints(dataset, dataset_metadata_.base_num_points, dataset_metadata_.num_dimensions)) {
        throw std::runtime_error("The converted trevi dataset is invalid.");
    }

    if (!CheckPoints(queries, dataset_metadata_.query_num_points, dataset_metadata_.num_dimensions)) {
        throw std::runtime_error("The converted trevi queries are invalid.");
    }

    Utils::WritePoints(dataset, output_ds_path.string());
    Utils::WritePoints(queries, output_q_path.string());
}