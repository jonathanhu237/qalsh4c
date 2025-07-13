#ifndef GENERATE_DATASET_COMMAND_H_
#define GENERATE_DATASET_COMMAND_H_

#include <spdlog/spdlog.h>
#include <toml++/toml.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <random>
#include <ratio>
#include <string>

#include "command.h"
#include "dataset_metadata.h"
#include "point_set.h"
#include "utils.h"

template <typename T>
class GenerateDatasetCommand : public Command {
   public:
    GenerateDatasetCommand(std::filesystem::path dataset_directory, unsigned int base_num_points,
                           unsigned int query_num_points, unsigned int num_dimensions, double left_boundary,
                           double right_boundary);
    auto Execute() -> void override;

   private:
    auto PrintConfiguration() -> void;
    auto GeneratePointSet(const std::filesystem::path& dataset_directory, const std::string& point_set_name,
                          unsigned int num_points) -> void;

    std::filesystem::path dataset_directory_;
    unsigned int base_num_points_{0};
    unsigned int query_num_points_{0};
    unsigned int num_dimensions_{0};
    double left_boundary_{0.0};
    double right_boundary_{0.0};

    std::mt19937 gen_;
};

template <typename T>
GenerateDatasetCommand<T>::GenerateDatasetCommand(std::filesystem::path dataset_directory, unsigned int base_num_points,
                                                  unsigned int query_num_points, unsigned int num_dimensions,
                                                  double left_boundary, double right_boundary)
    : dataset_directory_(std::move(dataset_directory)),
      base_num_points_(base_num_points),
      query_num_points_(query_num_points),
      num_dimensions_(num_dimensions),
      left_boundary_(left_boundary),
      right_boundary_(right_boundary),
      gen_(std::random_device{}()) {}

template <typename T>
auto GenerateDatasetCommand<T>::Execute() -> void {
    PrintConfiguration();

    // Create the directory if it does not exist
    if (!std::filesystem::exists(dataset_directory_)) {
        spdlog::info("Creating dataset directory: {}", dataset_directory_.string());
        std::filesystem::create_directory(dataset_directory_);
    }

    std::mt19937 gen(std::random_device{}());

    // Generate the base point set and the query point set
    GeneratePointSet(dataset_directory_, "base", base_num_points_);
    GeneratePointSet(dataset_directory_, "query", query_num_points_);

    // Calculate the Chamfer distance between the base and query sets
    spdlog::info("Calculating Chamfer distance between base and query sets...");
    PointSetReader<T> base_set_reader(dataset_directory_ / "base.bin", base_num_points_, num_dimensions_);
    PointSetReader<T> query_set_reader(dataset_directory_ / "query.bin", query_num_points_, num_dimensions_);

    double chamfer_distance = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < query_num_points_; i++) {
        std::vector<T> query = query_set_reader.GetPoint(i);
        chamfer_distance += base_set_reader.CalculateDistance(query);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    spdlog::info("Chamfer distance calculated: {}, took {:.2f} ms", chamfer_distance, elapsed.count());

    // Save the metadata to a TOML file
    DatasetMetadata metadata = {.base_num_points_ = base_num_points_,
                                .query_num_points_ = query_num_points_,
                                .num_dimensions_ = num_dimensions_,
                                .data_type_ = Utils::to_string<T>()};
    metadata.Save(dataset_directory_ / "metadata.toml");
}

template <typename T>
auto GenerateDatasetCommand<T>::PrintConfiguration() -> void {
    spdlog::debug(R"(The configuration is as follows:
---------- Dataset Generator Configuration ----------
Dataset Directory: {}
Number of Points in Base Set: {}
Number of Points in Query Set: {}
Number of Dimensions: {}
Left Boundary: {}
Right Boundary: {}
-----------------------------------------------------)",
                  Utils::to_string<T>(), dataset_directory_.string(), base_num_points_, query_num_points_,
                  num_dimensions_, left_boundary_, right_boundary_);
}

template <typename T>
auto GenerateDatasetCommand<T>::GeneratePointSet(const std::filesystem::path& dataset_directory,
                                                 const std::string& point_set_name, unsigned int num_points) -> void {
    spdlog::info("Generating {} point set...", point_set_name);
    std::filesystem::path point_set_file_path = dataset_directory / std::format("{}.bin", point_set_name);
    PointSetWriter<T> point_set_writer(point_set_file_path);

    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(static_cast<T>(left_boundary_), static_cast<T>(right_boundary_));
        for (unsigned int i = 0; i < num_points; i++) {
            std::vector<T> point(num_dimensions_);
            std::ranges::generate(point, [&]() { return dist(gen_); });
            point_set_writer.AddPoint(point);
        }
    } else if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(static_cast<T>(left_boundary_), static_cast<T>(right_boundary_));
        for (unsigned int i = 0; i < num_points; i++) {
            std::vector<T> point(num_dimensions_);
            std::ranges::generate(point, [&]() { return static_cast<T>(dist(gen_)); });
            point_set_writer.AddPoint(point);
        }
    }
    spdlog::info("The {} point set has been generated and saved to {}", point_set_name, point_set_file_path.string());
}

#endif