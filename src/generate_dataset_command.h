#ifndef GENERATE_DATASET_COMMAND_H_
#define GENERATE_DATASET_COMMAND_H_

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <random>
#include <string>

#include "command.h"
#include "point_set.h"
#include "utils.h"

template <typename T>
class GenerateDatasetCommand : public Command {
   public:
    GenerateDatasetCommand(std::filesystem::path parent_directory, std::string dataset_name,
                           unsigned int base_num_points, unsigned int query_num_points, unsigned int num_dimensions,
                           double left_boundary, double right_boundary);
    auto Execute() -> void override;

   private:
    auto PrintConfiguration() -> void;
    auto GeneratePointSet(const std::filesystem::path& dataset_directory, const std::string& point_set_name,
                          unsigned int num_points) -> void;

    std::filesystem::path parent_directory_;
    std::string dataset_name_;
    unsigned int base_num_points_{0};
    unsigned int query_num_points_{0};
    unsigned int num_dimensions_{0};
    double left_boundary_{0.0};
    double right_boundary_{0.0};

    std::mt19937 gen_;
};

template <typename T>
GenerateDatasetCommand<T>::GenerateDatasetCommand(std::filesystem::path parent_directory, std::string dataset_name,
                                                  unsigned int base_num_points, unsigned int query_num_points,
                                                  unsigned int num_dimensions, double left_boundary,
                                                  double right_boundary)
    : parent_directory_(std::move(parent_directory)),
      dataset_name_(std::move(dataset_name)),
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
    std::filesystem::path dataset_directory = parent_directory_ / dataset_name_;
    if (!std::filesystem::exists(dataset_directory)) {
        spdlog::info("Creating dataset directory: {}", dataset_directory.string());
        std::filesystem::create_directory(dataset_directory);
    }

    std::mt19937 gen(std::random_device{}());

    // Generate the base point set and the query point set
    GeneratePointSet(dataset_directory, "base", base_num_points_);
    GeneratePointSet(dataset_directory, "query", query_num_points_);

    // Calculate the Chamfer distance between the base and query sets
    spdlog::info("Calculating Chamfer distance between base and query sets...");
    PointSetReader<T> base_set_reader(dataset_directory / "base.bin", base_num_points_, num_dimensions_);
    PointSetReader<T> query_set_reader(dataset_directory / "query.bin", query_num_points_, num_dimensions_);

    double chamfer_distance = 0;
    for (unsigned int i = 0; i < query_num_points_; i++) {
        std::vector<T> query = query_set_reader.GetPoint(i);
        chamfer_distance += base_set_reader.CalculateDistance(query);
    }
    spdlog::info("Chamfer distance calculated: {}", chamfer_distance);
}

template <typename T>
auto GenerateDatasetCommand<T>::PrintConfiguration() -> void {
    spdlog::debug(R"(The configuration is as follows:
---------- Dataset Generator Configuration ----------
Data Type: {}
Data Directory: {}
Dataset Name: {}
Number of Points in Base Set: {}
Number of Points in Query Set: {}
Number of Dimensions: {}
Left Boundary: {}
Right Boundary: {}
-----------------------------------------------------)",
                  Utils::to_string<T>(), parent_directory_.string(), dataset_name_, base_num_points_, query_num_points_,
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