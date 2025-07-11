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

    std::filesystem::path parent_directory_;
    std::string dataset_name_;
    unsigned int base_num_points_{0};
    unsigned int query_num_points_{0};
    unsigned int num_dimensions_{0};
    double left_boundary_{0.0};
    double right_boundary_{0.0};
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
      right_boundary_(right_boundary) {}

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

    auto generate_point_set = [&](const std::string& point_set_name, unsigned int num_points) {
        spdlog::info("Generating {} point set...", point_set_name);
        std::filesystem::path point_set_file_path = dataset_directory / std::format("{}.bin", point_set_name);
        PointSetWriter<T> point_set_writer(point_set_file_path);

        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dist(static_cast<T>(left_boundary_), static_cast<T>(right_boundary_));
            for (unsigned int i = 0; i < num_points; i++) {
                std::vector<T> point(num_dimensions_);
                std::ranges::generate(point, [&]() { return dist(gen); });
                point_set_writer.AddPoint(point);
            }
        } else if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dist(static_cast<T>(left_boundary_), static_cast<T>(right_boundary_));
            for (unsigned int i = 0; i < num_points; i++) {
                std::vector<T> point(num_dimensions_);
                std::ranges::generate(point, [&]() { return static_cast<T>(dist(gen)); });
                point_set_writer.AddPoint(point);
            }
        }
    };

    // Generate the base point set and the query point set
    generate_point_set("base", base_num_points_);
    generate_point_set("query", query_num_points_);
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

#endif