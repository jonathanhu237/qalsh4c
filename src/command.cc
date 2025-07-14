#include "command.h"

#include <spdlog/spdlog.h>
#include <sys/types.h>

#include <any>
#include <cstdint>

#include "dataset_metadata.h"
#include "point_set.h"

GenerateDatasetCommand::GenerateDatasetCommand(std::string data_type_, std::filesystem::path dataset_directory,
                                               unsigned int base_num_points, unsigned int query_num_points,
                                               unsigned int num_dimensions, double left_boundary, double right_boundary)
    : data_type_(std::move(data_type_)),
      dataset_directory_(std::move(dataset_directory)),
      base_num_points_(base_num_points),
      query_num_points_(query_num_points),
      num_dimensions_(num_dimensions),
      left_boundary_(left_boundary),
      right_boundary_(right_boundary),
      gen_(std::random_device{}()) {}

auto GenerateDatasetCommand::Execute() -> void {
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
    auto base_set_reader =
        PointSetReaderFactory::Create(data_type_, dataset_directory_ / "base.bin", base_num_points_, num_dimensions_);
    auto query_set_reader =
        PointSetReaderFactory::Create(data_type_, dataset_directory_ / "query.bin", query_num_points_, num_dimensions_);

    double chamfer_distance = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < query_num_points_; i++) {
        std::any query = query_set_reader->GetPoint(i);
        chamfer_distance += base_set_reader->CalculateDistance(query);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    spdlog::info("Chamfer distance calculated: {}, took {:.2f} ms", chamfer_distance, elapsed.count());

    // Save the metadata to a TOML file
    DatasetMetadata metadata = {.data_type_ = data_type_,
                                .base_num_points_ = base_num_points_,
                                .query_num_points_ = query_num_points_,
                                .num_dimensions_ = num_dimensions_,
                                .chamfer_distance_ = chamfer_distance};
    metadata.Save(dataset_directory_ / "metadata.toml");
}

auto GenerateDatasetCommand::PrintConfiguration() -> void {
    spdlog::debug(R"(The configuration is as follows:
---------- Dataset Generator Configuration ----------
Data Type: {}
Dataset Directory: {}
Number of Points in Base Set: {}
Number of Points in Query Set: {}
Number of Dimensions: {}
Left Boundary: {}
Right Boundary: {}
-----------------------------------------------------)",
                  data_type_, dataset_directory_.string(), base_num_points_, query_num_points_, num_dimensions_,
                  left_boundary_, right_boundary_);
}

auto GenerateDatasetCommand::GeneratePointSet(const std::filesystem::path& dataset_directory,
                                              const std::string& point_set_name, unsigned int num_points) -> void {
    spdlog::info("Generating {} point set...", point_set_name);
    std::filesystem::path point_set_file_path = dataset_directory / std::format("{}.bin", point_set_name);
    auto point_set_writer = PointSetWriterFactory::Create(data_type_, point_set_file_path, num_dimensions_);

    if (data_type_ == "double") {
        std::uniform_real_distribution<double> dist(left_boundary_, right_boundary_);
        for (unsigned int i = 0; i < num_points; i++) {
            std::vector<double> point(num_dimensions_);
            std::ranges::generate(point, [&]() { return dist(gen_); });
            point_set_writer->AddPoint(point);
        }
    } else if (data_type_ == "int") {
        std::uniform_int_distribution<int> dist(static_cast<int>(left_boundary_), static_cast<int>(right_boundary_));
        for (unsigned int i = 0; i < num_points; i++) {
            std::vector<int> point(num_dimensions_);
            std::ranges::generate(point, [&]() { return static_cast<int>(dist(gen_)); });
            point_set_writer->AddPoint(point);
        }
    } else if (data_type_ == "uint8") {
        std::uniform_int_distribution<uint8_t> dist(static_cast<uint8_t>(left_boundary_),
                                                    static_cast<uint8_t>(right_boundary_));
        for (unsigned int i = 0; i < num_points; i++) {
            std::vector<uint8_t> point(num_dimensions_);
            std::ranges::generate(point, [&]() { return dist(gen_); });
            point_set_writer->AddPoint(point);
        }
    } else {
        throw std::invalid_argument(std::format("Unsupported data type: {}", data_type_));
    }
    spdlog::info("The {} point set has been generated and saved to {}", point_set_name, point_set_file_path.string());
}

IndexCommand::IndexCommand(std::unique_ptr<Indexer> indexer) : indexer_(std::move(indexer)) {}

auto IndexCommand::Execute() -> void {
    if (indexer_ == nullptr) {
        throw std::runtime_error("Indexer is not set.");
    }

    indexer_->BuildIndex();
};