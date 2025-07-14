#include "command.h"

#include <spdlog/spdlog.h>

#include "point_set.h"

GenerateDatasetCommand::GenerateDatasetCommand(std::string data_type_, std::filesystem::path dataset_directory,
                                               unsigned int base_num_points, unsigned int query_num_points,
                                               unsigned int num_dimensions, double left_boundary, double right_boundary,
                                               bool in_memory)
    : dataset_directory_(std::move(dataset_directory)),
      left_boundary_(left_boundary),
      right_boundary_(right_boundary),
      in_memory_(in_memory),
      gen_(std::random_device{}()) {
    dataset_metadata_ = {
        .data_type = std::move(data_type_),
        .base_num_points = base_num_points,
        .query_num_points = query_num_points,
        .num_dimensions = num_dimensions,
        .chamfer_distance = 0.0,  // Will be calculated later
    };
}

auto GenerateDatasetCommand::Execute() -> void {
    PrintConfiguration();

    // Create the directory if it does not exist
    if (!std::filesystem::exists(dataset_directory_)) {
        spdlog::info("Creating dataset directory: {}", dataset_directory_.string());
        std::filesystem::create_directory(dataset_directory_);
    }

    std::mt19937 gen(std::random_device{}());

    // Generate the base point set and the query point set
    GeneratePointSet(dataset_directory_, "base", dataset_metadata_.base_num_points);
    GeneratePointSet(dataset_directory_, "query", dataset_metadata_.query_num_points);

    // Calculate the Chamfer distance between the base and query sets
    spdlog::info("Calculating Chamfer distance between base and query sets...");
    auto base_set_reader =
        PointSetReaderFactory::Create(in_memory_, dataset_metadata_.data_type, dataset_directory_ / "base.bin",
                                      dataset_metadata_.base_num_points, dataset_metadata_.num_dimensions);
    auto query_set_reader =
        PointSetReaderFactory::Create(in_memory_, dataset_metadata_.data_type, dataset_directory_ / "query.bin",
                                      dataset_metadata_.query_num_points, dataset_metadata_.num_dimensions);

    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < dataset_metadata_.query_num_points; i++) {
        PointVariant query = query_set_reader->GetPoint(i);
        dataset_metadata_.chamfer_distance += base_set_reader->CalculateDistance(query);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    spdlog::info("Chamfer distance calculated: {}, took {:.2f} ms", dataset_metadata_.chamfer_distance,
                 elapsed.count());

    // Save the metadata to a TOML file
    dataset_metadata_.Save(dataset_directory_ / "metadata.toml");
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
In Memory: {}
-----------------------------------------------------)",
                  dataset_metadata_.data_type, dataset_directory_.string(), dataset_metadata_.base_num_points,
                  dataset_metadata_.query_num_points, dataset_metadata_.num_dimensions, left_boundary_, right_boundary_,
                  in_memory_);
}

auto GenerateDatasetCommand::GeneratePointSet(const std::filesystem::path& dataset_directory,
                                              const std::string& point_set_name, unsigned int num_points) -> void {
    spdlog::info("Generating {} point set...", point_set_name);
    std::filesystem::path point_set_file_path = dataset_directory / std::format("{}.bin", point_set_name);
    auto point_set_writer = PointSetWriterFactory::Create(in_memory_, dataset_metadata_.data_type, point_set_file_path,
                                                          dataset_metadata_.num_dimensions);

    if (dataset_metadata_.data_type == "double") {
        std::uniform_real_distribution<double> dist(left_boundary_, right_boundary_);
        for (unsigned int i = 0; i < num_points; i++) {
            Point<double> point(dataset_metadata_.num_dimensions);
            std::ranges::generate(point, [&]() { return dist(gen_); });
            point_set_writer->AddPoint(point);
        }
        point_set_writer->Flush();
    } else if (dataset_metadata_.data_type == "int") {
        std::uniform_int_distribution<int> dist(static_cast<int>(left_boundary_), static_cast<int>(right_boundary_));
        for (unsigned int i = 0; i < num_points; i++) {
            Point<int> point(dataset_metadata_.num_dimensions);
            std::ranges::generate(point, [&]() { return static_cast<int>(dist(gen_)); });
            point_set_writer->AddPoint(point);
        }
        point_set_writer->Flush();
    } else if (dataset_metadata_.data_type == "uint8") {
        std::uniform_int_distribution<uint8_t> dist(static_cast<uint8_t>(left_boundary_),
                                                    static_cast<uint8_t>(right_boundary_));
        for (unsigned int i = 0; i < num_points; i++) {
            Point<uint8_t> point(dataset_metadata_.num_dimensions);
            std::ranges::generate(point, [&]() { return dist(gen_); });
            point_set_writer->AddPoint(point);
        }
        point_set_writer->Flush();
    } else {
        throw std::invalid_argument(std::format("Unsupported data type: {}", dataset_metadata_.data_type));
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