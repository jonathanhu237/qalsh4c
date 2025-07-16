#include "dataset_generator.h"

#include <spdlog/spdlog.h>

#include <utility>

#include "point_set.h"
#include "types.h"

// ---------------------------------------------
// DatasetSynthesizer Implementation
// ---------------------------------------------

DatasetSynthesizer::DatasetSynthesizer(DatasetMetadata dataset_metadata, double left_boundary, double right_boundary,
                                       bool in_memory)
    : dataset_metadata_(std::move(dataset_metadata)),
      left_boundary_(left_boundary),
      right_boundary_(right_boundary),
      in_memory_(in_memory),
      gen_(std::random_device{}()) {
    spdlog::debug(
        "The configuration is as follows:\n"
        "    Data Type: {}\n"
        "    Num Points (Base Set): {}\n"
        "    Num Points (Query Set): {}\n"
        "    Number of Dimensions: {}\n"
        "    Left Boundary: {}\n"
        "    Right Boundary: {}\n"
        "    In Memory: {}",
        dataset_metadata_.data_type, dataset_metadata_.base_num_points, dataset_metadata_.query_num_points,
        dataset_metadata_.num_dimensions, left_boundary_, right_boundary_, in_memory_);
}

void DatasetSynthesizer::Generate(const std::filesystem::path &dataset_directory) {
    // Create the directory if it does not exist
    if (!std::filesystem::exists(dataset_directory)) {
        spdlog::info("Creating dataset directory: {}", dataset_directory.string());
        std::filesystem::create_directory(dataset_directory);
    }

    std::mt19937 gen(std::random_device{}());

    // Generate the base point set and the query point set
    spdlog::info("Generating base point set...");
    GeneratePointSet(dataset_directory, "base", dataset_metadata_.base_num_points);

    spdlog::info("Generating query point set...");
    GeneratePointSet(dataset_directory, "query", dataset_metadata_.query_num_points);

    // Calculate the Chamfer distance between the base and query sets
    spdlog::info("Calculating Chamfer distance between base and query sets...");
    auto base_set_reader =
        PointSetReaderFactory::Create(dataset_directory / "base.bin", dataset_metadata_.data_type,
                                      dataset_metadata_.base_num_points, dataset_metadata_.num_dimensions, in_memory_);
    auto query_set_reader =
        PointSetReaderFactory::Create(dataset_directory / "query.bin", dataset_metadata_.data_type,
                                      dataset_metadata_.query_num_points, dataset_metadata_.num_dimensions, in_memory_);

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
    dataset_metadata_.Save(dataset_directory / "metadata.toml");
}

void DatasetSynthesizer::GeneratePointSet(const std::filesystem::path &dataset_directory,
                                          const std::string &point_set_name, unsigned int num_points) {
    std::filesystem::path point_set_file_path = dataset_directory / std::format("{}.bin", point_set_name);
    auto point_set_writer = PointSetWriterFactory::Create(point_set_file_path, dataset_metadata_.data_type,
                                                          dataset_metadata_.num_dimensions, in_memory_);

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
    } else if (dataset_metadata_.data_type == "uint8") {
        std::uniform_int_distribution<uint8_t> dist(static_cast<uint8_t>(left_boundary_),
                                                    static_cast<uint8_t>(right_boundary_));
        for (unsigned int i = 0; i < num_points; i++) {
            Point<uint8_t> point(dataset_metadata_.num_dimensions);
            std::ranges::generate(point, [&]() { return dist(gen_); });
            point_set_writer->AddPoint(point);
        }
    } else {
        spdlog::critical(std::format("Unsupported data type: {}", dataset_metadata_.data_type));
    }

    point_set_writer->Flush();
}
