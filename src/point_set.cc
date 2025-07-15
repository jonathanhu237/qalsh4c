#include "point_set.h"

std::unique_ptr<PointSetWriter> PointSetWriterFactory::Create(bool in_memory, const std::string& data_type,
                                                              const std::filesystem::path& file_path,
                                                              unsigned int num_dimensions) {
    if (in_memory) {
        if (data_type == "uint8_t") {
            return std::make_unique<InMemoryPointSetWriter<uint8_t>>(file_path, num_dimensions);
        }
        if (data_type == "int") {
            return std::make_unique<InMemoryPointSetWriter<int>>(file_path, num_dimensions);
        }
        if (data_type == "double") {
            return std::make_unique<InMemoryPointSetWriter<double>>(file_path, num_dimensions);
        }
    } else {
        if (data_type == "uint8_t") {
            return std::make_unique<DiskPointSetWriter<uint8_t>>(file_path, num_dimensions);
        }
        if (data_type == "int") {
            return std::make_unique<DiskPointSetWriter<int>>(file_path, num_dimensions);
        }
        if (data_type == "double") {
            return std::make_unique<DiskPointSetWriter<double>>(file_path, num_dimensions);
        }
    }
    throw std::invalid_argument("Unsupported data type or configuration");
}

std::unique_ptr<PointSetReader> PointSetReaderFactory::Create(bool in_memory, const std::string& data_type,
                                                              const std::filesystem::path& file_path,
                                                              unsigned int num_points, unsigned int num_dimensions) {
    if (in_memory) {
        if (data_type == "uint8_t") {
            return std::make_unique<InMemoryPointSetReader<uint8_t>>(file_path, num_points, num_dimensions);
        }
        if (data_type == "int") {
            return std::make_unique<InMemoryPointSetReader<int>>(file_path, num_points, num_dimensions);
        }
        if (data_type == "double") {
            return std::make_unique<InMemoryPointSetReader<double>>(file_path, num_points, num_dimensions);
        }
    } else {
        if (data_type == "uint8_t") {
            return std::make_unique<DiskPointSetReader<uint8_t>>(file_path, num_points, num_dimensions);
        }
        if (data_type == "int") {
            return std::make_unique<DiskPointSetReader<int>>(file_path, num_points, num_dimensions);
        }
        if (data_type == "double") {
            return std::make_unique<DiskPointSetReader<double>>(file_path, num_points, num_dimensions);
        }
    }
    throw std::invalid_argument("Unsupported data type or configuration");
}