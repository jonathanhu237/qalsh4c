#include "point_set.h"

#include <spdlog/spdlog.h>

#include <utility>

#include "global.h"

std::unique_ptr<PointSetWriter> PointSetWriterFactory::Create(const std::filesystem::path& file_path,
                                                              const std::string& data_type,
                                                              unsigned int num_dimensions) {
    if (Global::high_memory_mode) {
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
    spdlog::error("Unsupported data type or configuration");
    return nullptr;
}

std::unique_ptr<PointSetReader> PointSetReaderFactory::Create(const std::filesystem::path& file_path,
                                                              const std::string& data_type, unsigned int num_points,
                                                              unsigned int num_dimensions) {
    if (Global::high_memory_mode) {
        if (data_type == "uint8_t") {
            return std::make_unique<InMemoryPointSetReader<uint8_t>>(file_path, num_points, num_dimensions);
        }
        if (data_type == "int") {
            return std::make_unique<InMemoryPointSetReader<int>>(file_path, num_points, num_dimensions);
        }
        if (data_type == "float") {
            return std::make_unique<InMemoryPointSetReader<double>>(file_path, num_points, num_dimensions);
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
        if (data_type == "float") {
            return std::make_unique<InMemoryPointSetReader<double>>(file_path, num_points, num_dimensions);
        }
        if (data_type == "double") {
            return std::make_unique<DiskPointSetReader<double>>(file_path, num_points, num_dimensions);
        }
    }
    spdlog::error("Unsupported data type or configuration");
    return nullptr;
}

// ---------------------------------------------
// CombinePointSetReader Implementation
// ---------------------------------------------
CombinePointSetReader::CombinePointSetReader(std::unique_ptr<PointSetReader> base_reader,
                                             std::unique_ptr<PointSetReader> query_reader)
    : base_reader_(std::move(base_reader)),
      query_reader_(std::move(query_reader)),
      num_points_(base_reader_->get_num_points() + query_reader_->get_num_dimensions()),
      num_dimensions_(base_reader_->get_num_dimensions()) {}

unsigned int CombinePointSetReader::get_num_points() const { return num_points_; }

unsigned int CombinePointSetReader::get_num_dimensions() const { return num_dimensions_; }

PointVariant CombinePointSetReader::GetPoint(unsigned int index) {
    if (index >= num_points_) {
        spdlog::error("Point index is out of range for the combined dataset.");
        // The program will terminate automatically, so we do not return manually here.
    }

    if (index < base_reader_->get_num_points()) {
        return base_reader_->GetPoint(index);
    }
    return query_reader_->GetPoint(index - base_reader_->get_num_points());
}