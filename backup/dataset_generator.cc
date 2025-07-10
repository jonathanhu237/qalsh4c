#include "dataset_generator.h"

#include <Eigen/Dense>
#include <filesystem>
#include <stdexcept>

namespace qalsh_chamfer {

namespace fs = std::filesystem;

// ---------- DatasetGeneratorBuilder Implementation ----------
auto DatasetGeneratorFactory::set_dataset_name(const std::string& dataset_name) -> DatasetGeneratorFactory& {
    dataset_name_ = dataset_name;
    return *this;
}

auto DatasetGeneratorFactory::set_parent_directory(const fs::path& parent_directory) -> DatasetGeneratorFactory& {
    parent_directory_ = parent_directory;
    return *this;
}

auto DatasetGeneratorFactory::set_num_points(unsigned int num_points) -> DatasetGeneratorFactory& {
    num_points_ = num_points;
    return *this;
}

auto DatasetGeneratorFactory::set_num_dimensions(unsigned int num_dimensions) -> DatasetGeneratorFactory& {
    num_dimensions_ = num_dimensions;
    return *this;
}

auto DatasetGeneratorFactory::set_left_boundary(int left_boundary) -> DatasetGeneratorFactory& {
    left_boundary_ = left_boundary;
    return *this;
}

auto DatasetGeneratorFactory::set_right_boundary(int right_boundary) -> DatasetGeneratorFactory& {
    right_boundary_ = right_boundary;
    return *this;
}

auto DatasetGeneratorFactory::set_verbose(bool verbose) -> DatasetGeneratorFactory& {
    verbose_ = verbose;
    return *this;
}

auto DatasetGeneratorFactory::Build(const std::string& type) const -> std::unique_ptr<IDatasetGenerator> {
    if (type == "int8") {
        return std::unique_ptr<IDatasetGenerator>(new DatasetGenerator<int8_t>(
            dataset_name_, parent_directory_, num_points_, num_dimensions_, left_boundary_, right_boundary_, verbose_));
    }
    if (type == "int16") {
        return std::unique_ptr<IDatasetGenerator>(new DatasetGenerator<int16_t>(
            dataset_name_, parent_directory_, num_points_, num_dimensions_, left_boundary_, right_boundary_, verbose_));
    }
    if (type == "int32") {
        return std::unique_ptr<IDatasetGenerator>(new DatasetGenerator<int32_t>(
            dataset_name_, parent_directory_, num_points_, num_dimensions_, left_boundary_, right_boundary_, verbose_));
    }
    if (type == "int64") {
        return std::unique_ptr<IDatasetGenerator>(new DatasetGenerator<int64_t>(
            dataset_name_, parent_directory_, num_points_, num_dimensions_, left_boundary_, right_boundary_, verbose_));
    }
    if (type == "float") {
        return std::unique_ptr<IDatasetGenerator>(new DatasetGenerator<float>(
            dataset_name_, parent_directory_, num_points_, num_dimensions_, left_boundary_, right_boundary_, verbose_));
    }
    if (type == "double") {
        return std::unique_ptr<IDatasetGenerator>(new DatasetGenerator<double>(
            dataset_name_, parent_directory_, num_points_, num_dimensions_, left_boundary_, right_boundary_, verbose_));
    }
    throw std::runtime_error(std::format("Unsupported dataset type: {}", type));
}

};  // namespace qalsh_chamfer