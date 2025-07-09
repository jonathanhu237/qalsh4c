#ifndef DATASET_GENERATOR_H_
#define DATASET_GENERATOR_H_

#include <Eigen/Dense>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

namespace qalsh_chamfer {

// ---------- Declaration ----------
class IDatasetGenerator {
   public:
    virtual ~IDatasetGenerator() = default;

    IDatasetGenerator(const IDatasetGenerator&) = delete;
    auto operator=(const IDatasetGenerator&) -> IDatasetGenerator& = delete;
    IDatasetGenerator(IDatasetGenerator&&) = delete;
    auto operator=(IDatasetGenerator&&) -> IDatasetGenerator& = delete;

    virtual void PrintConfiguration() const = 0;
    virtual void Generate() const = 0;

   protected:
    IDatasetGenerator() = default;
};

template <typename T>
class DatasetGenerator : public IDatasetGenerator {
   public:
    DatasetGenerator(std::string dataset_name, fs::path parent_directory, unsigned int num_points,
                     unsigned int num_dimensions, int left_boundary, int right_boundary, bool verbose);

    void PrintConfiguration() const override;
    void Generate() const override;

   private:
    std::string dataset_name_;
    fs::path parent_directory_;
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    int left_boundary_{0};
    int right_boundary_{0};
    bool verbose_{false};
};

class DatasetGeneratorFactory {
   public:
    DatasetGeneratorFactory() = default;

    auto set_dataset_name(const std::string& dataset_name) -> DatasetGeneratorFactory&;
    auto set_parent_directory(const fs::path& parent_directory) -> DatasetGeneratorFactory&;
    auto set_num_points(unsigned int num_points) -> DatasetGeneratorFactory&;
    auto set_num_dimensions(unsigned int num_dimensions) -> DatasetGeneratorFactory&;
    auto set_left_boundary(int left_boundary) -> DatasetGeneratorFactory&;
    auto set_right_boundary(int right_boundary) -> DatasetGeneratorFactory&;
    auto set_verbose(bool verbose) -> DatasetGeneratorFactory&;

    [[nodiscard]] auto Build(const std::string& type) const -> std::unique_ptr<IDatasetGenerator>;

   private:
    std::string dataset_name_;
    fs::path parent_directory_;
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    int left_boundary_{0};
    int right_boundary_{0};
    bool verbose_{false};
};

// ---------- Implementation ----------
template <typename T>
DatasetGenerator<T>::DatasetGenerator(std::string dataset_name, fs::path parent_directory, unsigned int num_points,
                                      unsigned int num_dimensions, int left_boundary, int right_boundary, bool verbose)
    : dataset_name_(std::move(dataset_name)),
      parent_directory_(std::move(parent_directory)),
      num_points_(num_points),
      num_dimensions_(num_dimensions),
      left_boundary_(left_boundary),
      right_boundary_(right_boundary),
      verbose_(verbose) {}

template <typename T>
void DatasetGenerator<T>::Generate() const {
    // TODO: add the implementation
}

template <typename T>
auto DatasetGenerator<T>::PrintConfiguration() const -> void {
    std::cout << std::format("---------- Dataset Generator Configuration ----------\n");
    std::cout << std::format("Dataset Name: {}\n", dataset_name_);
    std::cout << std::format("Parent Directory: {}\n", parent_directory_.string());
    std::cout << std::format("Number of Points: {}\n", num_points_);
    std::cout << std::format("Number of Dimensions: {}\n", num_dimensions_);
    std::cout << std::format("Left Boundary: {}\n", left_boundary_);
    std::cout << std::format("Right Boundary: {}\n", right_boundary_);
    std::cout << std::format("-----------------------------------------------------\n");
}

}  // namespace qalsh_chamfer

#endif