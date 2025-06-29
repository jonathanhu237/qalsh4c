#ifndef DATASET_GENERATOR_H_
#define DATASET_GENERATOR_H_

#include <memory>
#include <string>

namespace qalsh_chamfer {

class DatasetGenerator;

class DatasetGeneratorBuilder {
   public:
    DatasetGeneratorBuilder();

    auto set_dataset_name(const std::string& dataset_name) -> DatasetGeneratorBuilder&;
    auto set_parent_directory(const std::string& parent_directory) -> DatasetGeneratorBuilder&;
    auto set_num_points_(unsigned int num_points) -> DatasetGeneratorBuilder&;
    auto set_num_dimensions(unsigned int num_dimensions) -> DatasetGeneratorBuilder&;
    auto set_left_boundary(int left_boundary) -> DatasetGeneratorBuilder&;
    auto set_right_boundary(int right_boundary) -> DatasetGeneratorBuilder&;
    auto set_verbose(bool debug) -> DatasetGeneratorBuilder&;

    [[nodiscard]] auto Build() const -> std::unique_ptr<DatasetGenerator>;

   private:
    std::string dataset_name_;
    std::string parent_directory_;
    unsigned int num_points_{};
    unsigned int num_dimensions_{};
    int left_boundary_{};
    int right_boundary_{};
    bool verbose_{false};
};

class DatasetGenerator {
   public:
    void PrintConfiguration() const;
    void Execute() const;

    friend class DatasetGeneratorBuilder;

   private:
    DatasetGenerator(std::string dataset_name, std::string parent_directory, unsigned int num_points,
                     unsigned int num_dimensions, int left_boundary, int right_boundary, bool debug);

    std::string dataset_name_;
    std::string parent_directory_;
    unsigned int num_points_;
    unsigned int num_dimensions_;
    int left_boundary_;
    int right_boundary_;
    bool verbose_;
};

}  // namespace qalsh_chamfer

#endif