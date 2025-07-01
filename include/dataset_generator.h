#ifndef DATASET_GENERATOR_H_
#define DATASET_GENERATOR_H_

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <string>
#include <vector>

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
    unsigned int num_points_;
    unsigned int num_dimensions_;
    int left_boundary_;
    int right_boundary_;
    bool verbose_;
};

class DatasetGenerator {
   public:
    auto PrintConfiguration() const -> void;
    auto Execute() const -> void;

    friend class DatasetGeneratorBuilder;

   private:
    DatasetGenerator(std::string dataset_name, std::string parent_directory, unsigned int num_points,
                     unsigned int num_dimensions, int left_boundary, int right_boundary, bool debug);

    [[nodiscard]] auto GenerateSet(std::uniform_real_distribution<double>& dist, std::mt19937& gen,
                                   const std::string& set_name) const -> std::vector<std::vector<double>>;

    auto WriteSetToFile(const std::string& dataset_directory, const std::string& set_name,
                        const std::vector<std::vector<double>>& set) const -> void;

    [[nodiscard]] auto ToEigenMatrix(const std::vector<std::vector<double>>& set) const
        -> Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    [[nodiscard]] auto CalculateChamfer(const Eigen::MatrixXd& from_matrix, const Eigen::MatrixXd& to_matrix,
                                        const std::string& from_name, const std::string& to_name) const -> double;

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