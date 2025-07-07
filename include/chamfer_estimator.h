#ifndef CHAMFER_ESTIMATOR_H_
#define CHAMFER_ESTIMATOR_H_

#include <filesystem>
#include <string>
#include <vector>

namespace qalsh_chamfer {

namespace fs = std::filesystem;

class ChamferEstimator;

class ChamferEstimatorBuilder {
   public:
    ChamferEstimatorBuilder();

    auto set_dataset_name(const std::string& dataset_name) -> ChamferEstimatorBuilder&;
    auto set_parent_directory(const fs::path& parent_directory) -> ChamferEstimatorBuilder&;
    auto set_num_points(unsigned int num_points) -> ChamferEstimatorBuilder&;
    auto set_num_dimensions(unsigned int num_dimensions) -> ChamferEstimatorBuilder&;
    auto set_num_samples(unsigned int num_samples) -> ChamferEstimatorBuilder&;
    auto set_verbose(bool verbose) -> ChamferEstimatorBuilder&;

    [[nodiscard]] auto Build() const -> std::unique_ptr<ChamferEstimator>;

   private:
    std::string dataset_name_;
    fs::path parent_directory_;
    unsigned int num_points_;
    unsigned int num_dimensions_;
    unsigned int num_samples_;
    bool verbose_;
};

class ChamferEstimator {
   public:
    auto PrintConfiguration() const -> void;
    auto Execute() const -> void;

    friend class ChamferEstimatorBuilder;

   private:
    ChamferEstimator(std::string dataset_name, fs::path parent_directory, unsigned int num_points,
                     unsigned int num_dimensions, unsigned int num_samples, bool verbose);

    [[nodiscard]] auto ApproximateChamferDistance(const std::vector<std::vector<double>>& from_set,
                                                  const std::vector<std::vector<double>>& to_set,
                                                  const std::vector<double>& from_set_D_array,
                                                  const std::string& from_set_name = "",
                                                  const std::string& to_set_name = "") const -> double;

    std::string dataset_name_;
    fs::path parent_directory_;
    unsigned int num_points_;
    unsigned int num_dimensions_;
    unsigned int num_samples_;
    bool verbose_;
};

};  // namespace qalsh_chamfer

#endif