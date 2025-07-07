#ifndef CHAMFER_APPROX_H_
#define CHAMFER_APPROX_H_

#include <filesystem>
#include <string>
#include <vector>

namespace qalsh_chamfer {

namespace fs = std::filesystem;

class ChamferApprox;

class ChamferApproxBuilder {
   public:
    ChamferApproxBuilder();

    auto set_dataset_name(const std::string& dataset_name) -> ChamferApproxBuilder&;
    auto set_parent_directory(const fs::path& parent_directory) -> ChamferApproxBuilder&;
    auto set_num_points(unsigned int num_points) -> ChamferApproxBuilder&;
    auto set_num_dimensions(unsigned int num_dimensions) -> ChamferApproxBuilder&;
    auto set_num_samples(unsigned int num_samples) -> ChamferApproxBuilder&;
    auto set_verbose(bool verbose) -> ChamferApproxBuilder&;

    [[nodiscard]] auto Build() const -> std::unique_ptr<ChamferApprox>;

   private:
    std::string dataset_name_;
    fs::path parent_directory_;
    unsigned int num_points_;
    unsigned int num_dimensions_;
    unsigned int num_samples_;
    bool verbose_;
};

class ChamferApprox {
   public:
    auto PrintConfiguration() const -> void;
    auto Execute() const -> void;

    friend class ChamferApproxBuilder;

   private:
    ChamferApprox(std::string dataset_name, fs::path parent_directory, unsigned int num_points,
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