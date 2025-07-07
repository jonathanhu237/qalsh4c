#ifndef CHAMFER_APPROX_H_
#define CHAMFER_APPROX_H_

#include <filesystem>
#include <string>

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
    auto set_verbose(bool verbose) -> ChamferApproxBuilder&;

    [[nodiscard]] auto Build() const -> std::unique_ptr<ChamferApprox>;

   private:
    std::string dataset_name_;
    fs::path parent_directory_;
    unsigned int num_points_;
    unsigned int num_dimensions_;
    bool verbose_;
};

class ChamferApprox {
   public:
    auto PrintConfiguration() const -> void;
    auto Execute() const -> void;

    friend class ChamferApproxBuilder;

   private:
    ChamferApprox(std::string dataset_name, fs::path parent_directory, unsigned int num_points,
                  unsigned int num_dimensions, bool verbose);

    std::string dataset_name_;
    fs::path parent_directory_;
    unsigned int num_points_;
    unsigned int num_dimensions_;
    bool verbose_;
};

};  // namespace qalsh_chamfer

#endif