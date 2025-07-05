#ifndef CHAMFER_APPROX_H_
#define CHAMFER_APPROX_H_

#include <filesystem>
#include <memory>
#include <string>

namespace qalsh_chamfer {

namespace fs = std::filesystem;

class CrudeNn;

class CrudeNnBuilder {
   public:
    CrudeNnBuilder();

    auto set_dataset_name(const std::string& dataset_name) -> CrudeNnBuilder&;
    auto set_parent_directory(const fs::path& parent_directory) -> CrudeNnBuilder&;
    auto set_num_points(unsigned int num_points) -> CrudeNnBuilder&;
    auto set_num_dimensions(unsigned int num_dimensions) -> CrudeNnBuilder&;
    auto set_verbose(bool debug) -> CrudeNnBuilder&;

    [[nodiscard]] auto Build() const -> std::unique_ptr<CrudeNn>;

   private:
    std::string dataset_name_;
    fs::path parent_directory_;
    unsigned int num_points_;
    unsigned int num_dimensions_;
    bool verbose_;
};

class CrudeNn {
   public:
    auto PrintConfiguration() const -> void;
    auto Execute() const -> void;

    friend class CrudeNnBuilder;

   private:
    CrudeNn(std::string dataset_name, fs::path parent_directory, unsigned int num_points, unsigned int num_dimensions,
            bool verbose);

    std::string dataset_name_;
    fs::path parent_directory_;
    unsigned int num_points_;
    unsigned int num_dimensions_;
    bool verbose_;
};

}  // namespace qalsh_chamfer

#endif