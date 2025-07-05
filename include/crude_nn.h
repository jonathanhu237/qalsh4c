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

    auto ReadParamFromBinaryFile() -> CrudeNnBuilder&;
    [[nodiscard]] auto Build() const -> std::unique_ptr<CrudeNn>;

   private:
    std::string dataset_name_;
    fs::path parent_directory_;
    unsigned int num_points_;
    unsigned int num_dimensions_;
    double approximation_ratio_;
    double bucket_width_;
    double beta_;
    double error_probability_;
    unsigned int num_hash_tables_;
    unsigned int collision_threshold_;
    unsigned int page_size_;
    bool verbose_;
};

class CrudeNn {
   public:
    auto PrintConfiguration() const -> void;
    auto Execute() const -> void;

    friend class CrudeNnBuilder;

   private:
    CrudeNn(std::string dataset_name, fs::path parent_directory, unsigned int num_points, unsigned int num_dimensions,
            double approximation_ratio, double bucket_width, double beta, double error_probability,
            unsigned int num_hash_tables, unsigned int collision_threshold, unsigned int page_size, bool verbose);

    std::string dataset_name_;
    fs::path parent_directory_;
    unsigned int num_points_;
    unsigned int num_dimensions_;
    double approximation_ratio_;
    double bucket_width_;
    double beta_;
    double error_probability_;
    unsigned int num_hash_tables_;
    unsigned int collision_threshold_;
    unsigned int page_size_;
    bool verbose_;
};

}  // namespace qalsh_chamfer

#endif