#ifndef CHAMFER_APPROX_H_
#define CHAMFER_APPROX_H_

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "b_plus_tree.h"

namespace qalsh_chamfer {

namespace fs = std::filesystem;

class CrudeNn;

class CrudeNnSearchHelper;

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

    std::vector<std::vector<double>> dot_vectors_;
};

class CrudeNn {
   public:
    using Candidate = std::pair<double, unsigned int>;  // Distance, Point ID

    auto PrintConfiguration() const -> void;
    auto Execute() const -> void;

    friend class CrudeNnBuilder;

   private:
    CrudeNn(std::string dataset_name, fs::path parent_directory, unsigned int num_points, unsigned int num_dimensions,
            double approximation_ratio, double bucket_width, double beta, double error_probability,
            unsigned int num_hash_tables, unsigned int collision_threshold, unsigned int page_size,
            std::vector<std::vector<double>> dot_vectors, bool verbose);

    [[nodiscard]] auto GenerateDArrayForSet(const std::vector<std::vector<double>>& set_from,
                                            const std::vector<std::vector<double>>& set_to,
                                            const std::string& set_to_name) const -> std::vector<double>;
    [[nodiscard]] auto CAnnSearch(const std::vector<double>& query, const std::vector<std::vector<double>>& dataset,
                                  const std::string& set_name) const -> Candidate;

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
    std::vector<std::vector<double>> dot_vectors_;
    bool verbose_;
};

class CrudeNnSearchHelper {
   public:
    friend class CrudeNn;

   private:
    CrudeNnSearchHelper(const fs::path& index_file_path, unsigned int page_size, double key);
    auto IncrementalSearch(double bound) -> std::vector<unsigned int>;

    std::vector<BPlusTree::KeyValuePair> left_buffer_;
    std::vector<BPlusTree::KeyValuePair> right_buffer_;
    size_t left_buffer_index_;
    size_t right_buffer_index_;
    unsigned int left_page_num_;
    unsigned int right_page_num_;
    BPlusTree b_plus_tree_;
    double key_;
};

}  // namespace qalsh_chamfer

#endif