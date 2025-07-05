#ifndef INDEXER_H_
#define INDEXER_H_

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace qalsh_chamfer {

namespace fs = std::filesystem;

class Indexer;

class IndexerBuilder {
   public:
    IndexerBuilder();

    auto set_dataset_name(const std::string& dataset_name) -> IndexerBuilder&;
    auto set_parent_directory(const fs::path& parent_directory) -> IndexerBuilder&;
    auto set_num_points(unsigned int num_points) -> IndexerBuilder&;
    auto set_num_dimensions(unsigned int num_dimensions) -> IndexerBuilder&;
    auto set_approximation_ratio(double approximation_ratio) -> IndexerBuilder&;
    auto set_bucket_width(double bucket_width) -> IndexerBuilder&;
    auto set_beta(double beta) -> IndexerBuilder&;
    auto set_error_probability(double error_probability) -> IndexerBuilder&;
    auto set_collision_schema_param(unsigned int num_hash_tables, unsigned int collision_threshold) -> IndexerBuilder&;
    auto set_page_size(unsigned int page_size) -> IndexerBuilder&;
    auto set_verbose(bool verbose) -> IndexerBuilder&;

    [[nodiscard]] auto Build() const -> std::unique_ptr<Indexer>;

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

class Indexer {
   public:
    auto PrintConfiguration() const -> void;
    auto Execute() const -> void;

    friend class IndexerBuilder;

   private:
    Indexer(std::string dataset_name, fs::path parent_directory, unsigned int num_points, unsigned int num_dimensions,
            double approximation_ratio, double bucket_width, double beta, double error_probability,
            unsigned int num_hash_tables, unsigned int collision_threshold, unsigned int page_size, bool verbose);

    auto WriteParamInBinary(const fs::path& file_path) const -> void;

    auto BuildIndexForSet(std::vector<double>& dot_vector, std::vector<std::vector<double>>& set,
                          const fs::path& index_file_path) const -> void;

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