#ifndef INDEXER_H_
#define INDEXER_H_

#include <memory>
#include <string>

namespace qalsh_chamfer {

class Indexer;

class IndexerBuilder {
   public:
    IndexerBuilder();

    auto set_dataset_name(const std::string& dataset_name) -> IndexerBuilder&;
    auto set_parent_directory(const std::string& parent_directory) -> IndexerBuilder&;
    auto set_num_points(unsigned int num_points) -> IndexerBuilder&;
    auto set_num_dimensions(unsigned int num_dimensions) -> IndexerBuilder&;
    auto set_verbose(bool verbose) -> IndexerBuilder&;

    [[nodiscard]] auto Build() const -> std::unique_ptr<Indexer>;

   private:
    std::string dataset_name_;
    std::string parent_directory_;
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    bool verbose_{false};
};

class Indexer {
   public:
    auto PrintConfiguration() const -> void;
    auto Execute() const -> void;

    friend class IndexerBuilder;

   private:
    Indexer(std::string dataset_name, std::string parent_directory, unsigned int num_points,
            unsigned int num_dimensions, bool verbose);

    std::string dataset_name_;
    std::string parent_directory_;
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    bool verbose_{false};
};

}  // namespace qalsh_chamfer

#endif