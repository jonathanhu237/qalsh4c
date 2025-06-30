#include "indexer.h"

#include <format>
#include <iostream>
#include <memory>

#include "const.h"

namespace qalsh_chamfer {

// ------ IndexerBuilder Implementation ------

IndexerBuilder::IndexerBuilder() = default;

auto IndexerBuilder::set_dataset_name(const std::string& dataset_name) -> IndexerBuilder& {
    dataset_name_ = dataset_name;
    return *this;
}

auto IndexerBuilder::set_parent_directory(const std::string& parent_directory) -> IndexerBuilder& {
    parent_directory_ = parent_directory;
    return *this;
}

auto IndexerBuilder::set_num_points(unsigned int num_points) -> IndexerBuilder& {
    num_points_ = num_points;
    return *this;
}

auto IndexerBuilder::set_num_dimensions(unsigned int num_dimensions) -> IndexerBuilder& {
    num_dimensions_ = num_dimensions;
    return *this;
}

auto IndexerBuilder::set_num_hash_tables(unsigned int num_hash_tables) -> IndexerBuilder& {
    // TODO: add the logic to determine the number of hash tables based on the formula provided in the paper
    num_hash_tables_ = num_hash_tables;
    return *this;
}

auto IndexerBuilder::set_beta(float beta) -> IndexerBuilder& {
    if (beta <= kEpsilon) {
        beta_ = 100.0F / static_cast<float>(num_points_);
    }
    return *this;
}

auto IndexerBuilder::set_error_probability(float error_probability) -> IndexerBuilder& {
    error_probability_ = error_probability;
    return *this;
}

auto IndexerBuilder::set_page_size(unsigned int page_size) -> IndexerBuilder& {
    page_size_ = page_size;
    return *this;
}

auto IndexerBuilder::set_verbose(bool verbose) -> IndexerBuilder& {
    verbose_ = verbose;
    return *this;
}

auto IndexerBuilder::Build() const -> std::unique_ptr<Indexer> {
    return std::unique_ptr<Indexer>(new Indexer(dataset_name_, parent_directory_, num_points_, num_dimensions_,
                                                num_hash_tables_, beta_, error_probability_, page_size_, verbose_));
}

// ------ Indexer Implementation ------

Indexer::Indexer(std::string dataset_name, std::string parent_directory, unsigned int num_points,
                 unsigned int num_dimensions, unsigned int num_hash_tables, float beta, float error_probability,
                 unsigned int page_size, bool verbose)
    : dataset_name_(std::move(dataset_name)),
      parent_directory_(std::move(parent_directory)),
      num_points_(num_points),
      num_dimensions_(num_dimensions),
      num_hash_tables_(num_hash_tables),
      beta_(beta),
      error_probability_(error_probability),
      page_size_(page_size),
      verbose_(verbose) {}

auto Indexer::PrintConfiguration() const -> void {
    std::cout << "------------- Indexer Configuration -------------\n";
    std::cout << std::format("Dataset Name: {}\n", dataset_name_);
    std::cout << std::format("Parent Directory: {}\n", parent_directory_);
    std::cout << std::format("Number of Points: {}\n", num_points_);
    std::cout << std::format("Number of Dimensions: {}\n", num_dimensions_);
    std::cout << std::format("Number of Hash Tables: {}\n", num_hash_tables_);
    std::cout << std::format("Beta: {}\n", beta_);
    std::cout << std::format("Error Probability: {}\n", error_probability_);
    std::cout << std::format("Page Size: {}\n", page_size_);
    std::cout << "-------------------------------------------------\n";
}

auto Indexer::Execute() const -> void {
    // TODO: Implement the execution logic for the indexer.
}

}  // namespace qalsh_chamfer