#include "qalsh_indexer.h"

#include <spdlog/spdlog.h>

QalshIndexer::QalshIndexer(double approximation_ratio, double bucket_width, double beta, double error_probability,
                           unsigned int num_hash_tables, unsigned int collision_threshold, unsigned int page_size)
    : approximation_ratio_(approximation_ratio),
      bucket_width_(bucket_width),
      beta_(beta),
      error_probability_(error_probability),
      num_hash_tables_(num_hash_tables),
      collision_threshold_(collision_threshold),
      page_size_(page_size) {}

auto QalshIndexer::BuildIndex(const std::filesystem::path& dataset_directory) -> void {
    PrintConfiguration(dataset_directory);
}

auto QalshIndexer::PrintConfiguration(const std::filesystem::path& dataset_directory) const -> void {
    spdlog::debug(R"(The configuration is as follows:
---------- QALSH Indexer Configuration ----------
Dataset Directory: {}
Approximation Ratio: {}
Bucket Width: {}
Beta: {}
Error Probability: {}
Number of Hash Tables: {}
Collision Threshold: {}
Page Size: {}
-----------------------------------------------------)",
                  dataset_directory.string(), approximation_ratio_, bucket_width_, beta_, error_probability_,
                  num_hash_tables_, collision_threshold_, page_size_);
}