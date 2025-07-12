#include "qalsh_indexer.h"

#include <spdlog/spdlog.h>

#include <toml++/toml.hpp>

#include "constants.h"

QalshIndexer::QalshIndexer(std::filesystem::path dataset_directory, double approximation_ratio, double bucket_width,
                           double beta, double error_probability, unsigned int num_hash_tables,
                           unsigned int collision_threshold, unsigned int page_size)
    : dataset_directory_(std::move(dataset_directory)),
      approximation_ratio_(approximation_ratio),
      bucket_width_(bucket_width),
      beta_(beta),
      error_probability_(error_probability),
      num_hash_tables_(num_hash_tables),
      collision_threshold_(collision_threshold),
      page_size_(page_size) {
    // Read dataset metadata
    toml::table tbl = toml::parse_file((dataset_directory_ / "metadata.toml").string());

    base_num_points_ = tbl["base_num_points"].value_or(0U);
    if (base_num_points_ == 0) {
        throw std::runtime_error("base_num_points is not specified in metadata.toml");
    }

    query_num_points_ = tbl["query_num_points"].value_or(0U);
    if (query_num_points_ == 0) {
        throw std::runtime_error("query_num_points is not specified in metadata.toml");
    }

    num_dimensions_ = tbl["num_dimensions"].value_or(0U);
    if (num_dimensions_ == 0) {
        throw std::runtime_error("num_dimensions is not specified in metadata.toml");
    }

    data_type = tbl["data_type"].value_or("");
    if (data_type.empty()) {
        throw std::runtime_error("data_type is not specified in metadata.toml");
    }

    // Calculate QALSH specific parameters
    if (bucket_width <= Constants::kEpsilon) {
        bucket_width_ = 2.0 * std::sqrt(approximation_ratio_);
    }
    if (beta <= Constants::kEpsilon) {
        beta_ = 100.0 / static_cast<double>(base_num_points_);
    }

    double term1 = std::sqrt(std::log(2.0 / beta_));
    double term2 = std::sqrt(std::log(1.0 / error_probability_));
    double p1 = 2.0 / std::numbers::pi_v<double> * atan(bucket_width_ / 2.0);
    double p2 = 2.0 / std::numbers::pi_v<double> * atan(bucket_width_ / (2.0 * approximation_ratio_));

    if (num_hash_tables == 0) {
        double numerator = std::pow(term1 + term2, 2.0);
        double denominator = 2.0 * std::pow(p1 - p2, 2.0);
        num_hash_tables_ = static_cast<unsigned int>(std::ceil(numerator / denominator));
    }

    if (collision_threshold == 0) {
        double eta = term1 / term2;
        double alpha = (eta * p1 + p2) / (1 + eta);
        collision_threshold_ = static_cast<unsigned int>(std::ceil(alpha * num_hash_tables_));
    }
}

auto QalshIndexer::BuildIndex() -> void { PrintConfiguration(); }

auto QalshIndexer::PrintConfiguration() const -> void {
    spdlog::debug(R"(The configuration is as follows:
---------- QALSH Indexer Configuration ----------
Dataset Directory: {}
Number of points in base set: {}
Number of points in query set: {}
Number of Dimensions: {}
Data Type: {}
Approximation Ratio: {}
Bucket Width: {}
Beta: {}
Error Probability: {}
Number of Hash Tables: {}
Collision Threshold: {}
Page Size: {}
-----------------------------------------------------)",
                  dataset_directory_.string(), base_num_points_, query_num_points_, num_dimensions_, data_type,
                  approximation_ratio_, bucket_width_, beta_, error_probability_, num_hash_tables_,
                  collision_threshold_, page_size_);
}