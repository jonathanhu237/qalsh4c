#ifndef DATASET_GENERATOR_H_
#define DATASET_GENERATOR_H_

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <random>
#include <unordered_set>

#include "types.h"

class DatasetGenerator {
   public:
    virtual ~DatasetGenerator() = default;
    virtual void Generate(const std::filesystem::path &dataset_directory) = 0;
};

class DatasetSynthesizer : public DatasetGenerator {
   public:
    DatasetSynthesizer(DatasetMetadata dataset_metadata, double left_boundary, double right_boundary);
    void Generate(const std::filesystem::path &dataset_directory) override;

   private:
    void GeneratePointSet(const std::filesystem::path &dataset_directory, const std::string &point_set_name,
                          unsigned int num_points);

    DatasetMetadata dataset_metadata_;

    double left_boundary_{0.0};
    double right_boundary_{0.0};

    std::mt19937 gen_;
};

class DatasetConverter : public DatasetGenerator {
   public:
    DatasetConverter(std::string dataset_name, std::filesystem::path raw_dataset_directory,
                     unsigned int query_num_points);
    void Generate(const std::filesystem::path &dataset_directory) override;

   private:
    void ConvertTexmexDataset();

    template <typename T>
    bool CheckPoints(const std::vector<std::vector<T>> &points, uint32_t num_points, uint32_t num_dimensions);

    template <typename T>
    std::pair<std::vector<T>, std::vector<T>> SplitDataset(const std::vector<T> &original_dataset,
                                                           uint32_t num_queries);

    void ConvertSift();
    void ConvertGist();
    void ConvertTrevi();
    void ConvertP53();

    std::string dataset_name_;
    std::filesystem::path raw_dataset_directory_;
    std::filesystem::path output_dataset_directory_;
    unsigned int query_num_points_;
    DatasetMetadata dataset_metadata_;
};

template <typename T>
bool DatasetConverter::CheckPoints(const std::vector<std::vector<T>> &points, uint32_t num_points,
                                   uint32_t num_dimensions) {
    if (points.size() != num_points) {
        spdlog::error("The number of points is {}, expected {}", points.size(), num_points);
        return false;
    }

    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i].size() != num_dimensions) {
            spdlog::error("The number of dimensions of point {} is {}, expected {}", i, points[i].size(),
                          num_dimensions);
            return false;
        }
    }

    return true;
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> DatasetConverter::SplitDataset(const std::vector<T> &original_dataset,
                                                                         uint32_t num_queries) {
    if (num_queries > original_dataset.size()) {
        throw std::invalid_argument("Cannot sample more queries than available data points.");
    }

    std::vector<uint32_t> indices(original_dataset.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<uint32_t> sampled_indices;
    sampled_indices.reserve(num_queries);

    std::mt19937 gen(std::random_device{}());

    std::ranges::sample(indices, std::back_inserter(sampled_indices), num_queries, gen);

    std::unordered_set<uint32_t> sampled_indices_set(sampled_indices.begin(), sampled_indices.end());

    std::vector<T> queries;
    queries.reserve(num_queries);

    std::vector<T> dataset;
    dataset.reserve(original_dataset.size() - num_queries);

    for (uint32_t i = 0; i < original_dataset.size(); ++i) {
        if (sampled_indices_set.contains(i)) {
            queries.push_back(original_dataset[i]);
        } else {
            dataset.push_back(original_dataset[i]);
        }
    }
    return std::make_pair(std::move(dataset), std::move(queries));
}

#endif