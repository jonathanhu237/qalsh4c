#ifndef DATASET_GENERATOR_H_
#define DATASET_GENERATOR_H_

#include <filesystem>
#include <random>

#include "types.h"

class DatasetGenerator {
   public:
    virtual ~DatasetGenerator() = default;
    virtual auto Generate(const std::filesystem::path &dataset_directory) -> void = 0;
};

class DatasetSynthesizer : public DatasetGenerator {
   public:
    DatasetSynthesizer(DatasetMetadata dataset_metadata, double left_boundary, double right_boundary, bool in_memory);
    auto Generate(const std::filesystem::path &dataset_directory) -> void override;

   private:
    auto GeneratePointSet(const std::filesystem::path &dataset_directory, const std::string &point_set_name,
                          unsigned int num_points) -> void;

    DatasetMetadata dataset_metadata_;

    double left_boundary_{0.0};
    double right_boundary_{0.0};
    bool in_memory_{false};

    std::mt19937 gen_;
};

#endif