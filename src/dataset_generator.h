#ifndef DATASET_GENERATOR_H_
#define DATASET_GENERATOR_H_

#include <filesystem>
#include <random>

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
    void PrintConfiguration() const;
    void GeneratePointSet(const std::filesystem::path &dataset_directory, const std::string &point_set_name,
                          unsigned int num_points);

    DatasetMetadata dataset_metadata_;

    double left_boundary_{0.0};
    double right_boundary_{0.0};

    std::mt19937 gen_;
};

#endif