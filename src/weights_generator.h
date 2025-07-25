#ifndef WEIGHTS_GENERATOR_H_
#define WEIGHTS_GENERATOR_H_

#include <filesystem>
#include <vector>

class WeightsGenerator {
   public:
    virtual ~WeightsGenerator() = default;
    virtual std::vector<double> Generate(const std::filesystem::path& dataset_directory) = 0;
};

class UniformWeightsGenerator : public WeightsGenerator {
   public:
    UniformWeightsGenerator() = default;
    std::vector<double> Generate(const std::filesystem::path& dataset_directory) override;
};

#endif