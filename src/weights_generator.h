#ifndef WEIGHTS_GENERATOR_H_
#define WEIGHTS_GENERATOR_H_

#include <vector>

#include "types.h"

// --------------------------------------------------
// WeightsGenerator Definition
// --------------------------------------------------
class WeightsGenerator {
   public:
    virtual ~WeightsGenerator() = default;
    virtual std::vector<double> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                         bool use_cache) = 0;
};

// --------------------------------------------------
// UniformWeightsGenerator Definition
// --------------------------------------------------
class UniformWeightsGenerator : public WeightsGenerator {
   public:
    UniformWeightsGenerator() = default;
    std::vector<double> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 bool use_cache) override;
};

// --------------------------------------------------
// InMemoryQalshWeightsGenerator Definition
// --------------------------------------------------
class InMemoryQalshWeightsGenerator : public WeightsGenerator {
   public:
    InMemoryQalshWeightsGenerator(double approximation_ratio, double error_probability);
    std::vector<double> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 bool use_cache) override;

   private:
    double approximation_ratio_;
    double error_probability_;
};

// --------------------------------------------------
// DiskQalshWeightsGenerator Definition
// --------------------------------------------------
class DiskQalshWeightsGenerator : public WeightsGenerator {
   public:
    DiskQalshWeightsGenerator() = default;
    std::vector<double> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 bool use_cache) override;
};

#endif