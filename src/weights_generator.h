#ifndef WEIGHTS_GENERATOR_H_
#define WEIGHTS_GENERATOR_H_

#include <vector>

#include "point_set.h"

class WeightsGenerator {
   public:
    virtual ~WeightsGenerator() = default;
    virtual std::vector<double> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                         bool in_memory, bool use_cache) = 0;
};

class UniformWeightsGenerator : public WeightsGenerator {
   public:
    UniformWeightsGenerator() = default;
    std::vector<double> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 bool in_memory, bool use_cache) override;
};

class QalshWeightsGenerator : public WeightsGenerator {
   public:
    QalshWeightsGenerator(double approximation_ratio);
    std::vector<double> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 bool in_memory, bool use_cache) override;

   private:
    double approximation_ratio_;
};

#endif