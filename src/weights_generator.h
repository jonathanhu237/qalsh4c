#ifndef WEIGHTS_GENERATOR_H_
#define WEIGHTS_GENERATOR_H_

#include <vector>

#include "ann_searcher.h"
#include "types.h"

// --------------------------------------------------
// WeightsGenerator Definition
// --------------------------------------------------
class WeightsGenerator {
   public:
    virtual ~WeightsGenerator() = default;
    virtual std::vector<double> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                         double norm_order, bool use_cache) = 0;
};

// --------------------------------------------------
// UniformWeightsGenerator Definition
// --------------------------------------------------
class UniformWeightsGenerator : public WeightsGenerator {
   public:
    UniformWeightsGenerator() = default;
    std::vector<double> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 double norm_order, bool use_cache) override;
};

// --------------------------------------------------
// QalshWeightsGenerator Definition
// --------------------------------------------------
class QalshWeightsGenerator : public WeightsGenerator {
   public:
    QalshWeightsGenerator();
    std::vector<double> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 double norm_order, bool use_cache) override;

   private:
    std::unique_ptr<AnnSearcher> ann_searcher_;
};

#endif