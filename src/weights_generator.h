#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "ann_searcher.h"
#include "config.h"
#include "point_set.h"

// --------------------------------------------------
// WeightsGenerator Definition
// --------------------------------------------------
class WeightsGenerator {
   public:
    virtual ~WeightsGenerator() = default;
    [[nodiscard]] virtual bool RequiresIndex() const = 0;
    virtual std::vector<double> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                         double norm_order) = 0;
    [[nodiscard]] virtual std::optional<QalshConfig> GetQalshConfig() const;
};

// --------------------------------------------------
// UniformWeightsGenerator Definition
// --------------------------------------------------
class UniformWeightsGenerator : public WeightsGenerator {
   public:
    UniformWeightsGenerator() = default;
    [[nodiscard]] bool RequiresIndex() const override;
    std::vector<double> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 double norm_order) override;
};

// --------------------------------------------------
// QalshWeightsGenerator Definition
// --------------------------------------------------
class QalshWeightsGenerator : public WeightsGenerator {
   public:
    QalshWeightsGenerator(double approximation_ratio, std::optional<std::filesystem::path> save_weights_directory,
                          std::optional<std::filesystem::path> load_weights_directory);
    [[nodiscard]] bool RequiresIndex() const override;
    std::vector<double> Generate(const PointSetMetadata& from_metadata, const PointSetMetadata& to_metadata,
                                 double norm_order) override;
    [[nodiscard]] std::optional<QalshConfig> GetQalshConfig() const override;

   private:
    double approximation_ratio_{0.0};
    std::optional<std::filesystem::path> save_weights_directory_;
    std::optional<std::filesystem::path> load_weights_directory_;
    std::optional<QalshConfig> qalsh_config_;
};
