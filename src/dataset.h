#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string_view>
#include <string>
#include <vector>

#include "config.h"
#include "point_set.h"

namespace DatasetPrep {

enum class DatasetKind {
    kMnist,
    kP53,
    kSift,
    kGist,
    kTrevi,
};

enum class GroundTruthDevice {
    kAuto,
    kCpu,
    kCuda,
};

struct GroundTruthTask {
    double norm_order{0.0};
    ChamferDirection direction{ChamferDirection::kAb};
};

struct CommonOutputOptions {
    std::filesystem::path output_directory;
    std::string ground_truth_specs;
    bool overwrite{false};
    bool show_progress{true};
    GroundTruthDevice ground_truth_device{GroundTruthDevice::kAuto};
    size_t cuda_batch_size{0};
};

struct GmmOptions {
    CommonOutputOptions output;
    unsigned int num_points_a{1000};
    unsigned int num_points_b{1000};
    unsigned int num_dimensions{256};
    double outlier_fraction{0.1};
    double outlier_scale{4096.0};
    std::optional<unsigned int> seed;
};

struct ConvertOptions {
    CommonOutputOptions output;
    DatasetKind kind{DatasetKind::kMnist};
    std::filesystem::path raw_directory;
    std::filesystem::path images_file;
    std::optional<unsigned int> num_points_a;
    std::optional<unsigned int> seed;
    bool shuffle{true};
};

struct GroundTruthOptions {
    std::filesystem::path dataset_directory;
    std::string ground_truth_specs;
    bool overwrite{false};
    bool show_progress{true};
    GroundTruthDevice ground_truth_device{GroundTruthDevice::kAuto};
    size_t cuda_batch_size{0};
};

GroundTruthDevice ParseGroundTruthDevice(std::string_view value);
std::vector<GroundTruthTask> ParseGroundTruthTasks(const std::string& specs);

void WritePointSet(const std::filesystem::path& file_path, const std::vector<Point>& points);
double ComputeDirectedChamfer(const std::vector<Point>& from, const std::vector<Point>& to, double norm_order,
                              const std::string& label, bool show_progress,
                              GroundTruthDevice ground_truth_device = GroundTruthDevice::kCpu,
                              size_t cuda_batch_size = 0);

void GenerateGmmDataset(const GmmOptions& options);
void ConvertDataset(const ConvertOptions& options);
void ComputeDatasetGroundTruth(const GroundTruthOptions& options);

}  // namespace DatasetPrep
