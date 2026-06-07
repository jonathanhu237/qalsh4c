#include "dataset.h"
#include "dataset_cuda.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

bool Near(double actual, double expected) {
    return std::abs(actual - expected) < 1e-9;
}

int main() {
    const std::vector<Point> a = {Point{0.0, 0.0}, Point{2.0, 0.0}};
    const std::vector<Point> b = {Point{1.0, 0.0}, Point{4.0, 0.0}};

    size_t progress_value = 0;
    DatasetPrep::CudaChamferResult l2_result =
        DatasetPrep::ComputeDirectedChamferCuda(a, b, 2.0, [&](size_t completed) { progress_value = completed; });
    if (l2_result.status == DatasetPrep::CudaChamferStatus::kFallbackAllowed) {
        std::cerr << l2_result.message << '\n';
        return 77;
    }
    if (l2_result.status != DatasetPrep::CudaChamferStatus::kSuccess) {
        std::cerr << l2_result.message << '\n';
        return 1;
    }
    const double l2_cpu =
        DatasetPrep::ComputeDirectedChamfer(a, b, 2.0, "cuda test l2", false, DatasetPrep::GroundTruthDevice::kCpu);
    if (!Near(l2_result.value, l2_cpu) || progress_value != a.size()) {
        return 2;
    }

    DatasetPrep::CudaChamferResult l1_result = DatasetPrep::ComputeDirectedChamferCuda(a, b, 1.0, nullptr);
    if (l1_result.status != DatasetPrep::CudaChamferStatus::kSuccess) {
        std::cerr << l1_result.message << '\n';
        return 3;
    }
    const double l1_cpu =
        DatasetPrep::ComputeDirectedChamfer(a, b, 1.0, "cuda test l1", false, DatasetPrep::GroundTruthDevice::kCpu);
    if (!Near(l1_result.value, l1_cpu)) {
        return 4;
    }

    const std::vector<Point> batched_source = {Point{0.0, 0.0}, Point{2.0, 0.0}, Point{5.0, 0.0}};
    std::vector<size_t> batch_progress_values;
    DatasetPrep::CudaChamferOptions batch_options;
    batch_options.max_source_batch_points = 1;
    DatasetPrep::CudaChamferResult batched_result =
        DatasetPrep::ComputeDirectedChamferCuda(batched_source, b, 2.0,
                                                [&](size_t completed) { batch_progress_values.push_back(completed); },
                                                batch_options);
    if (batched_result.status != DatasetPrep::CudaChamferStatus::kSuccess) {
        std::cerr << batched_result.message << '\n';
        return 5;
    }
    const double batched_cpu = DatasetPrep::ComputeDirectedChamfer(
        batched_source, b, 2.0, "cuda test batched", false, DatasetPrep::GroundTruthDevice::kCpu);
    const bool progress_is_monotonic = std::is_sorted(batch_progress_values.begin(), batch_progress_values.end());
    if (!Near(batched_result.value, batched_cpu) || batch_progress_values.empty() ||
        batch_progress_values.back() != batched_source.size() || !progress_is_monotonic) {
        return 6;
    }

    DatasetPrep::CudaChamferOptions insufficient_memory_options;
    insufficient_memory_options.memory_budget_bytes_override = 1;
    DatasetPrep::CudaChamferResult insufficient_memory_result =
        DatasetPrep::ComputeDirectedChamferCuda(a, b, 2.0, nullptr, insufficient_memory_options);
    if (insufficient_memory_result.status != DatasetPrep::CudaChamferStatus::kFallbackAllowed) {
        std::cerr << insufficient_memory_result.message << '\n';
        return 7;
    }
    if (insufficient_memory_result.message.find("insufficient CUDA memory") == std::string::npos) {
        std::cerr << insufficient_memory_result.message << '\n';
        return 8;
    }

    return 0;
}
