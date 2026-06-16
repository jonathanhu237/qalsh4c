#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "point_set.h"

namespace DatasetPrep {

enum class CudaChamferStatus {
    kSuccess,
    kFallbackAllowed,
    kFailed,
};

struct CudaChamferResult {
    CudaChamferStatus status{CudaChamferStatus::kFailed};
    float value{0.0};
    std::string message;
};

struct CudaChamferOptions {
    size_t max_source_batch_points{0};
    size_t memory_budget_bytes_override{0};
    std::function<void(std::string_view)> stage_callback;
    std::function<void()> progress_start_callback;
};

using CudaProgressCallback = std::function<void(size_t)>;

CudaChamferResult ComputeDirectedChamferCuda(const std::vector<Point>& from, const std::vector<Point>& to,
                                             float norm_order, const CudaProgressCallback& progress_callback,
                                             CudaChamferOptions options = {});

}  // namespace DatasetPrep
