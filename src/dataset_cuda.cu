#include "dataset_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace DatasetPrep {
namespace {

constexpr double kNormEpsilon = 1e-12;
constexpr double kMemoryBudgetFraction = 0.80;
constexpr size_t kMaxSafetyMarginBytes = 64ULL * 1024ULL * 1024ULL;
constexpr int kCudaBlockSize = 128;
constexpr size_t kMaxKernelBatchPoints = static_cast<size_t>(std::numeric_limits<int>::max()) * kCudaBlockSize;

__global__ void ChamferMinDistancesKernel(const double* source_points, const double* target_points,
                                          size_t num_source_points, size_t num_target_points,
                                          unsigned int num_dimensions, int norm_kind, double* min_distances) {
    const size_t source_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (source_index >= num_source_points) {
        return;
    }

    const double* source = source_points + (source_index * num_dimensions);
    double min_distance = DBL_MAX;

    for (size_t target_index = 0; target_index < num_target_points; ++target_index) {
        const double* target = target_points + (target_index * num_dimensions);
        double distance = 0.0;
        for (unsigned int dimension = 0; dimension < num_dimensions; ++dimension) {
            const double diff = source[dimension] - target[dimension];
            if (norm_kind == 1) {
                distance += fabs(diff);
            } else {
                distance += diff * diff;
            }
        }
        if (norm_kind == 2) {
            distance = sqrt(distance);
        }
        min_distance = fmin(min_distance, distance);
    }

    min_distances[source_index] = min_distance;
}

CudaChamferResult MakeResult(CudaChamferStatus status, double value, std::string message) {
    CudaChamferResult result;
    result.status = status;
    result.value = value;
    result.message = std::move(message);
    return result;
}

std::string CudaErrorMessage(const std::string& prefix, cudaError_t error) {
    std::ostringstream oss;
    oss << prefix << ": " << cudaGetErrorString(error);
    return oss.str();
}

std::string FormatBytes(size_t bytes) {
    std::ostringstream oss;
    oss << bytes << " bytes";
    return oss.str();
}

bool CheckedMultiply(size_t left, size_t right, size_t& output) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        return false;
    }
    output = left * right;
    return true;
}

bool PointBytes(size_t num_points, size_t num_dimensions, size_t& output) {
    size_t values = 0;
    if (!CheckedMultiply(num_points, num_dimensions, values)) {
        return false;
    }
    return CheckedMultiply(values, sizeof(double), output);
}

int NormKind(double norm_order) {
    if (std::abs(norm_order - 1.0) < kNormEpsilon) {
        return 1;
    }
    if (std::abs(norm_order - 2.0) < kNormEpsilon) {
        return 2;
    }
    return 0;
}

std::vector<double> FlattenPoints(const std::vector<Point>& points, size_t begin, size_t count,
                                  size_t num_dimensions) {
    std::vector<double> flattened;
    flattened.reserve(count * num_dimensions);
    for (size_t point_index = begin; point_index < begin + count; ++point_index) {
        flattened.insert(flattened.end(), points[point_index].begin(), points[point_index].end());
    }
    return flattened;
}

class DeviceBuffer {
   public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    cudaError_t Allocate(size_t bytes) {
        if (ptr_ != nullptr) {
            cudaError_t free_error = cudaFree(ptr_);
            ptr_ = nullptr;
            if (free_error != cudaSuccess) {
                return free_error;
            }
        }
        return cudaMalloc(reinterpret_cast<void**>(&ptr_), bytes);
    }

    [[nodiscard]] double* get() const { return ptr_; }

   private:
    double* ptr_{nullptr};
};

size_t SafetyMargin(size_t budget_bytes) {
    return std::min(kMaxSafetyMarginBytes, budget_bytes / 10U);
}

CudaChamferResult ComputeBatchCapacity(size_t free_bytes, size_t num_source_points, size_t num_target_points,
                                       size_t num_dimensions, CudaChamferOptions options, size_t& batch_size,
                                       size_t& target_bytes) {
    size_t one_source_bytes = 0;
    if (!PointBytes(1, num_dimensions, one_source_bytes)) {
        return MakeResult(CudaChamferStatus::kFallbackAllowed, 0.0, "point dimensions overflow CUDA memory sizing");
    }
    if (!PointBytes(num_target_points, num_dimensions, target_bytes)) {
        return MakeResult(CudaChamferStatus::kFallbackAllowed, 0.0, "target point set overflows CUDA memory sizing");
    }
    if (one_source_bytes > std::numeric_limits<size_t>::max() - sizeof(double)) {
        return MakeResult(CudaChamferStatus::kFallbackAllowed, 0.0, "source batch memory sizing overflow");
    }

    const size_t budget_bytes =
        options.memory_budget_bytes_override == 0
            ? static_cast<size_t>(static_cast<double>(free_bytes) * kMemoryBudgetFraction)
            : options.memory_budget_bytes_override;
    const size_t margin_bytes = SafetyMargin(budget_bytes);
    const size_t per_source_bytes = one_source_bytes + sizeof(double);

    if (budget_bytes <= target_bytes || budget_bytes - target_bytes <= margin_bytes ||
        budget_bytes - target_bytes - margin_bytes < per_source_bytes) {
        std::ostringstream oss;
        oss << "insufficient CUDA memory for target-resident Chamfer computation; need at least target "
            << FormatBytes(target_bytes) << " plus one source point/output " << FormatBytes(per_source_bytes)
            << " within budget " << FormatBytes(budget_bytes);
        return MakeResult(CudaChamferStatus::kFallbackAllowed, 0.0, oss.str());
    }

    const size_t batch_budget = budget_bytes - target_bytes - margin_bytes;
    const size_t max_batch_points =
        options.max_source_batch_points == 0 ? kMaxKernelBatchPoints
                                             : std::min(options.max_source_batch_points, kMaxKernelBatchPoints);
    batch_size = std::max<size_t>(1, std::min({num_source_points, batch_budget / per_source_bytes,
                                               max_batch_points}));
    return MakeResult(CudaChamferStatus::kSuccess, 0.0, "");
}

}  // namespace

CudaChamferResult ComputeDirectedChamferCuda(const std::vector<Point>& from, const std::vector<Point>& to,
                                             double norm_order, const CudaProgressCallback& progress_callback,
                                             CudaChamferOptions options) {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        return MakeResult(CudaChamferStatus::kFallbackAllowed, 0.0,
                          error == cudaSuccess ? "no CUDA devices are available"
                                               : CudaErrorMessage("failed to query CUDA devices", error));
    }

    const int norm_kind = NormKind(norm_order);
    if (norm_kind == 0) {
        return MakeResult(CudaChamferStatus::kFailed, 0.0, "unsupported norm order");
    }

    const size_t num_dimensions = from.front().size();
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) {
        return MakeResult(CudaChamferStatus::kFallbackAllowed, 0.0,
                          CudaErrorMessage("failed to query CUDA memory", error));
    }

    size_t batch_size = 0;
    size_t target_bytes = 0;
    CudaChamferResult capacity =
        ComputeBatchCapacity(free_bytes, from.size(), to.size(), num_dimensions, options, batch_size, target_bytes);
    if (capacity.status != CudaChamferStatus::kSuccess) {
        return capacity;
    }

    std::vector<double> target_host = FlattenPoints(to, 0, to.size(), num_dimensions);
    DeviceBuffer target_device;
    error = target_device.Allocate(target_bytes);
    if (error != cudaSuccess) {
        return MakeResult(CudaChamferStatus::kFallbackAllowed, 0.0,
                          CudaErrorMessage("failed to allocate CUDA target buffer", error));
    }
    error = cudaMemcpy(target_device.get(), target_host.data(), target_bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        return MakeResult(CudaChamferStatus::kFallbackAllowed, 0.0,
                          CudaErrorMessage("failed to copy target points to CUDA device", error));
    }

    DeviceBuffer source_device;
    DeviceBuffer distances_device;
    size_t source_bytes = 0;
    bool allocated_batch_buffers = false;
    while (batch_size > 0) {
        if (!PointBytes(batch_size, num_dimensions, source_bytes)) {
            return MakeResult(CudaChamferStatus::kFallbackAllowed, 0.0, "source batch memory sizing overflow");
        }
        error = source_device.Allocate(source_bytes);
        if (error == cudaSuccess) {
            error = distances_device.Allocate(batch_size * sizeof(double));
            if (error == cudaSuccess) {
                allocated_batch_buffers = true;
                break;
            }
        }
        batch_size /= 2U;
    }
    if (!allocated_batch_buffers) {
        return MakeResult(CudaChamferStatus::kFallbackAllowed, 0.0,
                          "failed to allocate CUDA source batch buffers");
    }

    double total = 0.0;
    size_t completed = 0;
    std::vector<double> distances_host(batch_size, 0.0);

    while (completed < from.size()) {
        const size_t current_batch = std::min(batch_size, from.size() - completed);
        const size_t current_source_bytes = current_batch * num_dimensions * sizeof(double);
        std::vector<double> source_host = FlattenPoints(from, completed, current_batch, num_dimensions);

        error = cudaMemcpy(source_device.get(), source_host.data(), current_source_bytes, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            return MakeResult(CudaChamferStatus::kFailed, 0.0,
                              CudaErrorMessage("failed to copy source batch to CUDA device", error));
        }

        const int grid_size = static_cast<int>((current_batch + kCudaBlockSize - 1U) / kCudaBlockSize);
        ChamferMinDistancesKernel<<<grid_size, kCudaBlockSize>>>(source_device.get(), target_device.get(),
                                                                 current_batch, to.size(),
                                                                 static_cast<unsigned int>(num_dimensions), norm_kind,
                                                                 distances_device.get());
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            return MakeResult(CudaChamferStatus::kFailed, 0.0,
                              CudaErrorMessage("failed to launch CUDA Chamfer kernel", error));
        }
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            return MakeResult(CudaChamferStatus::kFailed, 0.0,
                              CudaErrorMessage("CUDA Chamfer kernel failed", error));
        }

        error = cudaMemcpy(distances_host.data(), distances_device.get(), current_batch * sizeof(double),
                           cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            return MakeResult(CudaChamferStatus::kFailed, 0.0,
                              CudaErrorMessage("failed to copy CUDA Chamfer distances to host", error));
        }

        for (size_t i = 0; i < current_batch; ++i) {
            total += distances_host[i];
        }
        completed += current_batch;
        if (progress_callback) {
            progress_callback(completed);
        }
    }

    return MakeResult(CudaChamferStatus::kSuccess, total, "");
}

}  // namespace DatasetPrep
