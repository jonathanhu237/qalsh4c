#include "dataset_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace DatasetPrep {
namespace {

constexpr double kNormEpsilon = 1e-12;
constexpr double kMemoryBudgetFraction = 0.80;
constexpr size_t kMaxSafetyMarginBytes = 64ULL * 1024ULL * 1024ULL;
constexpr int kCudaBlockSize = 128;
constexpr size_t kMaxKernelBatchPoints = static_cast<size_t>(std::numeric_limits<int>::max()) * kCudaBlockSize;
constexpr auto kProgressPollInterval = std::chrono::milliseconds(250);

__global__ void ChamferMinDistancesKernel(const double* source_points, const double* target_points,
                                          size_t num_source_points, size_t num_target_points,
                                          unsigned int num_dimensions, int norm_kind, double* min_distances,
                                          unsigned long long* progress_counter) {
    const size_t source_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = source_index < num_source_points;

    if (active) {
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

    __syncthreads();
    if (threadIdx.x == 0 && progress_counter != nullptr) {
        const size_t block_begin = static_cast<size_t>(blockIdx.x) * blockDim.x;
        const size_t remaining = block_begin < num_source_points ? num_source_points - block_begin : 0;
        const size_t completed = remaining < static_cast<size_t>(blockDim.x)
                                     ? remaining
                                     : static_cast<size_t>(blockDim.x);
        if (completed > 0) {
            atomicAdd(progress_counter, static_cast<unsigned long long>(completed));
        }
    }
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
    constexpr double kib = 1024.0;
    constexpr double mib = kib * 1024.0;
    constexpr double gib = mib * 1024.0;
    const double value = static_cast<double>(bytes);
    oss.setf(std::ios::fixed, std::ios::floatfield);
    oss.precision(2);
    if (value >= gib) {
        oss << (value / gib) << " GiB";
    } else if (value >= mib) {
        oss << (value / mib) << " MiB";
    } else if (value >= kib) {
        oss << (value / kib) << " KiB";
    } else {
        oss << bytes << " bytes";
    }
    return oss.str();
}

void ReportStage(const CudaChamferOptions& options, std::string_view message) {
    if (options.stage_callback) {
        options.stage_callback(message);
    }
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

class DeviceCounter {
   public:
    DeviceCounter() = default;
    DeviceCounter(const DeviceCounter&) = delete;
    DeviceCounter& operator=(const DeviceCounter&) = delete;

    ~DeviceCounter() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    cudaError_t Allocate() {
        return cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(unsigned long long));
    }

    cudaError_t Reset(cudaStream_t stream) {
        return cudaMemsetAsync(ptr_, 0, sizeof(unsigned long long), stream);
    }

    [[nodiscard]] unsigned long long* get() const {
        return ptr_;
    }

   private:
    unsigned long long* ptr_{nullptr};
};

class HostCounter {
   public:
    HostCounter() = default;
    HostCounter(const HostCounter&) = delete;
    HostCounter& operator=(const HostCounter&) = delete;

    ~HostCounter() {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
        }
    }

    cudaError_t Allocate() {
        return cudaHostAlloc(reinterpret_cast<void**>(&ptr_), sizeof(unsigned long long), cudaHostAllocDefault);
    }

    [[nodiscard]] unsigned long long* get() const {
        return ptr_;
    }

   private:
    unsigned long long* ptr_{nullptr};
};

class CudaStream {
   public:
    CudaStream() = default;
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    cudaError_t Create() {
        return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    }

    [[nodiscard]] cudaStream_t get() const {
        return stream_;
    }

   private:
    cudaStream_t stream_{nullptr};
};

class CudaEvent {
   public:
    CudaEvent() = default;
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    cudaError_t Create() {
        return cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    }

    [[nodiscard]] cudaEvent_t get() const {
        return event_;
    }

   private:
    cudaEvent_t event_{nullptr};
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

CudaChamferResult PollKernelProgress(cudaEvent_t kernel_done, cudaStream_t progress_stream,
                                     const DeviceCounter& progress_device, HostCounter& progress_host,
                                     size_t completed_before_batch, size_t current_batch,
                                     const CudaProgressCallback& progress_callback) {
    size_t last_reported = 0;
    while (true) {
        cudaError_t event_status = cudaEventQuery(kernel_done);
        if (event_status == cudaSuccess) {
            break;
        }
        if (event_status != cudaErrorNotReady) {
            return MakeResult(CudaChamferStatus::kFailed, 0.0,
                              CudaErrorMessage("CUDA Chamfer kernel failed", event_status));
        }

        cudaError_t error = cudaMemcpyAsync(progress_host.get(), progress_device.get(), sizeof(unsigned long long),
                                            cudaMemcpyDeviceToHost, progress_stream);
        if (error != cudaSuccess) {
            return MakeResult(CudaChamferStatus::kFailed, 0.0,
                              CudaErrorMessage("failed to copy CUDA progress counter", error));
        }
        error = cudaStreamSynchronize(progress_stream);
        if (error != cudaSuccess) {
            return MakeResult(CudaChamferStatus::kFailed, 0.0,
                              CudaErrorMessage("failed to synchronize CUDA progress stream", error));
        }

        const size_t batch_progress =
            std::min(static_cast<size_t>(*progress_host.get()), current_batch);
        if (batch_progress > last_reported) {
            last_reported = batch_progress;
            progress_callback(completed_before_batch + batch_progress);
        }
        std::this_thread::sleep_for(kProgressPollInterval);
    }
    return MakeResult(CudaChamferStatus::kSuccess, 0.0, "");
}

}  // namespace

CudaChamferResult ComputeDirectedChamferCuda(const std::vector<Point>& from, const std::vector<Point>& to,
                                             double norm_order, const CudaProgressCallback& progress_callback,
                                             CudaChamferOptions options) {
    ReportStage(options, "querying CUDA devices");
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
    ReportStage(options, "querying CUDA memory");
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
    {
        std::ostringstream oss;
        oss << "source=" << from.size() << " points, target=" << to.size()
            << " points, dimensions=" << num_dimensions << ", initial batch=" << batch_size
            << ", free memory=" << FormatBytes(free_bytes) << ", total memory=" << FormatBytes(total_bytes);
        ReportStage(options, oss.str());
    }

    CudaStream compute_stream;
    error = compute_stream.Create();
    if (error != cudaSuccess) {
        return MakeResult(CudaChamferStatus::kFallbackAllowed, 0.0,
                          CudaErrorMessage("failed to create CUDA compute stream", error));
    }

    bool block_progress_enabled = progress_callback != nullptr;
    CudaStream progress_stream;
    DeviceCounter progress_device;
    HostCounter progress_host;
    CudaEvent kernel_done;
    auto disable_block_progress = [&](const std::string& message) {
        block_progress_enabled = false;
        std::ostringstream oss;
        oss << message << "; using batch-level progress";
        ReportStage(options, oss.str());
    };
    if (block_progress_enabled) {
        error = progress_stream.Create();
        if (error != cudaSuccess) {
            disable_block_progress(CudaErrorMessage("failed to create CUDA progress stream", error));
        }
    }
    if (block_progress_enabled) {
        error = progress_device.Allocate();
        if (error != cudaSuccess) {
            disable_block_progress(CudaErrorMessage("failed to allocate CUDA progress counter", error));
        }
    }
    if (block_progress_enabled) {
        error = progress_host.Allocate();
        if (error != cudaSuccess) {
            disable_block_progress(CudaErrorMessage("failed to allocate CUDA host progress counter", error));
        }
    }
    if (block_progress_enabled) {
        error = kernel_done.Create();
        if (error != cudaSuccess) {
            disable_block_progress(CudaErrorMessage("failed to create CUDA progress event", error));
        }
    }
    if (block_progress_enabled) {
        ReportStage(options, "using CUDA block-level progress updates");
    }

    ReportStage(options, "flattening target points");
    std::vector<double> target_host = FlattenPoints(to, 0, to.size(), num_dimensions);
    DeviceBuffer target_device;
    ReportStage(options, "allocating target buffer on device");
    error = target_device.Allocate(target_bytes);
    if (error != cudaSuccess) {
        return MakeResult(CudaChamferStatus::kFallbackAllowed, 0.0,
                          CudaErrorMessage("failed to allocate CUDA target buffer", error));
    }
    ReportStage(options, "copying target points to device");
    error = cudaMemcpyAsync(target_device.get(), target_host.data(), target_bytes, cudaMemcpyHostToDevice,
                            compute_stream.get());
    if (error != cudaSuccess) {
        return MakeResult(CudaChamferStatus::kFallbackAllowed, 0.0,
                          CudaErrorMessage("failed to copy target points to CUDA device", error));
    }

    DeviceBuffer source_device;
    DeviceBuffer distances_device;
    size_t source_bytes = 0;
    bool allocated_batch_buffers = false;
    ReportStage(options, "allocating source batch buffers on device");
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
    {
        std::ostringstream oss;
        oss << "running CUDA batches with batch=" << batch_size;
        ReportStage(options, oss.str());
    }
    if (options.progress_start_callback) {
        options.progress_start_callback();
    }

    double total = 0.0;
    size_t completed = 0;
    std::vector<double> distances_host(batch_size, 0.0);

    while (completed < from.size()) {
        const size_t current_batch = std::min(batch_size, from.size() - completed);
        const size_t current_source_bytes = current_batch * num_dimensions * sizeof(double);
        std::vector<double> source_host = FlattenPoints(from, completed, current_batch, num_dimensions);

        error = cudaMemcpyAsync(source_device.get(), source_host.data(), current_source_bytes, cudaMemcpyHostToDevice,
                                compute_stream.get());
        if (error != cudaSuccess) {
            return MakeResult(CudaChamferStatus::kFailed, 0.0,
                              CudaErrorMessage("failed to copy source batch to CUDA device", error));
        }
        if (block_progress_enabled) {
            error = progress_device.Reset(compute_stream.get());
            if (error != cudaSuccess) {
                return MakeResult(CudaChamferStatus::kFailed, 0.0,
                                  CudaErrorMessage("failed to reset CUDA progress counter", error));
            }
            error = cudaStreamSynchronize(compute_stream.get());
            if (error != cudaSuccess) {
                return MakeResult(CudaChamferStatus::kFailed, 0.0,
                                  CudaErrorMessage("failed to synchronize CUDA progress counter reset", error));
            }
        }

        const int grid_size = static_cast<int>((current_batch + kCudaBlockSize - 1U) / kCudaBlockSize);
        ChamferMinDistancesKernel<<<grid_size, kCudaBlockSize, 0, compute_stream.get()>>>(
            source_device.get(), target_device.get(), current_batch, to.size(),
            static_cast<unsigned int>(num_dimensions), norm_kind, distances_device.get(),
            block_progress_enabled ? progress_device.get() : nullptr);
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            return MakeResult(CudaChamferStatus::kFailed, 0.0,
                              CudaErrorMessage("failed to launch CUDA Chamfer kernel", error));
        }
        if (block_progress_enabled) {
            error = cudaEventRecord(kernel_done.get(), compute_stream.get());
            if (error != cudaSuccess) {
                return MakeResult(CudaChamferStatus::kFailed, 0.0,
                                  CudaErrorMessage("failed to record CUDA progress event", error));
            }
            CudaChamferResult progress_result =
                PollKernelProgress(kernel_done.get(), progress_stream.get(), progress_device, progress_host,
                                   completed, current_batch, progress_callback);
            if (progress_result.status != CudaChamferStatus::kSuccess) {
                return progress_result;
            }
            error = cudaEventSynchronize(kernel_done.get());
            if (error != cudaSuccess) {
                return MakeResult(CudaChamferStatus::kFailed, 0.0,
                                  CudaErrorMessage("CUDA Chamfer kernel failed", error));
            }
        } else {
            error = cudaStreamSynchronize(compute_stream.get());
            if (error != cudaSuccess) {
                return MakeResult(CudaChamferStatus::kFailed, 0.0,
                                  CudaErrorMessage("CUDA Chamfer kernel failed", error));
            }
        }

        error = cudaMemcpyAsync(distances_host.data(), distances_device.get(), current_batch * sizeof(double),
                                cudaMemcpyDeviceToHost, compute_stream.get());
        if (error != cudaSuccess) {
            return MakeResult(CudaChamferStatus::kFailed, 0.0,
                              CudaErrorMessage("failed to copy CUDA Chamfer distances to host", error));
        }
        error = cudaStreamSynchronize(compute_stream.get());
        if (error != cudaSuccess) {
            return MakeResult(CudaChamferStatus::kFailed, 0.0,
                              CudaErrorMessage("failed to synchronize CUDA distance copy", error));
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
