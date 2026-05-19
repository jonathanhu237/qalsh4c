#include "dataset_cuda.h"

namespace DatasetPrep {

CudaChamferResult ComputeDirectedChamferCuda(const std::vector<Point>&, const std::vector<Point>&, double,
                                             const CudaProgressCallback&, CudaChamferOptions) {
    return CudaChamferResult{.status = CudaChamferStatus::kFallbackAllowed,
                             .value = 0.0,
                             .message = "CUDA support is not compiled in"};
}

}  // namespace DatasetPrep
