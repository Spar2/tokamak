#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void launch_nvfp4_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_nvfp4_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_nvfp4_w4a16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_nvfp4_w4a16_mma_add(const Tensor& x, const Weight& weight, Tensor& residual,
                                cudaStream_t stream);
void launch_nvfp4_w4a16_mma_swiglu(const Tensor& x, const Weight& weight, Tensor& out,
                                   cudaStream_t stream);

} // namespace ninfer::ops::detail
