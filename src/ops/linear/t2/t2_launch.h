#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using T2Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_t2_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_gemv_t1a(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_row_persistent(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_chunked(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
