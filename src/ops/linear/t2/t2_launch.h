#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using T2Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_t2_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_gemv_t1a(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
// D2 occupancy probe (NOT dispatched): T1-A kernel at 128 threads/block.
void launch_t2_gemv_t1a_b128(const Tensor& x, const Weight& w, Tensor& out,
                             cudaStream_t stream);
// D2-A experiment (NOT dispatched): scale-batched variant, same FP order.
void launch_t2_gemv_d2a(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_row_persistent(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_chunked(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
