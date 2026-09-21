#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

using T2Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_t2_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_gemv_t1a(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
// M7-P1 prototype (NOT production-dispatched): tiled PQ2 -> shared BF16 MMA.
void launch_t2_prefill_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
// B2 tile-sensitivity probes (NOT dispatched).
void launch_t2_prefill_mma_128x32(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream);
void launch_t2_prefill_mma_64x32(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_t2_prefill_mma_64x64(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);

[[nodiscard]] std::size_t t2_prefill_mma_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                                  std::int32_t t);
void launch_t2_row_persistent(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_chunked(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
