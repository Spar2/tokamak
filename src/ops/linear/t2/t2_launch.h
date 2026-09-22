#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

using T2Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_t2_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_gemv_t1a(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
// M7-P1 tiled PQ2 -> shared BF16 MMA paths. Scratch-dispatched behind the
// NINFER_T2_PREFILL_P1 env gate for exact tiles (T=8 async prototype,
// T=32/64/128 P1-SYNC); direct calls remain for micro/oracle tests.
void launch_t2_prefill_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
// B2 tile-sensitivity probes (NOT dispatched).
void launch_t2_prefill_mma_128x32(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream);
void launch_t2_prefill_mma_64x32(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_t2_prefill_mma_64x64(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream);
// P1-SYNC: deterministic synchronous 64x32 fused K-loop (T=32/64/128).
// NOT dispatched; called by determinism/numerics probes and (after the
// determinism gate) by scratch large-T dispatch.
void launch_t2_prefill_mma_sync(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
// P1-SYNC-8: synchronous 64x8 path for T=8 (replaces async in dispatch).
void launch_t2_prefill_mma_sync8(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream);
// P1-SYNC masked-tail path (BM=64/BN=32, arbitrary T>=9): full tiles plus
// one zero-masked final token tile. NOT dispatched until the tail matrix
// passes; called by tail probes.
void launch_t2_prefill_mma_sync_mt(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
// Serialized per-slab diagnostic (NOT production): single-slab partial.
void launch_t2_prefill_mma_sync_slab(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream, std::int32_t kt_sel);

[[nodiscard]] std::size_t t2_prefill_mma_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                                  std::int32_t t);
void launch_t2_row_persistent(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_chunked(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
