#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops {

/**
 * Selects the Gated DeltaNet decay-gate formula applied to the A tensor.
 * The choice is an explicit caller contract: kernels never infer it from
 * model identity.
 *
 * - ExpScaled:   g[h,t] = -exp(A_log[h]) * softplus(a[h,t] + dt_bias[h]).
 *   Used by Qwen3.6-35B-A3B, Qwen3.5-9B, Qwen3.8-27B, whose A tensors store
 *   log-domain decay rates.
 * - RawMultiply: g[h,t] = A_raw[h] * softplus(a[h,t] + dt_bias[h]).
 *   Used by Ternary Bonsai, whose A tensor stores a direct (negative) decay
 *   scale. Verified against the live model: gate/softplus == ssm_a.
 */
enum class GdnGateFormula : std::uint8_t {
    ExpScaled,
    RawMultiply,
};

/**
 * Prepares Gated DeltaNet decay and update gates:
 *
 *   g[h,t]    = gate_scale(A[h], formula) * softplus(a[h,t] + dt_bias[h])
 *   beta[h,t] = sigmoid(b[h,t]).
 *
 * `a` and `b` are contiguous BF16 [48,T], `A` and `dt_bias` are contiguous
 * FP32 [48], and `g` and `beta` are contiguous FP32 [48,T]. `A` carries
 * log-domain decay rates under ExpScaled and direct decay scales under
 * RawMultiply; `formula` selects the interpretation explicitly. The oracle
 * evaluates the formula naively in FP64; transcendental implementation and
 * intermediate precision are private kernel choices. Inputs and the two
 * outputs must be mutually non-overlapping. There is no workspace or
 * persistent state side effect.
 */
void gdn_gating(const Tensor& a, const Tensor& b, const Tensor& A, const Tensor& dt_bias,
                GdnGateFormula formula, Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace ninfer::ops
