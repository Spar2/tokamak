#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::ops {

// Bonsai ternary-profile activation rotation: out = FWHT(signs .* x) / sqrt(1024),
// applied per 1024-element block along the K axis (x/out shape [K,T], BF16).
// Signs are an immutable device-global F32 vector of length K (see artifact
// RotationPlan); passed explicitly, never baked into weights. Internal
// accumulation is FP32; output rounds once to BF16 (A16 contract).
// Workspace: none (out-of-place).
void t2_fwht_sign(const Tensor& x, const float* signs, Tensor& out, cudaStream_t stream);

// Inverse (embedding) rotation: out = signs .* (FWHT(x) / sqrt(1024)).
// Restores the primal basis from Hadamard-latent rows right after embedding
// lookup (h = s * (H z)). Note the order differs from the forward path
// (forward: signs then FWHT; inverse: FWHT then signs); they do not commute.
// Same shapes, dtypes, signs ownership, and numerics as t2_fwht_sign.
void t2_fwht_sign_inverse(const Tensor& x, const float* signs, Tensor& out, cudaStream_t stream);

std::size_t t2_fwht_workspace_capacity_bytes(std::int32_t k, std::int32_t min_tokens,
                                             std::int32_t max_tokens);

} // namespace ninfer::ops
