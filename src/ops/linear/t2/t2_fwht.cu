// Production FWHT+sign path for the Bonsai ternary profile (M5 step 3).
// One 1024-thread block per (1024-chunk, token); shared-memory butterfly;
// 1/sqrt(1024) pre-scale (orthonormal convention, matches Prism CPU impl).
#include "ops/linear/t2/t2_rotation.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHadamardBlock = 1024;
constexpr float kInvSqrt1024          = 0.03125F;

__global__ void t2_fwht_sign_kernel(const __nv_bfloat16* __restrict__ x,
                                    const float* __restrict__ signs,
                                    __nv_bfloat16* __restrict__ out, std::int32_t k,
                                    std::int32_t t) {
    const std::int32_t blk = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t tok = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t base = blk * kHadamardBlock + lane;
    __shared__ float sm[kHadamardBlock];
    float v = 0.0F;
    if (base < k) {
        v = __bfloat162float(x[static_cast<std::size_t>(tok) * k + base]) * signs[base];
    }
    sm[lane] = v * kInvSqrt1024;
    __syncthreads();
    // Redundant-but-benign mapping: each pair claimed twice with identical writes.
    for (int len = 1; len < kHadamardBlock; len <<= 1) {
        const int i0 = (lane / (2 * len)) * (2 * len) + (lane % len);
        const int i1 = i0 + len;
        const float u = sm[i0];
        const float w = sm[i1];
        __syncthreads();
        sm[i0] = u + w;
        sm[i1] = u - w;
        __syncthreads();
    }
    if (base < k) {
        out[static_cast<std::size_t>(tok) * k + base] = __float2bfloat16(sm[lane]);
    }
}

__global__ void t2_fwht_sign_inverse_kernel(const __nv_bfloat16* __restrict__ x,
                                            const float* __restrict__ signs,
                                            __nv_bfloat16* __restrict__ out, std::int32_t k,
                                            std::int32_t t) {
    const std::int32_t blk = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t tok = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t base = blk * kHadamardBlock + lane;
    __shared__ float sm[kHadamardBlock];
    float v = 0.0F;
    if (base < k) {
        v = __bfloat162float(x[static_cast<std::size_t>(tok) * k + base]);
    }
    sm[lane] = v * kInvSqrt1024;
    __syncthreads();
    for (int len = 1; len < kHadamardBlock; len <<= 1) {
        const int i0 = (lane / (2 * len)) * (2 * len) + (lane % len);
        const int i1 = i0 + len;
        const float u = sm[i0];
        const float w = sm[i1];
        __syncthreads();
        sm[i0] = u + w;
        sm[i1] = u - w;
        __syncthreads();
    }
    if (base < k) {
        out[static_cast<std::size_t>(tok) * k + base] =
            __float2bfloat16(sm[lane] * signs[base]);
    }
}

} // namespace

void t2_fwht_sign(const Tensor& x, const float* signs, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("t2_fwht_sign: x/out must be BF16");
    }
    if (x.ne[0] != out.ne[0] || x.ne[1] != out.ne[1] || x.ne[2] != 1 || x.ne[3] != 1 ||
        out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("t2_fwht_sign: shape mismatch, expected [K,T]");
    }
    if (x.ne[0] <= 0 || x.ne[1] <= 0 || (x.ne[0] % kHadamardBlock) != 0) {
        throw std::invalid_argument("t2_fwht_sign: K must be a positive multiple of 1024");
    }
    if (signs == nullptr) { throw std::invalid_argument("t2_fwht_sign: signs must be non-null"); }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("t2_fwht_sign: x/out must be contiguous");
    }
    const dim3 grid(static_cast<unsigned>(x.ne[0] / kHadamardBlock),
                    static_cast<unsigned>(x.ne[1]), 1u);
    constexpr dim3 block(1024u, 1u, 1u);
    t2_fwht_sign_kernel<<<grid, block, 0u, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), signs,
        static_cast<__nv_bfloat16*>(out.data), x.ne[0], x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void t2_fwht_sign_inverse(const Tensor& x, const float* signs, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("t2_fwht_sign_inverse: x/out must be BF16");
    }
    if (x.ne[0] != out.ne[0] || x.ne[1] != out.ne[1] || x.ne[2] != 1 || x.ne[3] != 1 ||
        out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("t2_fwht_sign_inverse: shape mismatch, expected [K,T]");
    }
    if (x.ne[0] <= 0 || x.ne[1] <= 0 || (x.ne[0] % kHadamardBlock) != 0) {
        throw std::invalid_argument("t2_fwht_sign_inverse: K must be a positive multiple of 1024");
    }
    if (signs == nullptr) {
        throw std::invalid_argument("t2_fwht_sign_inverse: signs must be non-null");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("t2_fwht_sign_inverse: x/out must be contiguous");
    }
    const dim3 grid(static_cast<unsigned>(x.ne[0] / kHadamardBlock),
                    static_cast<unsigned>(x.ne[1]), 1u);
    constexpr dim3 block(1024u, 1u, 1u);
    t2_fwht_sign_inverse_kernel<<<grid, block, 0u, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), signs,
        static_cast<__nv_bfloat16*>(out.data), x.ne[0], x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t t2_fwht_workspace_capacity_bytes(std::int32_t k, std::int32_t min_tokens,
                                             std::int32_t max_tokens) {
    if (k <= 0 || (k % kHadamardBlock) != 0 || min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("t2_fwht workspace: invalid geometry");
    }
    return 0;
}

namespace {

__global__ void t2_gdn_v_group_kernel(const __nv_bfloat16* __restrict__ x,
                                      __nv_bfloat16* __restrict__ out, std::int64_t k,
                                      std::int64_t t, std::int32_t h_qk, std::int32_t rep,
                                      std::int32_t d) {
    const std::int64_t n = k * t;
    for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < n; idx += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t pos = idx % k;
        const std::int64_t tok = idx / k;
        const std::int32_t hh  = static_cast<std::int32_t>(pos / d);
        const std::int32_t i   = static_cast<std::int32_t>(pos % d);
        // Original head hh = r*H_qk+k ([D,H_qk,rep] order) moves to grouped
        // slot k*rep+r ([D,rep,H_qk] order), matching Prism's reshape +
        // permute(0,2,1) before signs+FWHT.
        const std::int32_t r    = hh / h_qk;
        const std::int32_t kk   = hh % h_qk;
        const std::int64_t npos = (static_cast<std::int64_t>(kk) * rep + r) * d + i;
        out[tok * k + npos]     = x[idx];
    }
}

} // namespace

void t2_gdn_v_group(const Tensor& x, std::int32_t qk_heads, std::int32_t value_heads,
                    std::int32_t head_dim, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("t2_gdn_v_group: x/out must be BF16");
    }
    if (x.ne[0] != out.ne[0] || x.ne[1] != out.ne[1] || x.ne[2] != 1 || x.ne[3] != 1 ||
        out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("t2_gdn_v_group: shape mismatch, expected [K,T]");
    }
    if (qk_heads <= 0 || value_heads <= 0 || head_dim <= 0 ||
        (value_heads % qk_heads) != 0) {
        throw std::invalid_argument("t2_gdn_v_group: value heads must be a positive multiple of qk heads");
    }
    const std::int32_t rep = value_heads / qk_heads;
    if (x.ne[0] != static_cast<std::int64_t>(value_heads) * head_dim || x.ne[1] <= 0) {
        throw std::invalid_argument("t2_gdn_v_group: K must equal value_heads*head_dim with T>0");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("t2_gdn_v_group: x/out must be contiguous");
    }
    constexpr unsigned kThreads = 256u;
    const std::int64_t n        = x.ne[0] * x.ne[1];
    const unsigned grid =
        static_cast<unsigned>(std::min<std::int64_t>((n + kThreads - 1) / kThreads, 65535));
    t2_gdn_v_group_kernel<<<grid, kThreads, 0u, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data),
        x.ne[0], x.ne[1], qk_heads, rep, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
