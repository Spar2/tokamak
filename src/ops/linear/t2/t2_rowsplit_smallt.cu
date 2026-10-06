// Ternary G128 row-persistent small-T (2 <= T <= 32): one thread owns row r x
// all T tokens; each group unpacked once, FMAs across T. T>32 chunked fallback
// loops 32-token slices (host loop in launch_t2_chunked). BF16 I/O, FP32 acc.
#include "ops/linear/t2/t2_launch.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kMaxSmallT = 32;

__device__ __forceinline__ float f16_bits_to_float(std::uint16_t bits) {
    __half h;
    memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

__global__ void t2_rowsplit_persistent_kernel(const __nv_bfloat16* __restrict__ x,
                                             const std::uint8_t* __restrict__ qdata,
                                             const std::uint16_t* __restrict__ scales,
                                             __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                             std::int32_t k, std::int32_t t) {
    const std::int32_t r = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= rows) { return; }
    const std::int32_t groups = k / 128;
    const std::uint32_t* b32 =
        reinterpret_cast<const std::uint32_t*>(qdata + static_cast<std::size_t>(r) * groups * 32);
    float acc[kMaxSmallT];
#pragma unroll
    for (int tt = 0; tt < kMaxSmallT; ++tt) { acc[tt] = 0.0F; }
    for (std::int32_t g = 0; g < groups; ++g) {
        const float s = f16_bits_to_float(scales[static_cast<std::size_t>(r) * groups + g]);
        const std::uint32_t* words = b32 + static_cast<std::size_t>(g) * 8;
        float wv[16];
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const std::uint32_t u = words[w];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                wv[j] = static_cast<float>(static_cast<int>((u >> (j * 2)) & 3u) - 1);
            }
            for (int tt = 0; tt < t; ++tt) {
                const std::uint32_t* xwords = reinterpret_cast<const std::uint32_t*>(
                    x + (static_cast<std::size_t>(tt) * k + static_cast<std::size_t>(g) * 128 +
                         static_cast<std::size_t>(w) * 16));
                float a = 0.0F;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const std::uint32_t xword = xwords[j >> 1];
                    const float aj = __bfloat162float(
                        __ushort_as_bfloat16(static_cast<unsigned short>(
                            (j & 1) ? (xword >> 16) : xword)));
                    a += wv[j] * aj;
                }
                acc[tt] += s * a;
            }
        }
    }
#pragma unroll
    for (int tt = 0; tt < kMaxSmallT; ++tt) {
        if (tt < t) { out[static_cast<std::size_t>(tt) * rows + r] = __float2bfloat16(acc[tt]); }
    }
}

} // namespace

void launch_t2_row_persistent(const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const std::int32_t k    = x.ne[0];
    const std::int32_t t    = x.ne[1];
    if (t < 2 || t > kMaxSmallT) {
        throw std::invalid_argument("t2 row-persistent: T out of [2,32]");
    }
    const dim3 grid(static_cast<unsigned>((rows + 255) / 256), 1u, 1u);
    constexpr dim3 block(256u, 1u, 1u);
    t2_rowsplit_persistent_kernel<<<grid, block, 0u, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint16_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
        rows, k, t);
    CUDA_CHECK(cudaGetLastError());
}

void launch_t2_chunked(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const std::int32_t k    = x.ne[0];
    const std::int32_t t    = x.ne[1];
    if (t <= kMaxSmallT) {
        throw std::invalid_argument("t2 chunked: use row-persistent for T<=32");
    }
    // Correctness-first chunking: 32-token slices sharing one kernel launch shape.
    // (Slices are independent; no cross-slice state.)
    for (std::int32_t t0 = 0; t0 < t; t0 += kMaxSmallT) {
        const std::int32_t tc = (t - t0) < kMaxSmallT ? (t - t0) : kMaxSmallT;
        const dim3 grid(static_cast<unsigned>((rows + 255) / 256), 1u, 1u);
        constexpr dim3 block(256u, 1u, 1u);
        // Slice views are expressed as raw offsets; Tensor is trivially constructible here
        // because launch bounds/views below only need data pointers and extents.
        t2_rowsplit_persistent_kernel<<<grid, block, 0u, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data) + static_cast<std::size_t>(t0) * k,
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint16_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data) + static_cast<std::size_t>(t0) * rows, rows,
            k, tc);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
