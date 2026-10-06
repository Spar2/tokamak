// Ternary G128 GEMV (T=1): thread-per-row, factored scales, uint32-vectorized
// unpack, dual accumulators. Production port of the validated M4 XVA prototype.
// Weights: NInfer row-split planes (base 32B/group, F16 scale/group, group 128).
// Logical value = (code - 1); reserved code 3 maps mechanically, never validated here.
// I/O: BF16 activations (A16 policy), FP32 accumulation.
#include "ops/linear/t2/t2_launch.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ float f16_bits_to_float(std::uint16_t bits) {
    __half h;
    memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

__global__ void t2_rowsplit_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                        const std::uint8_t* __restrict__ qdata,
                                        const std::uint16_t* __restrict__ scales,
                                        __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                        std::int32_t k) {
    const std::int32_t r = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= rows) { return; }
    const std::int32_t groups = k / 128;
    const std::uint32_t* b32 =
        reinterpret_cast<const std::uint32_t*>(qdata + static_cast<std::size_t>(r) * groups * 32);
    const std::uint32_t* xw = reinterpret_cast<const std::uint32_t*>(x);
    float acc0 = 0.0F;
    float acc1 = 0.0F;
    for (std::int32_t g = 0; g < groups; ++g) {
        const float s = f16_bits_to_float(scales[static_cast<std::size_t>(r) * groups + g]);
        const std::uint32_t* words = b32 + static_cast<std::size_t>(g) * 8;
        const std::uint32_t* xwords = xw + static_cast<std::size_t>(g) * 64;
        float e = 0.0F;
        float o = 0.0F;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const std::uint32_t u = words[w];
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                const int q0 = static_cast<int>((u >> (j * 2)) & 3u);
                const int q1 = static_cast<int>((u >> ((j + 1) * 2)) & 3u);
                const std::uint32_t xword = xwords[w * 8 + (j >> 1)];
                const float a0 =
                    __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(xword)));
                const float a1 =
                    __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(xword >> 16)));
                e += static_cast<float>(q0 - 1) * a0;
                o += static_cast<float>(q1 - 1) * a1;
            }
        }
        acc0 += s * e;
        acc1 += s * o;
    }
    out[r] = __float2bfloat16(acc0 + acc1);
}

} // namespace

void launch_t2_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const std::int32_t k    = x.ne[0];
    const dim3 grid(static_cast<unsigned>((rows + 255) / 256), 1u, 1u);
    constexpr dim3 block(256u, 1u, 1u);
    t2_rowsplit_gemv_kernel<<<grid, block, 0u, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint16_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
        rows, k);
    CUDA_CHECK(cudaGetLastError());
}

// T1-A: widened direct PQ2 GEMV (T=1). Same execution model, mapping,
// dispatch contract, and FP order as t2_rowsplit_gemv_kernel (bit-identical
// outputs by construction); only global-load width changes:
// - codes: 2x LDG.E.128 (uint4) per group instead of 8x u32. Row bases are
//   always 32B-aligned (row r starts at r*groups*32 from a cudaMalloc base),
//   so the wide loads are unconditional and safe.
// - activations: 2x uint4 per w-slice when x is 16B-aligned (uniform branch,
//   u32 fallback otherwise); u32 lanes read via uint4 .x/.y/.z/.w members
//   (register selects, no shift/mask), so per-pair ALU is unchanged.
// No shared memory, no new sync, same grid (thread-per-row).
__global__ void t2_rowsplit_gemv_t1a_kernel(const __nv_bfloat16* __restrict__ x,
                                            const std::uint8_t* __restrict__ qdata,
                                            const std::uint16_t* __restrict__ scales,
                                            __nv_bfloat16* __restrict__ out,
                                            std::int32_t rows, std::int32_t k) {
    const std::int32_t r = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= rows) { return; }
    const std::int32_t groups = k / 128;
    const std::uint8_t* row_base =
        qdata + static_cast<std::size_t>(r) * static_cast<std::size_t>(groups) * 32;
    const std::uint32_t* xw = reinterpret_cast<const std::uint32_t*>(x);
    const bool xwide        = ((reinterpret_cast<std::uintptr_t>(x) & 15u) == 0u);
    const uint4* xw4   = reinterpret_cast<const uint4*>(x);
    float acc0 = 0.0F;
    float acc1 = 0.0F;
    for (std::int32_t g = 0; g < groups; ++g) {
        const float s = f16_bits_to_float(scales[static_cast<std::size_t>(r) * groups + g]);
        const uint4* crow =
            reinterpret_cast<const uint4*>(row_base + static_cast<std::size_t>(g) * 32);
        const uint4 c0 = crow[0];
        const uint4 c1 = crow[1];
        const std::uint32_t words[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        float e = 0.0F;
        float o = 0.0F;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const std::uint32_t u = words[w];
            std::uint32_t xs[8];
            if (xwide) {
                const uint4 xq0 = xw4[static_cast<std::size_t>(g) * 16 + w * 2];
                const uint4 xq1 = xw4[static_cast<std::size_t>(g) * 16 + w * 2 + 1];
                xs[0] = xq0.x;
                xs[1] = xq0.y;
                xs[2] = xq0.z;
                xs[3] = xq0.w;
                xs[4] = xq1.x;
                xs[5] = xq1.y;
                xs[6] = xq1.z;
                xs[7] = xq1.w;
            } else {
                const std::uint32_t* xb = xw + static_cast<std::size_t>(g) * 64 + w * 8;
#pragma unroll
                for (int m = 0; m < 8; ++m) { xs[m] = xb[m]; }
            }
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                const int q0 = static_cast<int>((u >> (m * 4)) & 3u);
                const int q1 = static_cast<int>((u >> (m * 4 + 2)) & 3u);
                const std::uint32_t xword = xs[m];
                const float a0 =
                    __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(xword)));
                const float a1 =
                    __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(xword >> 16)));
                e += static_cast<float>(q0 - 1) * a0;
                o += static_cast<float>(q1 - 1) * a1;
            }
        }
        acc0 += s * e;
        acc1 += s * o;
    }
    out[r] = __float2bfloat16(acc0 + acc1);
}

void launch_t2_gemv_t1a(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const std::int32_t k    = x.ne[0];
    const dim3 grid(static_cast<unsigned>((rows + 255) / 256), 1u, 1u);
    constexpr dim3 block(256u, 1u, 1u);
    t2_rowsplit_gemv_t1a_kernel<<<grid, block, 0u, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint16_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
        rows, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
