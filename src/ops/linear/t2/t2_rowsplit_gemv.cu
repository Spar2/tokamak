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

} // namespace ninfer::ops::detail
