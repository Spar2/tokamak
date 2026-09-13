#pragma once

// NVFP4 W4A16 large-T GEMM: dequant packed E2M1+E4M3 tiles into BF16 shared memory,
// then m16n8k16 BF16 Tensor Core MMA. Activations stay BF16. No W4A4.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int BlockRows_, int BlockCols_, int WarpRows_, int WarpCols_, int MinBlocks_>
struct Nvfp4W4a16MmaSchedule {
    static constexpr int kBlockRows  = BlockRows_;
    static constexpr int kBlockCols  = BlockCols_;
    static constexpr int kBlockK     = 64;
    static constexpr int kWarpRows   = WarpRows_;
    static constexpr int kWarpCols   = WarpCols_;
    static constexpr int kMinBlocks  = MinBlocks_;
    static constexpr int kWarpGridM  = kBlockRows / kWarpRows;
    static constexpr int kWarpGridN  = kBlockCols / kWarpCols;
    static constexpr int kWarps      = kWarpGridM * kWarpGridN;
    static constexpr int kThreads    = kWarps * 32;
    static constexpr int kMmaRows    = kWarpRows / 16;
    static constexpr int kMmaCols    = kWarpCols / 8;
    static constexpr int kMmaKSteps  = kBlockK / 16;
    static constexpr int kStages     = 2;
    static constexpr int kCodeBytes  = kBlockK / 2;
    static constexpr int kScaleBytes = kBlockK / 16;
    static constexpr int kSharedBytes =
        kBlockRows * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kStages * kBlockCols * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kStages * kBlockRows * kCodeBytes + kStages * kBlockRows * kScaleBytes;

    static_assert(kBlockRows % kWarpRows == 0 && kBlockCols % kWarpCols == 0);
    static_assert(kWarpRows % 16 == 0 && kWarpCols % 8 == 0);
    static_assert(kThreads >= 32 && kThreads <= 1024);
    static_assert(kSharedBytes <= 99 * 1024);
};

__device__ __forceinline__ int nvfp4_mma_swizzle_k64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

enum class Nvfp4W4a16Epilogue { Store, AddResidual, SwiGlu };

template <class Schedule, bool Full, Nvfp4W4a16Epilogue Epilogue>
__global__ __launch_bounds__(Schedule::kThreads,
                             Schedule::kMinBlocks) void nvfp4_w4a16_gemm_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out, std::int32_t rows,
    std::int32_t k, std::int32_t cols, float inverse_dw) {
    constexpr bool AddResidual = Epilogue == Nvfp4W4a16Epilogue::AddResidual;
    constexpr bool SwiGlu      = Epilogue == Nvfp4W4a16Epilogue::SwiGlu;
    using S              = Schedule;
    constexpr int BM     = S::kBlockRows;
    constexpr int BN     = S::kBlockCols;
    constexpr int BK     = S::kBlockK;
    constexpr int WM     = S::kWarpRows;
    constexpr int WN     = S::kWarpCols;
    constexpr int MT     = S::kMmaRows;
    constexpr int NT     = S::kMmaCols;
    constexpr int KSUB   = S::kMmaKSteps;
    constexpr int STAGES = S::kStages;

    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[STAGES][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[STAGES][BM * S::kCodeBytes];
    __shared__ __align__(16) std::uint8_t Sr[STAGES][BM * S::kScaleBytes];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / S::kWarpGridN;
    const int wn   = warp % S::kWarpGridN;

    constexpr int kOutRowsPerCta = SwiGlu ? BM / 2 : BM;
    const int row0               = static_cast<int>(blockIdx.x) * kOutRowsPerCta;
    const int col0               = static_cast<int>(blockIdx.y) * BN;
    const int intermediate       = rows / 2;
    auto parent_row              = [&](int local_row) {
        if constexpr (!SwiGlu) { return row0 + local_row; }
        return local_row < kOutRowsPerCta ? row0 + local_row
                                          : row0 + intermediate + (local_row - kOutRowsPerCta);
    };
    const int k_tiles = k / BK;
    const int k_scale_tiles = k / 64;

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            acc[mi][ni][0] = 0.0f;
            acc[mi][ni][1] = 0.0f;
            acc[mi][ni][2] = 0.0f;
            acc[mi][ni][3] = 0.0f;
        }
    }

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;
    const int mma_row  = lane >> 2;
    const int mma_col  = lane & 3;

    auto stage_activation = [&](int stage, int k_tile) {
        const int k0 = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += S::kThreads) {
            const int local_col = item / (BK / 8);
            const int k8        = item - local_col * (BK / 8);
            const int kk        = k0 + k8 * 8;
            const int col       = col0 + local_col;
            auto* dst           = &Bs[stage][local_col * BK + nvfp4_mma_swizzle_k64(local_col, k8 * 8)];
            if constexpr (Full) {
                cp_async<16>(dst, &x[static_cast<std::int64_t>(col) * k + kk]);
            } else if (col < cols && kk + 8 <= k) {
                cp_async<16>(dst, &x[static_cast<std::int64_t>(col) * k + kk]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
    };

    auto stage_quant = [&](int stage, int k_tile) {
        const int k0         = k_tile * BK;
        const int scale_tile = k0 / 64;
#pragma unroll 1
        for (int item = tid; item < BM * 2; item += S::kThreads) {
            const int local_row = item / 2;
            const int half      = item & 1;
            const int row       = parent_row(local_row);
            auto* dst           = &Cr[stage][local_row * S::kCodeBytes + half * 16];
            if constexpr (Full) {
                cp_async<16>(dst, &codes[static_cast<std::int64_t>(row) * (k / 2) + k0 / 2 +
                                         half * 16]);
            } else if (row < rows) {
                cp_async<16>(dst, &codes[static_cast<std::int64_t>(row) * (k / 2) + k0 / 2 +
                                         half * 16]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
#pragma unroll 1
        for (int local_row = tid; local_row < BM; local_row += S::kThreads) {
            const int row = parent_row(local_row);
            auto* dst     = &Sr[stage][local_row * S::kScaleBytes];
            if constexpr (Full) {
                const int m_tile   = row / 128;
                const int rem      = row - m_tile * 128;
                const int quartile = rem / 32;
                const int rmod     = rem - quartile * 32;
                const int offset =
                    (m_tile * k_scale_tiles + scale_tile) * 512 + rmod * 16 + quartile * 4;
                *reinterpret_cast<std::uint32_t*>(dst) =
                    *reinterpret_cast<const std::uint32_t*>(scales + offset);
            } else if (row < rows) {
                const int m_tile   = row / 128;
                const int rem      = row - m_tile * 128;
                const int quartile = rem / 32;
                const int rmod     = rem - quartile * 32;
                const int offset =
                    (m_tile * k_scale_tiles + scale_tile) * 512 + rmod * 16 + quartile * 4;
                *reinterpret_cast<std::uint32_t*>(dst) =
                    *reinterpret_cast<const std::uint32_t*>(scales + offset);
            } else {
                *reinterpret_cast<std::uint32_t*>(dst) = 0;
            }
        }
    };

    auto decode_weight = [&](int stage) {
        for (int local_row = warp; local_row < BM; local_row += S::kWarps) {
            const std::uint8_t packed = Cr[stage][local_row * S::kCodeBytes + lane];
            const float2 values       = decode_nvfp4_e2m1x2(packed);
            const int group           = lane >> 3;
            const float scale =
                decode_nvfp4_e4m3(Sr[stage][local_row * S::kScaleBytes + group]) * inverse_dw;
            const int k0 = lane * 2;
            auto* dst    = &As[local_row * BK];
            dst[nvfp4_mma_swizzle_k64(local_row, k0)]     = __float2bfloat16_rn(values.x * scale);
            dst[nvfp4_mma_swizzle_k64(local_row, k0 + 1)] = __float2bfloat16_rn(values.y * scale);
        }
    };

#pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
        if (stage < k_tiles) {
            stage_activation(stage, stage);
            stage_quant(stage, stage);
        }
        cp_commit();
    }

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile % STAGES;
        cp_wait<STAGES - 1>();
        __syncthreads();
        decode_weight(stage);
        __syncthreads();

        unsigned a_frag[MT][4];
        unsigned b_frag[NT][2];
#pragma unroll
        for (int ki = 0; ki < KSUB; ++ki) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = wm * WM + mi * 16 + a_rowoff;
                const int col = ki * 16 + a_coloff;
                ldmatrix_x4(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                            smem_addr(&As[row * BK + nvfp4_mma_swizzle_k64(row, col)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int row = wn * WN + ni * 8 + b_rin;
                const int col = ki * 16 + b_koff;
                ldmatrix_x2(b_frag[ni][0], b_frag[ni][1],
                            smem_addr(&Bs[stage][row * BK + nvfp4_mma_swizzle_k64(row, col)]));
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                             a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                             b_frag[ni][0], b_frag[ni][1]);
                }
            }
        }

        __syncthreads();
        const int prefetch = k_tile + STAGES;
        if (prefetch < k_tiles) {
            stage_activation(stage, prefetch);
            stage_quant(stage, prefetch);
        }
        cp_commit();
    }

    if constexpr (SwiGlu) {
        __shared__ float Csmem[BM * BN];
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int local0 = wm * WM + mi * 16 + mma_row;
            const int local1 = local0 + 8;
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int local_col0                    = wn * WN + ni * 8 + 2 * mma_col;
                const int local_col1                    = local_col0 + 1;
                Csmem[local0 * BN + local_col0]         = acc[mi][ni][0];
                Csmem[local0 * BN + local_col1]         = acc[mi][ni][1];
                Csmem[local1 * BN + local_col0]         = acc[mi][ni][2];
                Csmem[local1 * BN + local_col1]         = acc[mi][ni][3];
            }
        }
        __syncthreads();
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int local0 = wm * WM + mi * 16 + mma_row;
            const int local1 = local0 + 8;
            if (local0 >= kOutRowsPerCta) { continue; }
            const int out_row0 = row0 + local0;
            const int out_row1 = row0 + local1;
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int local_col0 = wn * WN + ni * 8 + 2 * mma_col;
                const int local_col1 = local_col0 + 1;
                const int col0_g     = col0 + local_col0;
                const int col1_g     = col0 + local_col1;
                auto emit            = [&](int out_row, int local_row, int col, int local_col) {
                    if constexpr (!Full) {
                        if (out_row >= intermediate || col >= cols) { return; }
                    }
                    const float gate = Csmem[local_row * BN + local_col];
                    const float up   = Csmem[(local_row + kOutRowsPerCta) * BN + local_col];
                    out[static_cast<std::int64_t>(col) * intermediate + out_row] =
                        __float2bfloat16_rn(silu(gate) * up);
                };
                emit(out_row0, local0, col0_g, local_col0);
                emit(out_row0, local0, col1_g, local_col1);
                if (local1 < kOutRowsPerCta) {
                    emit(out_row1, local1, col0_g, local_col0);
                    emit(out_row1, local1, col1_g, local_col1);
                }
            }
        }
        return;
    }

#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int output_row0 = row0 + wm * WM + mi * 16 + mma_row;
        const int output_row1 = output_row0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int output_col0 = col0 + wn * WN + ni * 8 + 2 * mma_col;
            const int output_col1 = output_col0 + 1;
            const float* values   = acc[mi][ni];
            auto store            = [&](int row, int col, float value) {
                if constexpr (!Full) {
                    if (row >= rows || col >= cols) { return; }
                }
                auto* slot = &out[static_cast<std::int64_t>(col) * rows + row];
                if constexpr (AddResidual) { value += __bfloat162float(*slot); }
                *slot = __float2bfloat16_rn(value);
            };
            store(output_row0, output_col0, values[0]);
            store(output_row0, output_col1, values[1]);
            store(output_row1, output_col0, values[2]);
            store(output_row1, output_col1, values[3]);
        }
    }
}

} // namespace ninfer::ops::detail
