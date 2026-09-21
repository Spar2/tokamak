// M7-P1 prototype: tiled PQ2 -> shared BF16 -> MMA prefill GEMM (Full tiles
// only; NOT production-dispatched). Cloned scaffolding from the proven
// W8RowSplitMmaGemmSchedule path (cp.async staging, XOR swizzle, ldmatrix +
// mma_bf16, FP32 accum, RNE store); only the dequant differs (T2 2-bit codes
// + per-128 F16 scale instead of W8G32 int8 + per-32 scales).
//
// CTA tile BM rows x BN tokens, K swept in BK=128 slabs (exactly one G128
// group per row per slab, so scale handling is trivial). Per slab: packed
// codes staged once, decoded once to shared BF16, activations loaded once,
// reused across BM rows (weight reuse xBN) and across BN tokens (activation
// reuse xBM). Grid (ceil(N/BM), ceil(T/BN)); T=128 via 4x BN=32 blocks.
#pragma once

#include "core/tensor.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int BM_, int BN_, int WN_ = 16>
struct T2PrefillMmaSchedule {
    static constexpr int BM = BM_;
    static constexpr int BN = BN_;
    static constexpr int BK = 128;
    static constexpr int WM = 32;
    static constexpr int WN = WN_;
    static constexpr int WARPS_M = BM / WM;
    static constexpr int WARPS_N = BN / WN;
    static constexpr int WARPS   = WARPS_M * WARPS_N;
    static constexpr int THREADS = WARPS * 32;
    static constexpr int MT      = WM / 16;
    static constexpr int NT      = WN / 8;
    static constexpr int KSUB    = BK / 16;
    static constexpr int SMEM_BYTES =
        BM * BK * 2 + BN * BK * 2 + BM * BK + BM * 2;
    static_assert(BM % WM == 0 && BN % WN == 0);
    static_assert(THREADS <= 1024);
    static_assert(SMEM_BYTES <= 48 * 1024);
};

__device__ __forceinline__ int t2p_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ float t2p_f16_to_float(std::uint16_t bits) {
    __half h;
    memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

template <class Cfg>
__global__ __launch_bounds__(Cfg::THREADS) void t2_prefill_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint16_t* __restrict__ scales, __nv_bfloat16* __restrict__ out, std::int32_t n,
    std::int32_t k, std::int32_t t) {
    constexpr int BM = Cfg::BM;
    constexpr int BN = Cfg::BN;
    constexpr int BK = Cfg::BK;
    constexpr int WM = Cfg::WM;
    constexpr int WN = Cfg::WN;
    constexpr int MT = Cfg::MT;
    constexpr int NT = Cfg::NT;
    constexpr int KSUB = Cfg::KSUB;

    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[BM * BK / 4];
    __shared__ __align__(16) std::uint16_t Sr[BM];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Cfg::WARPS_N;
    const int wn   = warp % Cfg::WARPS_N;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    const int m0 = static_cast<int>(blockIdx.x) * BM;
    const int n0 = static_cast<int>(blockIdx.y) * BN;
    const int groups = k / 128;

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

    const int nkt = k / BK;
    for (int kt = 0; kt < nkt; ++kt) {
        // Stage codes: BM rows x 32B, cooperative 16B chunks.
#pragma unroll 1
        for (int item = tid; item < BM * 2; item += Cfg::THREADS) {
            const int row = item / 2;
            const int half = item % 2;
            const int grow = m0 + row;
            const std::int64_t gi =
                static_cast<std::int64_t>(grow) * groups + kt;
            cp_async<16, Cache::cg>(&Cr[row * 32 + half * 16],
                                    &codes[gi * 32 + half * 16]);
        }
        // Stage scales: BM x F16 (one per row per slab).
#pragma unroll 1
        for (int item = tid; item < BM; item += Cfg::THREADS) {
            const int grow = m0 + item;
            const std::int64_t gi =
                static_cast<std::int64_t>(grow) * groups + kt;
            Sr[item] = scales[gi];
        }
        // Stage activations: BN x BK BF16.
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += Cfg::THREADS) {
            const int nl = item / (BK / 8);
            const int k8 = item - nl * (BK / 8);
            const int kk = kt * BK + k8 * 8;
            const int nn = n0 + nl;
            cp_async<16, Cache::cg>(&Bs[nl * BK + t2p_swz(nl, k8 * 8)],
                                    &x[static_cast<std::int64_t>(nn) * k + kk]);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();

        // Dequant: 128 codes/row -> BF16 As (swizzled), scale applied.
        // Cooperative over 32-bit code words: BM*8 words (32B staged per
        // row), 16 codes each.
        for (int item = tid; item < BM * 8; item += Cfg::THREADS) {
            const int row = item / 8;
            const int wrd = item % 8;
            const float scale = t2p_f16_to_float(Sr[row]);
            const std::uint32_t u =
                reinterpret_cast<const std::uint32_t*>(&Cr[row * 32])[wrd];
            const int base = wrd * 16;
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const int code = static_cast<int>((u >> (c * 2)) & 3u);
                const float f  = static_cast<float>(code - 1) * scale;
                As[row * BK + t2p_swz(row, base + c)] = __float2bfloat16_rn(f);
            }
        }
        __syncthreads();

        unsigned af[2][MT][4];
        unsigned bf[2][NT][2];
        auto load_fragments = [&](int slot, int ks) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int ar = wm * WM + mi * 16 + a_rowoff;
                const int ac = ks * 16 + a_coloff;
                ldmatrix_x4(af[slot][mi][0], af[slot][mi][1], af[slot][mi][2],
                            af[slot][mi][3], smem_addr(&As[ar * BK + t2p_swz(ar, ac)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int br = wn * WN + ni * 8 + b_rin;
                const int bc = ks * 16 + b_koff;
                ldmatrix_x2(bf[slot][ni][0], bf[slot][ni][1],
                            smem_addr(&Bs[br * BK + t2p_swz(br, bc)]));
            }
        };

        load_fragments(0, 0);
#pragma unroll
        for (int ks = 0; ks < KSUB; ++ks) {
            const int slot = ks & 1;
            if (ks + 1 < KSUB) { load_fragments(slot ^ 1, ks + 1); }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                             af[slot][mi][0], af[slot][mi][1], af[slot][mi][2], af[slot][mi][3],
                             bf[slot][ni][0], bf[slot][ni][1]);
                }
            }
        }
    }

    // Epilogue: RNE store to [N,T] ne0-fastest out[(n0+c)*n + m0+r].
    for (int mi = 0; mi < MT; ++mi) {
        for (int ni = 0; ni < NT; ++ni) {
            const int r0 = m0 + wm * WM + mi * 16 + gid;
            const int r1 = r0 + 8;
            const int c0 = n0 + wn * WN + ni * 8 + 2 * lid;
            const int c1 = c0 + 1;
            out[static_cast<std::int64_t>(c0) * n + r0] =
                __float2bfloat16_rn(acc[mi][ni][0]);
            out[static_cast<std::int64_t>(c1) * n + r0] =
                __float2bfloat16_rn(acc[mi][ni][1]);
            out[static_cast<std::int64_t>(c0) * n + r1] =
                __float2bfloat16_rn(acc[mi][ni][2]);
            out[static_cast<std::int64_t>(c1) * n + r1] =
                __float2bfloat16_rn(acc[mi][ni][3]);
        }
    }
}

} // namespace ninfer::ops::detail
