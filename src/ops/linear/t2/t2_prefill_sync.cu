// M7 P1-SYNC: deterministic synchronous tiled PQ2 -> shared BF16 MMA GEMM.
//
// Same architecture as the P1 prototype (BM=64, BN=32, BK=128, one G128 group
// per row per slab, XOR-swizzled shared tiles, ldmatrix + mma_bf16, FP32
// accumulators, RNE store) but with a fully synchronous, single-buffered,
// barrier-separated pipeline designed for bitwise determinism:
//
//   per K slab: sync stage (codes/scales/activations) -> barrier ->
//   decode weight tile -> barrier -> MMA (fresh fragments per k-sub,
//   NO double-buffer rotation, NO prefetch) -> barrier ->
//   next slab reuses shared storage.
//
// NO cp.async. NO TMA. NO double buffering. NO overlapped producer/consumer.
// One CTA owns one output tile across the full K loop; FP32 accumulators
// live in registers for the entire launch; nothing spills to global.
//
// OWNERSHIP (BM=64, BN=32, BK=128, THREADS=128, MT=2, NT=2, KSUB=8):
// - Cr[row*32+half*16] (16B): writer is the unique thread with
//   item=row*2+half (item = tid + k*THREADS covers 0..127 exactly once).
//   Readers: dequant threads, after barrier. Disjoint writes.
// - Sr[item] (2B): writer tid==item for tid<64; tids>=64 write nothing.
//   Readers: dequant (all tids), after barrier.
// - Bs[nl*128+swz(nl,8*k8)] (16B): writer is the unique thread with
//   item=nl*16+k8 (0..511 exactly once). swz permutes 8-blocks bijectively,
//   so the 16 chunks cover each row exactly once. Readers: ldmatrix,
//   after barrier. Disjoint writes.
// - As[row*128+swz(row,16*wrd+c)] (2B): writer is the unique thread with
//   item=row*8+wrd (0..511 exactly once); swz bijective per row.
//   Readers: ldmatrix, after barrier. Disjoint writes.
// - af[mi][4], bf[ni][2], acc[mi][ni][4]: per-thread registers. Each warp
//   (wm,wn) owns rows [wm*32,+32) x cols [wn*16,+16); (mi,ni) subdivide;
//   lane mapping (gid/lid) partitions each 32x16 warp tile into unique
//   (r,c) per thread. CTAs partition (N,T) exactly (caller-validated
//   N%64==0, exact BN tiles). No two threads store the same output.
// - K-slab reuse: As/Bs/Cr/Sr fully overwritten every slab with the same
//   index sets (no slab-dependent aliasing); overwrite happens after a
//   barrier that follows MMA completion, so no consumer can still be
//   reading the previous slab's tile. No async transaction is ever
//   outstanding across a barrier (there are no async ops at all).
#include "ops/linear/t2/t2_prefill_mma.cuh"

#include "core/device.h"

#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

// P1-SYNC local fragment helpers WITH "memory" clobbers. The shared
// mma.cuh ldmatrix/mma asm blocks declare no memory dependency, so the
// compiler is free to reorder shared-memory staging accesses across
// fragment loads (stale-fragment race). These fenced variants force
// correct ordering; if they fix determinism, upstream the clobbers.
__device__ __forceinline__ void sync_ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2,
                                                 unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr)
                 : "memory");
}

__device__ __forceinline__ void sync_ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr)
                 : "memory");
}

__device__ __forceinline__ void sync_mma_bf16(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                 : "memory");
}

// P1-SYNC geometries: 64x32 for T>=32 (grid.y token blocks), 64x8 for T=8.
// The kernel is templated on the schedule; both share the synchronous
// ownership model (see header proof).
using SyncCfg = T2PrefillMmaSchedule<64, 32, 16>;
using Sync8Cfg = T2PrefillMmaSchedule<64, 8, 8>;

template <class Cfg>
__global__ __launch_bounds__(Cfg::THREADS) void t2_prefill_mma_sync_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint16_t* __restrict__ scales, __nv_bfloat16* __restrict__ out, std::int32_t n,
    std::int32_t k, std::int32_t t, std::int32_t kt_sel) {
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
    (void)t; // Token count is expressed in the launch grid, not the body.

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
        // Serialized-diagnostic selection: kt_sel>=0 computes ONLY that slab
        // (partial output); kt_sel<0 runs the full fused K loop. Uniform
        // across all threads (host-provided), so no divergence.
        if (kt_sel >= 0 && kt != kt_sel) { continue; }
        // 1-3. Synchronous staging (plain loads + shared stores).
        // Codes: BM rows x 32B, cooperative 16B chunks.
#pragma unroll 1
        for (int item = tid; item < BM * 2; item += Cfg::THREADS) {
            const int row  = item / 2;
            const int half = item % 2;
            const std::int64_t gi =
                static_cast<std::int64_t>(m0 + row) * groups + kt;
            const uint4 u = load_vec<uint4>(&codes[gi * 32 + half * 16]);
            *reinterpret_cast<uint4*>(&Cr[row * 32 + half * 16]) = u;
        }
        // Scales: BM x F16, one per row per slab.
#pragma unroll 1
        for (int item = tid; item < BM; item += Cfg::THREADS) {
            const std::int64_t gi =
                static_cast<std::int64_t>(m0 + item) * groups + kt;
            Sr[item] = scales[gi];
        }
        // Activations: BN x BK BF16, 16B chunks.
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += Cfg::THREADS) {
            const int nl = item / (BK / 8);
            const int k8 = item - nl * (BK / 8);
            const int kk = kt * BK + k8 * 8;
            const int nn = n0 + nl;
            const uint4 u = load_vec<uint4>(&x[static_cast<std::int64_t>(nn) * k + kk]);
            *reinterpret_cast<uint4*>(&Bs[nl * BK + t2p_swz(nl, k8 * 8)]) = u;
        }
        // 4. Barrier: staging complete, shared tiles owned by consumers.
        __syncthreads();

        // 5. Decode current weight tile into As (swizzled BF16).
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
        // 6. Barrier: decode complete, execution tiles owned by MMA.
        __syncthreads();

        // 7. MMA over k-sub tiles with freshly loaded single-buffered
        // fragments (NO rotation, NO prefetch). Barrier after each load so
        // every thread's fragments are settled before any thread computes;
        // barrier after compute keeps phases strictly separated.
        unsigned af[MT][4];
        unsigned bf[NT][2];
#pragma unroll 1
        for (int ks = 0; ks < KSUB; ++ks) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int ar = wm * WM + mi * 16 + a_rowoff;
                const int ac = ks * 16 + a_coloff;
                sync_ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                 smem_addr(&As[ar * BK + t2p_swz(ar, ac)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int br = wn * WN + ni * 8 + b_rin;
                const int bc = ks * 16 + b_koff;
                sync_ldmatrix_x2(bf[ni][0], bf[ni][1],
                                 smem_addr(&Bs[br * BK + t2p_swz(br, bc)]));
            }
            __syncthreads();
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    sync_mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2],
                                  acc[mi][ni][3], af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                  bf[ni][0], bf[ni][1]);
                }
            }
            __syncthreads();
        }
        // 8. Barrier: all consumers finished with this slab's shared tiles;
        // next slab may overwrite. (Also separates back-edge scheduling.)
        __syncthreads();
    }

    // Epilogue: RNE store to [N,T] ne0-fastest out[(n0+c)*n + m0+r].
    // (mi,ni,lane) partition the CTA tile: verified disjoint in the header.
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

} // namespace

void launch_t2_prefill_mma_sync(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("t2_prefill_mma_sync: x/out must be BF16");
    }
    if (n <= 0 || (n % 64) != 0 || k <= 0 || (k % 128) != 0) {
        throw std::invalid_argument("t2_prefill_mma_sync: need N%64==0, K%128==0");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("t2_prefill_mma_sync: x/out must be contiguous");
    }
    if (t != 32 && t != 64 && t != 128) {
        throw std::invalid_argument("t2_prefill_mma_sync: T must be 32, 64, or 128");
    }
    const dim3 grid(static_cast<unsigned>((n + SyncCfg::BM - 1) / SyncCfg::BM),
                    static_cast<unsigned>((t + SyncCfg::BN - 1) / SyncCfg::BN), 1u);
    constexpr dim3 block(static_cast<unsigned>(SyncCfg::THREADS), 1u, 1u);
    t2_prefill_mma_sync_kernel<SyncCfg><<<grid, block, 0u, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint16_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), n, k,
        t, -1);
    CUDA_CHECK(cudaGetLastError());
}

// P1-SYNC-8: synchronous 64x8 path for T=8 (same ownership model as the
// 64x32 path; replaces the deprecated async prototype in dispatch).
void launch_t2_prefill_mma_sync8(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("t2_prefill_mma_sync8: x/out must be BF16");
    }
    if (n <= 0 || (n % 64) != 0 || k <= 0 || (k % 128) != 0) {
        throw std::invalid_argument("t2_prefill_mma_sync8: need N%64==0, K%128==0");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("t2_prefill_mma_sync8: x/out must be contiguous");
    }
    if (t != 8) { throw std::invalid_argument("t2_prefill_mma_sync8: T must be 8"); }
    const dim3 grid(static_cast<unsigned>((n + Sync8Cfg::BM - 1) / Sync8Cfg::BM),
                    static_cast<unsigned>((t + Sync8Cfg::BN - 1) / Sync8Cfg::BN), 1u);
    constexpr dim3 block(static_cast<unsigned>(Sync8Cfg::THREADS), 1u, 1u);
    t2_prefill_mma_sync_kernel<Sync8Cfg><<<grid, block, 0u, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint16_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), n, k,
        t, -1);
    CUDA_CHECK(cudaGetLastError());
}

// Serialized per-slab diagnostic (NOT a production candidate): computes ONLY
// slab kt_sel's MMA contribution (FP32-accumulated single slab, RNE-stored
// BF16 partial). Used to bisect fused-K-loop ownership: if every slab is
// deterministic in isolation but the fused loop jitters, the bug is in
// cross-slab ownership; if isolated slabs jitter, it is systemic.
void launch_t2_prefill_mma_sync_slab(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream, std::int32_t kt_sel) {
    const std::int32_t n = out.ne[0];
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("t2_prefill_mma_sync_slab: x/out must be BF16");
    }
    if (n <= 0 || (n % 64) != 0 || k <= 0 || (k % 128) != 0) {
        throw std::invalid_argument("t2_prefill_mma_sync_slab: need N%64==0, K%128==0");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("t2_prefill_mma_sync_slab: x/out must be contiguous");
    }
    if (t != 32 && t != 64 && t != 128) {
        throw std::invalid_argument("t2_prefill_mma_sync_slab: T must be 32, 64, or 128");
    }
    if (kt_sel < 0 || kt_sel >= k / 128) {
        throw std::invalid_argument("t2_prefill_mma_sync_slab: kt_sel out of range");
    }
    const dim3 grid(static_cast<unsigned>((n + SyncCfg::BM - 1) / SyncCfg::BM),
                    static_cast<unsigned>((t + SyncCfg::BN - 1) / SyncCfg::BN), 1u);
    constexpr dim3 block(static_cast<unsigned>(SyncCfg::THREADS), 1u, 1u);
    t2_prefill_mma_sync_kernel<SyncCfg><<<grid, block, 0u, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint16_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), n, k,
        t, kt_sel);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
