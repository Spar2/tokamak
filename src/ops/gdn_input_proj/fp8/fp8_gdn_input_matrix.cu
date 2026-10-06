#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_small_t.cuh"
#include "ops/linear/fp8/fp8_a16_small_t_mma.cuh"
#include "ops/linear/fp8/fp8_a16_gemm_mma.cuh"


namespace ninfer::ops::detail {
namespace {

template <class Geometry, class Output>
struct GdnInputMatrixLauncher {
    template <int ActiveTokens>
    static void launch_exact(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             cudaStream_t stream) {
        using Schedule = typename Fp8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
        constexpr int kTokenTiles =
            (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
        constexpr int kBlocks = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
        const Output output{static_cast<__nv_bfloat16*>(qkv.data),
                            static_cast<__nv_bfloat16*>(z.data)};
        fp8_small_t_kernel<Geometry, ActiveTokens, Schedule>
            <<<kBlocks, Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const __nv_bfloat16*>(weight.scales), output);
        CUDA_CHECK(cudaGetLastError());
    }

    template <int Capacity>
    static void launch_small_mma(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream) {
        constexpr int warps = Capacity <= 8 ? 16 : Capacity <= 24 ? 8 : 4;
        using Schedule = Fp8A16SmallTMmaSchedule<warps, Capacity, warps == 16 ? 1 : 2>;
        const Output output{static_cast<__nv_bfloat16*>(qkv.data),
                            static_cast<__nv_bfloat16*>(z.data)};
        fp8_a16_small_t_mma_kernel<Geometry, Capacity, Schedule, Output, true>
            <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const __nv_bfloat16*>(weight.scales), output, x.ne[1]);
        CUDA_CHECK(cudaGetLastError());
    }

    template <class Schedule>
    static void launch_gemm(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            cudaStream_t stream) {
        static_assert(Output::kQkvRows % Schedule::kBlockRows == 0);
        static_assert(Output::kZRows % Schedule::kBlockRows == 0);
        static_assert(Schedule::kSharedBytes <= 48 * 1024);
        const dim3 grid(Geometry::kOutputRows / Schedule::kBlockRows,
                        (x.ne[1] + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens);
        const Output output{static_cast<__nv_bfloat16*>(qkv.data),
                            static_cast<__nv_bfloat16*>(z.data)};
        fp8_a16_gemm_mma_kernel<Geometry, Schedule, false>
            <<<grid, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const __nv_bfloat16*>(weight.scales), output, x.ne[1]);
        CUDA_CHECK(cudaGetLastError());
    }

    static void launch_matrix(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              cudaStream_t stream) {
        // SIMT for the latency regime, bounded MMA column capacities, then amortized weight decode.
        const int columns = x.ne[1];
        if (columns == 2) return launch_exact<2>(x, weight, qkv, z, stream);
        if (columns == 3) return launch_exact<3>(x, weight, qkv, z, stream);
        if (columns == 4) return launch_exact<4>(x, weight, qkv, z, stream);
        if (columns <= 8) return launch_small_mma<8>(x, weight, qkv, z, stream);
        if (columns <= 16) return launch_small_mma<16>(x, weight, qkv, z, stream);
        if (columns <= 24) return launch_small_mma<24>(x, weight, qkv, z, stream);
        if (columns <= 32) return launch_small_mma<32>(x, weight, qkv, z, stream);
        if (columns <= 64)
            return launch_gemm<Fp8A16GemmSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, qkv, z,
                                                                              stream);
        if (columns <= 96)
            return launch_gemm<Fp8A16GemmSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, qkv, z,
                                                                              stream);
        return launch_gemm<Fp8A16GemmSchedule<64, 128, 64, 32, 16, 2, 2>>(x, weight, qkv, z,
                                                                          stream);
    }
};

} // namespace

void fp8_gdn_input_matrix_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream) {
    GdnInputMatrixLauncher<Fp8GdnInputGeometry, Fp8GdnInputOutput>::launch_matrix(x, weight, qkv,
                                                                                  z, stream);
}

// Ornith-1.5-9B 12288x4096 parent.
void fp8_gdn_input_4096_matrix_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                      Tensor& z, cudaStream_t stream) {
    GdnInputMatrixLauncher<Fp8GdnInput4096Geometry, Fp8GdnInput4096Output>::launch_matrix(
        x, weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
