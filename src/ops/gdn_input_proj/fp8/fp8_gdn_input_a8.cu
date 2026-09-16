#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, class Output>
struct GdnInputA8Launcher {
    using Schedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;

    static void check() {
        static_assert((Output::kQkvRows % Schedule::kBlockRows) == 0);
        static_assert((Output::kZRows % Schedule::kBlockRows) == 0);
    }

    template <bool FullTokens>
    static void launch_mma(const Weight& weight, Tensor& qkv, Tensor& z, Fp8A8Workspace workspace,
                           std::int32_t tokens, cudaStream_t stream) {
        constexpr int kRowTiles = Geometry::kOutputRows / Schedule::kBlockRows;
        const int token_tiles   = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
        const int blocks        = kRowTiles * token_tiles;
        const Output output{static_cast<__nv_bfloat16*>(qkv.data),
                            static_cast<__nv_bfloat16*>(z.data)};

        if constexpr (Schedule::kSharedBytes > 48 * 1024) {
            static const cudaError_t attribute = cudaFuncSetAttribute(
                fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8IdentityEpilogue, Output>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
            CUDA_CHECK(attribute);
        }
        fp8_mma_kernel<Geometry, Schedule, FullTokens>
            <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
                workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{},
                output);
        CUDA_CHECK(cudaGetLastError());
    }

    static void launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                       Fp8A8Workspace workspace, cudaStream_t stream) {
        check();
        launch_fp8_a8_quantize(x, weight, workspace, stream);
        if ((x.ne[1] % Schedule::kBlockTokens) == 0) {
            launch_mma<true>(weight, qkv, z, workspace, x.ne[1], stream);
        } else {
            launch_mma<false>(weight, qkv, z, workspace, x.ne[1], stream);
        }
    }
};

} // namespace

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream) {
    GdnInputA8Launcher<Fp8GdnInputGeometry, Fp8GdnInputOutput>::launch(x, weight, qkv, z,
                                                                      workspace, stream);
}

// Ornith-1.5-9B 12288x4096 parent.
void fp8_gdn_input_4096_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                  Fp8A8Workspace workspace, cudaStream_t stream) {
    GdnInputA8Launcher<Fp8GdnInput4096Geometry, Fp8GdnInput4096Output>::launch(x, weight, qkv, z,
                                                                               workspace, stream);
}

} // namespace ninfer::ops::detail
