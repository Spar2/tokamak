#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

using M32N64                      = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128                     = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128                     = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined           = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident            = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;
constexpr std::int32_t kTmaBlockM = 256;

template <class Geometry, class Schedule>
void launch_gemm(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace workspace,
                 std::int32_t tokens, float alpha, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                       Geometry::kOutputRows};
    nvfp4_w4a4_mma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        output);
    CUDA_CHECK(cudaGetLastError());
}

template <class ActivationGeometry>
void launch_quantize_exact(const Tensor& x, Nvfp4W4a4Workspace workspace, float dx,
                           cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    constexpr int kThreads    = 256;
    const std::int32_t tasks  = tokens * ActivationGeometry::kGroupsPerRow;
    nvfp4_w4a4_quantize_kernel<ActivationGeometry>
        <<<(tasks + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), workspace.codes, workspace.scales, tokens,
            dx);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace workspace,
                    std::int32_t tokens, float alpha, cudaStream_t stream) {
    constexpr bool kResidualGeometry = std::is_same_v<Geometry, Nvfp4Residual6144Geometry> ||
                                       std::is_same_v<Geometry, Nvfp4Residual17408Geometry>;
    if (tokens >= 1024 && (tokens % kTmaBlockM) == 0) {
        launch_nvfp4_w4a4_tma_linear(
            resolve_nvfp4_problem(Geometry::kOutputRows, Geometry::kInputRows), workspace.codes,
            workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            tokens, alpha, stream);
    } else if (tokens <= 64) {
        launch_gemm<Geometry, M32N64>(weight, out, workspace, tokens, alpha, stream);
    } else if (tokens <= 96) {
        launch_gemm<Geometry, M32N128>(weight, out, workspace, tokens, alpha, stream);
    } else if (tokens <= 128) {
        if constexpr (kResidualGeometry) {
            launch_gemm<Geometry, M32N128>(weight, out, workspace, tokens, alpha, stream);
        } else {
            launch_gemm<Geometry, M128N128Pipelined>(weight, out, workspace, tokens, alpha, stream);
        }
    } else if (tokens <= 192) {
        launch_gemm<Geometry, M64N128>(weight, out, workspace, tokens, alpha, stream);
    } else if (tokens <= 384) {
        launch_gemm<Geometry, M128N128Resident>(weight, out, workspace, tokens, alpha, stream);
    } else if (tokens <= 512) {
        if constexpr (Geometry::kOutputRows == Nvfp4GdnInputGeometry::kOutputRows) {
            launch_gemm<Geometry, M128N128Resident>(weight, out, workspace, tokens, alpha, stream);
        } else {
            launch_gemm<Geometry, M128N128Pipelined>(weight, out, workspace, tokens, alpha, stream);
        }
    } else {
        launch_gemm<Geometry, M128N128Resident>(weight, out, workspace, tokens, alpha, stream);
    }
}

template <class Geometry, class Schedule>
void launch_gemm_add(const Weight& weight, Tensor& residual, Nvfp4W4a4Workspace workspace,
                     std::int32_t tokens, float alpha, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    auto* output = static_cast<__nv_bfloat16*>(residual.data);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha,
        Nvfp4AddResidualEpilogue{output, Geometry::kOutputRows},
        Nvfp4ContiguousOutput{output, Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_problem_add(const Weight& weight, Tensor& residual, Nvfp4W4a4Workspace workspace,
                        std::int32_t tokens, float alpha, cudaStream_t stream) {
    if (tokens >= 1024 && (tokens % kTmaBlockM) == 0) {
        launch_nvfp4_w4a4_tma_linear_add(
            resolve_nvfp4_problem(Geometry::kOutputRows, Geometry::kInputRows), workspace.codes,
            workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(residual.data), tokens, alpha, stream);
        return;
    }
    if (tokens <= 64) {
        launch_gemm_add<Geometry, M32N64>(weight, residual, workspace, tokens, alpha, stream);
    } else if (tokens <= 128) {
        launch_gemm_add<Geometry, M32N128>(weight, residual, workspace, tokens, alpha, stream);
    } else if (tokens <= 192) {
        launch_gemm_add<Geometry, M64N128>(weight, residual, workspace, tokens, alpha, stream);
    } else if (tokens <= 384) {
        launch_gemm_add<Geometry, M128N128Resident>(weight, residual, workspace, tokens, alpha,
                                                    stream);
    } else if (tokens <= 512) {
        launch_gemm_add<Geometry, M128N128Pipelined>(weight, residual, workspace, tokens, alpha,
                                                     stream);
    } else {
        launch_gemm_add<Geometry, M128N128Resident>(weight, residual, workspace, tokens, alpha,
                                                    stream);
    }
}

void quantize_dynamic(const Tensor& x, Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    switch (x.ne[0]) {
    case Nvfp4Activation4096Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation4096Geometry>(x, workspace, kNvfp4DynamicPerK16Dx,
                                                           stream);
        return;
    case Nvfp4Activation12288Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation12288Geometry>(x, workspace, kNvfp4DynamicPerK16Dx,
                                                            stream);
        return;
    default:
        throw std::invalid_argument("nvfp4 dynamic W4A4: unsupported K");
    }
}

} // namespace

void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                                cudaStream_t stream) {
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 requires caller workspace");
    }
    switch (weight.k) {
    case Nvfp4Activation5120Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation5120Geometry>(x, workspace,
                                                           weight.input_scale_divisor, stream);
        return;
    case Nvfp4Activation6144Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation6144Geometry>(x, workspace,
                                                           weight.input_scale_divisor, stream);
        return;
    case Nvfp4Activation17408Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation17408Geometry>(x, workspace,
                                                            weight.input_scale_divisor, stream);
        return;
    default:
        throw std::invalid_argument("nvfp4 W4A4 quantize: unsupported K");
    }
}

void launch_nvfp4_w4a4(const Tensor& x, const Weight& weight, Tensor& out,
                       Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    const std::int32_t tokens = x.ne[1];
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::AttnInput:
        launch_problem<Nvfp4AttnInputGeometry>(weight, out, workspace, tokens, alpha, stream);
        return;
    case Nvfp4Problem::GdnInput:
        launch_problem<Nvfp4GdnInputGeometry>(weight, out, workspace, tokens, alpha, stream);
        return;
    case Nvfp4Problem::MlpGateUp:
        launch_problem<Nvfp4MlpGateUpGeometry>(weight, out, workspace, tokens, alpha, stream);
        return;
    case Nvfp4Problem::MlpGateUp4096:
        throw std::invalid_argument(
            "nvfp4 W4A4: 4096-K requires launch_nvfp4_dynamic_w4a4, not calibrated d_x");
    case Nvfp4Problem::MlpDown12288:
        throw std::invalid_argument(
            "nvfp4 W4A4: 12288-K requires launch_nvfp4_dynamic_w4a4, not calibrated d_x");
    case Nvfp4Problem::Residual6144:
        launch_problem<Nvfp4Residual6144Geometry>(weight, out, workspace, tokens, alpha, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch_problem<Nvfp4Residual17408Geometry>(weight, out, workspace, tokens, alpha, stream);
        return;
    }
}

void launch_nvfp4_dynamic_w4a4_quantize(const Tensor& x, Nvfp4W4a4Workspace workspace,
                                        cudaStream_t stream) {
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("nvfp4 dynamic W4A4 requires caller workspace");
    }
    quantize_dynamic(x, workspace, stream);
}

void launch_nvfp4_dynamic_w4a4_mma(const Tensor& x, const Weight& weight, Tensor& out,
                                   Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    const float alpha = 1.0F / (kNvfp4DynamicPerK16Dx * weight.weight_scale_divisor);
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::MlpGateUp4096:
        launch_problem<Nvfp4MlpGateUp4096Geometry>(weight, out, workspace, x.ne[1], alpha, stream);
        return;
    case Nvfp4Problem::MlpDown12288:
        launch_problem<Nvfp4MlpDown12288Geometry>(weight, out, workspace, x.ne[1], alpha, stream);
        return;
    default:
        throw std::invalid_argument("nvfp4 dynamic W4A4: unsupported problem");
    }
}

void launch_nvfp4_dynamic_w4a4(const Tensor& x, const Weight& weight, Tensor& out,
                               Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    launch_nvfp4_dynamic_w4a4_quantize(x, workspace, stream);
    launch_nvfp4_dynamic_w4a4_mma(x, weight, out, workspace, stream);
}

void launch_nvfp4_dynamic_w4a4_add(const Tensor& x, const Weight& weight, Tensor& residual,
                                   Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    launch_nvfp4_dynamic_w4a4_quantize(x, workspace, stream);
    const float alpha = 1.0F / (kNvfp4DynamicPerK16Dx * weight.weight_scale_divisor);
    if (resolve_nvfp4_problem(weight.n, weight.k) != Nvfp4Problem::MlpDown12288) {
        throw std::invalid_argument("nvfp4 dynamic W4A4 add: unsupported problem");
    }
    launch_problem_add<Nvfp4MlpDown12288Geometry>(weight, residual, workspace, x.ne[1], alpha,
                                                  stream);
}

} // namespace ninfer::ops::detail
