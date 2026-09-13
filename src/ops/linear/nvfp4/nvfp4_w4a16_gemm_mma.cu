#include "ops/linear/nvfp4/nvfp4_w4a16_gemm_mma.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using MmaR64C32 = Nvfp4W4a16MmaSchedule<64, 32, 32, 32, 2>;
using MmaR64C64 = Nvfp4W4a16MmaSchedule<64, 64, 32, 32, 2>;
using MmaR64C128 = Nvfp4W4a16MmaSchedule<64, 128, 32, 32, 1>;

template <class Schedule, bool Full, Nvfp4W4a16Epilogue Epilogue>
void launch_schedule(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t parent_rows = weight.n;
    const std::int32_t k           = x.ne[0];
    const std::int32_t cols        = x.ne[1];
    constexpr int kOutRowsPerCta =
        Epilogue == Nvfp4W4a16Epilogue::SwiGlu ? Schedule::kBlockRows / 2 : Schedule::kBlockRows;
    const dim3 grid(static_cast<unsigned>(div_up(out.ne[0], kOutRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kBlockCols)), 1u);
    const float inverse_dw = 1.0F / weight.weight_scale_divisor;
    nvfp4_w4a16_gemm_mma_kernel<Schedule, Full, Epilogue><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        parent_rows, k, cols, inverse_dw);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, Nvfp4W4a16Epilogue Epilogue>
void launch_route(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int kOutRowsPerCta =
        Epilogue == Nvfp4W4a16Epilogue::SwiGlu ? Schedule::kBlockRows / 2 : Schedule::kBlockRows;
    const bool full =
        (out.ne[0] % kOutRowsPerCta) == 0 && (x.ne[1] % Schedule::kBlockCols) == 0;
    for_each_token_slice(x.ne[1], Schedule::kBlockCols,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor x_slice = x.slice(1, offset, count);
                             Tensor out_slice     = out.slice(1, offset, count);
                             if (full) {
                                 launch_schedule<Schedule, true, Epilogue>(x_slice, weight,
                                                                           out_slice, stream);
                             } else {
                                 launch_schedule<Schedule, false, Epilogue>(x_slice, weight,
                                                                            out_slice, stream);
                             }
                         });
}

template <Nvfp4W4a16Epilogue Epilogue>
void launch_by_tokens(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    if (t <= 32) {
        launch_route<MmaR64C32, Epilogue>(x, weight, out, stream);
    } else if (t <= 64) {
        launch_route<MmaR64C64, Epilogue>(x, weight, out, stream);
    } else {
        launch_route<MmaR64C128, Epilogue>(x, weight, out, stream);
    }
}

} // namespace

void launch_nvfp4_w4a16_mma(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    launch_by_tokens<Nvfp4W4a16Epilogue::Store>(x, weight, out, stream);
}

void launch_nvfp4_w4a16_mma_add(const Tensor& x, const Weight& weight, Tensor& residual,
                                cudaStream_t stream) {
    launch_by_tokens<Nvfp4W4a16Epilogue::AddResidual>(x, weight, residual, stream);
}

void launch_nvfp4_w4a16_mma_swiglu(const Tensor& x, const Weight& weight, Tensor& out,
                                   cudaStream_t stream) {
    launch_by_tokens<Nvfp4W4a16Epilogue::SwiGlu>(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
