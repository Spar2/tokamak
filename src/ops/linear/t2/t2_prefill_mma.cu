// M7-P1 prototype TU: explicit instantiations + launch wrapper.
// NOT production-dispatched (called directly by the P1 micro test).
#include "ops/linear/t2/t2_prefill_mma.cuh"

#include "core/device.h"

#include <stdexcept>

namespace ninfer::ops::detail {

template <class Cfg>
void launch_cfg(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    constexpr int BN     = Cfg::BN;
    const dim3 grid(static_cast<unsigned>((n + Cfg::BM - 1) / Cfg::BM),
                    static_cast<unsigned>((t + BN - 1) / BN), 1u);
    constexpr dim3 block(static_cast<unsigned>(Cfg::THREADS), 1u, 1u);
    t2_prefill_mma_kernel<Cfg><<<grid, block, 0u, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint16_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), n, k,
        t);
    CUDA_CHECK(cudaGetLastError());
}

void launch_t2_prefill_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("t2_prefill_mma: x/out must be BF16");
    }
    if (n <= 0 || (n % 64) != 0 || k <= 0 || (k % 128) != 0) {
        throw std::invalid_argument("t2_prefill_mma: need N%64==0, K%128==0");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("t2_prefill_mma: x/out must be contiguous");
    }
    if (t == 8) {
        launch_cfg<T2PrefillMmaSchedule<64, 8, 8>>(x, w, out, stream);
    } else if (t == 32 || t == 128) {
        launch_cfg<T2PrefillMmaSchedule<64, 32, 16>>(x, w, out, stream);
    } else {
        throw std::invalid_argument("t2_prefill_mma prototype: T must be 8, 32, or 128");
    }
}

} // namespace ninfer::ops::detail
