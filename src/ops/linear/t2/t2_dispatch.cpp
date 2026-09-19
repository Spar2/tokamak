#include "ops/linear/t2/t2_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

T2Launch select_t2_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (n <= 0 || k <= 0 || (k % 128) != 0 || t <= 0) {
        throw std::invalid_argument("t2 linear: unsupported shape or T");
    }
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
        break;
    case LinearPolicy::AllowA4:
        throw std::invalid_argument("t2 linear: unsupported policy");
    }
    // M5-minimal routing (measured on ffn_gate-class geometry; refine per-shape later):
    // T=1 decode GEMV, 2..32 row-persistent, >32 chunked fallback.
    if (t == 1) { return launch_t2_gemv; }
    if (t <= 32) { return launch_t2_row_persistent; }
    return launch_t2_chunked;
}

void t2_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    const T2Launch launch = select_t2_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
