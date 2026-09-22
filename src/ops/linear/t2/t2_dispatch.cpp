#include "ops/linear/t2/t2_dispatch.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {
std::atomic<std::uint64_t>& p1_counter() {
    static std::atomic<std::uint64_t> c{0};
    return c;
}
std::atomic<std::uint64_t>& p1_fallback_counter() {
    static std::atomic<std::uint64_t> c{0};
    return c;
}
} // namespace

bool t2_prefill_p1_enabled() {
    const char* e = std::getenv("NINFER_T2_PREFILL_P1");
    return e != nullptr && std::strcmp(e, "1") == 0;
}

bool t2_prefill_p1_applies(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (!t2_prefill_p1_enabled()) { return false; }
    // Exact-tile scratch routing (init contract: persistent no-churn buffers
    // + mode-E warmup at startup; see M7-P1 determinism report).
    // T=8 -> P1-SYNC-8; T=32/64/128 -> P1-SYNC. All other T keep legacy.
    if (t != 8 && t != 32 && t != 64 && t != 128) { return false; }
    if (n <= 0 || (n % 64) != 0 || k <= 0 || (k % 128) != 0) { return false; }
    return true;
}

void t2_prefill_p1_stats(std::uint64_t& p1, std::uint64_t& fallback) {
    p1       = p1_counter().load(std::memory_order_relaxed);
    fallback = p1_fallback_counter().load(std::memory_order_relaxed);
}

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
    // T1-A (widened unpack, bit-identical FP order) serves T=1.
    // M7-P1 scratch (see applies()): T=8 routes to the async 64x8 prototype,
    // T=32/64/128 route to P1-SYNC; everything else keeps legacy.
    if (t == 1) { return launch_t2_gemv_t1a; }
    if (t2_prefill_p1_applies(n, k, t)) {
        return (t == 8) ? launch_t2_prefill_mma_sync8 : launch_t2_prefill_mma_sync;
    }
    if (t <= 32) { return launch_t2_row_persistent; }
    return launch_t2_chunked;
}

void t2_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    const T2Launch launch = select_t2_launch(w.n, w.k, x.ne[1], policy);
    if (launch == launch_t2_prefill_mma_sync8 || launch == launch_t2_prefill_mma_sync) {
        // Prototype requirements beyond (N,K,T): BF16 contiguous I/O.
        // Guard here and fall back silently so an unexpected layout can
        // never throw into the model; routing tests assert selection
        // explicitly via select_t2_launch + stats counters.
        if (x.dtype == DType::BF16 && out.dtype == DType::BF16 && x.is_contiguous() &&
            out.is_contiguous()) {
            p1_counter().fetch_add(1, std::memory_order_relaxed);
            launch(x, w, out, stream);
            return;
        }
        p1_fallback_counter().fetch_add(1, std::memory_order_relaxed);
        const T2Launch legacy =
            (x.ne[1] == 1) ? launch_t2_gemv_t1a
                           : (x.ne[1] <= 32 ? launch_t2_row_persistent : launch_t2_chunked);
        legacy(x, w, out, stream);
        return;
    }
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
