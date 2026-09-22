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
    // Final sync-P1 dispatch (measured crossover: masked P1 beats legacy at
    // every T>=8, min 2.7x at fg-T9; init contract: persistent no-churn
    // buffers + mode-E warmup at startup).
    // T=8 -> P1-SYNC-8; exact T=32/64/128 -> P1-SYNC; other T>=9 (tails)
    // -> masked P1-SYNC. T<=7 keep legacy.
    if (t < 8) { return false; }
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
    // M7-P1 final dispatch: T=1 T1-A; T=2..7 legacy row-persistent;
    // T>=8 synchronous P1 (T=8 sync8, exact 32/64/128 sync32, tails masked).
    if (t == 1) { return launch_t2_gemv_t1a; }
    if (t2_prefill_p1_applies(n, k, t)) {
        if (t == 8) { return launch_t2_prefill_mma_sync8; }
        if (t == 32 || t == 64 || t == 128) { return launch_t2_prefill_mma_sync; }
        return launch_t2_prefill_mma_sync_mt;
    }
    if (t <= 32) { return launch_t2_row_persistent; }
    return launch_t2_chunked;
}

void t2_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    const T2Launch launch = select_t2_launch(w.n, w.k, x.ne[1], policy);
    if (launch == launch_t2_prefill_mma_sync8 || launch == launch_t2_prefill_mma_sync ||
        launch == launch_t2_prefill_mma_sync_mt) {
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
