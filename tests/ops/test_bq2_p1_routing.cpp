// M7-P1 scratch dispatch routing test.
// Proves select_t2_launch routes T=8 to the async P1 path and T=32/64/128
// to P1-SYNC iff the NINFER_T2_PREFILL_P1=1 env gate is on; every other T
// keeps the legacy path. Live smoke (needs NINFER_BQ2_FULL_ART) proves
// t2_dispatch engagement via stats counters; numeric equality dispatch-vs-
// direct is asserted bitwise for T=8/T=1 and under tolerance for P1-SYNC
// (churn-hazard discipline: warmup launches before trusted comparison).
#include "ops/bq2_model_common.h"
#include "ops/linear/t2/t2_dispatch.h"
#include "ops/linear/t2/t2_launch.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::ops::detail;

int failures = 0;
#define CHECK(cond, ...)                                                                            \
    do {                                                                                            \
        if (!(cond)) {                                                                              \
            std::printf("ROUTING-FAIL: ");                                                          \
            std::printf(__VA_ARGS__);                                                               \
            std::printf("\n");                                                                      \
            ++failures;                                                                             \
        }                                                                                           \
    } while (0)

void check_select(bool gate, int t, T2Launch want, const char* want_name) {
    const T2Launch got = select_t2_launch(17408, 5120, t, LinearPolicy::A16Only);
    CHECK(got == want, "gate=%d T=%d want=%s", gate ? 1 : 0, t, want_name);
}

} // namespace

int main() {
    // ---- pure selection, gate OFF: everything legacy ----
    unsetenv("NINFER_T2_PREFILL_P1");
    CHECK(!t2_prefill_p1_enabled(), "gate should be off");
    for (int t : {1, 2, 7, 8, 31, 32, 33, 64, 128}) {
        const T2Launch want =
            (t == 1) ? launch_t2_gemv_t1a : (t <= 32 ? launch_t2_row_persistent : launch_t2_chunked);
        check_select(false, t, want, t == 1 ? "t1a" : (t <= 32 ? "row-persistent" : "chunked"));
    }
    CHECK(!t2_prefill_p1_applies(17408, 5120, 8), "applies must be false with gate off");

    // ---- pure selection, gate ON: T=8 async P1, T=32/64/128 P1-SYNC ----
    setenv("NINFER_T2_PREFILL_P1", "1", 1);
    CHECK(t2_prefill_p1_enabled(), "gate should be on");
    check_select(true, 1, launch_t2_gemv_t1a, "t1a");
    check_select(true, 2, launch_t2_row_persistent, "row-persistent");
    check_select(true, 7, launch_t2_row_persistent, "row-persistent");
    check_select(true, 8, launch_t2_prefill_mma, "prefill-mma");
    check_select(true, 31, launch_t2_row_persistent, "row-persistent");
    check_select(true, 32, launch_t2_prefill_mma_sync, "prefill-sync");
    check_select(true, 33, launch_t2_chunked, "chunked");
    check_select(true, 64, launch_t2_prefill_mma_sync, "prefill-sync");
    check_select(true, 128, launch_t2_prefill_mma_sync, "prefill-sync");
    // Shape guard: odd N never routes to P1 even at supported T (legacy);
    // K%128!=0 is rejected by the pre-existing shape validation (throws).
    CHECK(select_t2_launch(17407, 5120, 32, ops::LinearPolicy::A16Only) == launch_t2_row_persistent,
          "odd N must stay legacy");
    {
        bool threw = false;
        try {
            select_t2_launch(17408, 5119, 32, ops::LinearPolicy::A16Only);
        } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "K%128!=0 must throw (pre-existing validation)");
    }
    CHECK(t2_prefill_p1_applies(17408, 5120, 8), "applies(fg,8) must be true");
    CHECK(t2_prefill_p1_applies(17408, 5120, 32), "applies(fg,32) must be true");
    CHECK(t2_prefill_p1_applies(17408, 5120, 64), "applies(fg,64) must be true");
    CHECK(t2_prefill_p1_applies(17408, 5120, 128), "applies(fg,128) must be true");
    CHECK(!t2_prefill_p1_applies(17408, 5120, 33), "applies(fg,33) must be false");
    CHECK(!t2_prefill_p1_applies(17408, 5120, 7), "applies(fg,7) must be false");

    const char* art_path = std::getenv("NINFER_BQ2_FULL_ART");
    if (art_path == nullptr) {
        std::printf("routing-select OK (no artifact: live smoke skipped)\n");
        unsetenv("NINFER_T2_PREFILL_P1");
        if (failures != 0) { std::printf("ROUTING FAIL (%d)\n", failures); return 1; }
        std::printf("OK ROUTING\n");
        return 0;
    }
    try {
        bq2full::Bq2Model model(art_path);
        DeviceContext& device = model.device;
        cudaStream_t stream   = model.stream;
        const Weight& w       = model.gdn[8].fg; // T2 [17408,5120]
        auto run_case = [&](int T, int expect_p1) {
            // expect_p1: 0=legacy, 1=async P1 (T8), 2=P1-SYNC.
            bq2full::DeviceBuffer dx(5120 * T * 2), dy0(17408 * T * 2), dy1(17408 * T * 2);
            dx.fill(0);
            device.synchronize();
            // Deterministic non-trivial input bits.
            std::vector<std::uint16_t> xb(5120 * T);
            for (std::size_t i = 0; i < xb.size(); ++i) {
                xb[i] = static_cast<std::uint16_t>(0x3C00 + (i * 37u % 512u));
            }
            CUDA_CHECK(cudaMemcpy(dx.p, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
            const Tensor x(dx.p, DType::BF16, {5120, T});
            Tensor y0(dy0.p, DType::BF16, {17408, T});
            Tensor y1(dy1.p, DType::BF16, {17408, T});
            auto b2f = [](std::uint16_t b) {
                const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
                float f;
                std::memcpy(&f, &u, 4);
                return f;
            };
            std::uint64_t p1a = 0, fba = 0, p1b = 0, fbb = 0;
            // Churn-hazard discipline: warmup launches (discarded) before
            // any trusted comparison on fresh buffers.
            if (expect_p1 == 1) {
                launch_t2_prefill_mma(x, w, y0, stream);
                launch_t2_prefill_mma(x, w, y0, stream);
            } else if (expect_p1 == 2) {
                launch_t2_prefill_mma_sync(x, w, y0, stream);
                launch_t2_prefill_mma_sync(x, w, y0, stream);
            }
            device.synchronize();
            t2_prefill_p1_stats(p1a, fba);
            ops::linear(x, w, y0, stream); // goes through t2_dispatch
            t2_prefill_p1_stats(p1b, fbb);
            if (expect_p1 != 0) {
                CHECK(p1b == p1a + 1 && fbb == fba, "T=%d must count one P1 launch", T);
                if (expect_p1 == 1) {
                    launch_t2_prefill_mma(x, w, y1, stream);
                } else {
                    launch_t2_prefill_mma_sync(x, w, y1, stream);
                }
                device.synchronize();
                std::vector<std::uint16_t> b0(17408 * T), b1(17408 * T);
                CUDA_CHECK(
                    cudaMemcpy(b0.data(), dy0.p, b0.size() * 2, cudaMemcpyDeviceToHost));
                CUDA_CHECK(
                    cudaMemcpy(b1.data(), dy1.p, b1.size() * 2, cudaMemcpyDeviceToHost));
                if (expect_p1 == 1) {
                    std::size_t mism = 0;
                    for (std::size_t i = 0; i < b0.size(); ++i) {
                        if (b0[i] != b1[i]) { ++mism; }
                    }
                    CHECK(mism == 0, "T=%d dispatch-vs-direct bitwise mismatch=%zu", T,
                          mism);
                    std::printf("  live T=%-4d P1 bitwise-equal\n", T);
                } else {
                    double mx = 0, se = 0, d = 0, o2 = 0, r2 = 0;
                    for (std::size_t i = 0; i < b0.size(); ++i) {
                        const double a = b2f(b0[i]), b = b2f(b1[i]);
                        const double e = std::fabs(a - b);
                        if (e > mx) { mx = e; }
                        se += e * e;
                        d += a * b;
                        o2 += a * a;
                        r2 += b * b;
                    }
                    const double rmse = std::sqrt(se / b0.size());
                    const double cos =
                        (o2 > 0 && r2 > 0) ? d / std::sqrt(o2 * r2) : 0.0;
                    CHECK(mx <= 1e-3 && cos >= 0.99999,
                          "T=%d sync dispatch-vs-direct maxabs=%.6f cos=%.8f", T, mx, cos);
                    std::printf("  live T=%-4d P1-SYNC routed (maxabs=%.6f cos=%.8f)\n",
                                T, mx, cos);
                }
            } else {
                CHECK(p1b == p1a && fbb == fba, "T=%d must not touch P1 counters", T);
                std::printf("  live T=%-4d legacy routed (counters verified)\n", T);
            }
        };
        run_case(8, 1);
        run_case(32, 2);
        run_case(64, 2);
        run_case(128, 2);
        run_case(33, 0);
        run_case(31, 0);
        run_case(7, 0);
        run_case(2, 0);
        // T=1 legacy is T1-A, not row-persistent.
        {
            bq2full::DeviceBuffer dx1(5120 * 2), dyA1(17408 * 2), dyB1(17408 * 2);
            std::vector<std::uint16_t> xb1(5120, 0x3C00);
            CUDA_CHECK(cudaMemcpy(dx1.p, xb1.data(), xb1.size() * 2, cudaMemcpyHostToDevice));
            const Tensor x1(dx1.p, DType::BF16, {5120, 1});
            Tensor yA1(dyA1.p, DType::BF16, {17408, 1});
            Tensor yB1(dyB1.p, DType::BF16, {17408, 1});
            std::uint64_t p1a = 0, fba = 0, p1b = 0, fbb = 0;
            t2_prefill_p1_stats(p1a, fba);
            ops::linear(x1, w, yA1, stream);
            t2_prefill_p1_stats(p1b, fbb);
            CHECK(p1b == p1a && fbb == fba, "T=1 must not touch P1 counters");
            launch_t2_gemv_t1a(x1, w, yB1, stream);
            device.synchronize();
            std::vector<std::uint16_t> bA1(17408), bB1(17408);
            CUDA_CHECK(cudaMemcpy(bA1.data(), dyA1.p, bA1.size() * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(bB1.data(), dyB1.p, bB1.size() * 2, cudaMemcpyDeviceToHost));
            CHECK(bA1 == bB1, "T=1 dispatch-vs-t1a bitwise mismatch");
            std::printf("  live T=1    T1-A bitwise-equal\n");
        }
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        unsetenv("NINFER_T2_PREFILL_P1");
        return 77;
    }
    unsetenv("NINFER_T2_PREFILL_P1");
    if (failures != 0) { std::printf("ROUTING FAIL (%d)\n", failures); return 1; }
    std::printf("OK ROUTING\n");
    return 0;
}
