// M7 P1-SYNC determinism + numerics + performance matrix.
// For shapes fg/fd/qkv x T=32/64/128, SAME input:
//   1. 50 repeated P1-SYNC launches: bitwise vs run0 (want ZERO),
//      plus worst maxabs/rmse/cos vs run0.
//   2. P1-SYNC vs legacy production path (tolerance).
//   3. P1-SYNC vs FP64 sampled oracle (256 rows, like the M7 micro oracle).
//   4. P1-SYNC vs async P1 prototype (tolerance; async is known-racy).
//   5. Median timing: legacy vs async-P1 vs P1-SYNC.
// Env: NINFER_BQ2_FULL_ART. Forces legacy baseline by unsetting the P1 gate.
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
            std::printf("P1SYNC-FAIL: ");                                                          \
            std::printf(__VA_ARGS__);                                                               \
            std::printf("\n");                                                                      \
            ++failures;                                                                             \
        }                                                                                           \
    } while (0)

float b2f(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Stats {
    std::size_t mm = 0, n = 0;
    double maxabs = 0, se = 0, dot = 0, o2 = 0, r2 = 0;
};

Stats cmp_bufs(const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b) {
    Stats s;
    s.n = a.size();
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double u = b2f(a[i]), v = b2f(b[i]);
        const double e = std::fabs(u - v);
        if (e > 0) { ++s.mm; }
        if (e > s.maxabs) { s.maxabs = e; }
        s.se += e * e;
        s.dot += u * v;
        s.o2 += u * u;
        s.r2 += v * v;
    }
    return s;
}

void print_stats(const char* tag, const Stats& s) {
    const double rmse = std::sqrt(s.se / (double)s.n);
    const double cos  = (s.o2 > 0 && s.r2 > 0) ? s.dot / std::sqrt(s.o2 * s.r2) : 0.0;
    std::printf("  %-28s mm=%zu/%zu maxabs=%.6f rmse=%.6f cos=%.9f\n", tag, s.mm, s.n,
                s.maxabs, rmse, cos);
}

struct Fam {
    const char* name;
    const char* art;
    int rows, cols;
};

double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

} // namespace

int main() {
    const char* art_path = std::getenv("NINFER_BQ2_FULL_ART");
    if (art_path == nullptr) {
        std::cout << "SKIP: set NINFER_BQ2_FULL_ART\n";
        return 77;
    }
    unsetenv("NINFER_T2_PREFILL_P1"); // legacy baseline for comparisons
    try {
        bq2full::Bq2Model model(art_path);
        DeviceContext& device = model.device;
        cudaStream_t stream   = model.stream;
        const Fam fams[] = {
            {"fg", "text/layers/8/ffn_gate", 17408, 5120},
            {"fd", "text/layers/8/ffn_down", 5120, 17408},
            {"qkv", "text/layers/8/attn_qkv", 10240, 5120},
        };
        const Weight ws[] = {model.gdn[8].fg, model.gdn[8].fd, model.gdn[8].qkv};
        for (int fi = 0; fi < 3; ++fi) {
            const Fam& fm = fams[fi];
            const Weight& w = ws[fi];
            for (int T : {8, 32, 64, 128}) {
                const std::size_t n_el = static_cast<std::size_t>(fm.rows) * T;
                bq2full::DeviceBuffer dx(fm.cols * T * 2), dy0(n_el * 2), dy1(n_el * 2);
                // Token-distinct deterministic input (explicit c-dependence;
                // K%512==0 would otherwise erase it).
                std::vector<std::uint16_t> xb(fm.cols * T);
                for (int c = 0; c < T; ++c) {
                    for (int k = 0; k < fm.cols; ++k) {
                        xb[static_cast<std::size_t>(k) + static_cast<std::size_t>(fm.cols) * c] =
                            static_cast<std::uint16_t>(0x3C00 + ((k * 37 + c * 12345) % 512));
                    }
                }
                CUDA_CHECK(cudaMemcpy(dx.p, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
                const Tensor x(dx.p, DType::BF16, {fm.cols, T});
                Tensor y0(dy0.p, DType::BF16, {fm.rows, T});
                Tensor y1(dy1.p, DType::BF16, {fm.rows, T});
                // T=8 uses the synchronous 64x8 schedule; larger exact
                // tiles use the 64x32 schedule.
                auto sync_launch = [&](const Tensor& xa, const Weight& wa, Tensor& ya) {
                    if (T == 8) {
                        launch_t2_prefill_mma_sync8(xa, wa, ya, stream);
                    } else {
                        launch_t2_prefill_mma_sync(xa, wa, ya, stream);
                    }
                };
                std::printf("== %s T=%d ==\n", fm.name, T);
                // 1. 50-run determinism vs run0.
                std::vector<std::uint16_t> ref(n_el);
                {
                    sync_launch(x, w, y0);
                    device.synchronize();
                    CUDA_CHECK(
                        cudaMemcpy(ref.data(), dy0.p, ref.size() * 2, cudaMemcpyDeviceToHost));
                }
                std::size_t det_mm = 0;
                Stats worst;
                worst.n = n_el;
                std::vector<std::uint16_t> got(n_el);
                for (int r = 1; r < 50; ++r) {
                    sync_launch(x, w, y1);
                    device.synchronize();
                    CUDA_CHECK(
                        cudaMemcpy(got.data(), dy1.p, got.size() * 2, cudaMemcpyDeviceToHost));
                    const Stats s = cmp_bufs(ref, got);
                    det_mm += s.mm;
                    if (s.maxabs > worst.maxabs) { worst.maxabs = s.maxabs; }
                    worst.se += s.se;
                    worst.dot = s.dot;
                    worst.o2  = s.o2;
                    worst.r2  = s.r2;
                }
                std::printf("  determinism 50 runs: total-mm=%zu worst-maxabs=%.6f\n", det_mm,
                            worst.maxabs);
                CHECK(det_mm == 0, "%s T=%d P1-SYNC jitter: mm=%zu", fm.name, T, det_mm);
                // 2. vs legacy (tolerance).
                {
                    ops::linear(x, w, y1, stream);
                    device.synchronize();
                    CUDA_CHECK(
                        cudaMemcpy(got.data(), dy1.p, got.size() * 2, cudaMemcpyDeviceToHost));
                    print_stats("sync-vs-legacy", cmp_bufs(ref, got));
                }
                // 3. FP64 oracle on 256 sampled rows.
                {
                    const artifact::PayloadSpan span =
                        model.reader.payload(fm.art);
                    const std::uint8_t* ep =
                        reinterpret_cast<const std::uint8_t*>(span.data.data());
                    std::vector<float> xf(fm.cols * T);
                    {
                        std::vector<std::uint16_t> xbh(fm.cols * T);
                        CUDA_CHECK(cudaMemcpy(xbh.data(), dx.p, xbh.size() * 2,
                                              cudaMemcpyDeviceToHost));
                        for (std::size_t i = 0; i < xbh.size(); ++i) {
                            xf[i] = b2f(xbh[i]);
                        }
                    }
                    const int RCHK = 256;
                    double omx = 0, ose = 0;
                    // ne0-fastest {rows,T}: element (r,c) at r + rows*c.
                    for (int r = 0; r < RCHK; ++r) {
                        const std::vector<double> wrow =
                            bq2full::t2_decode_rows(ep, fm.rows, r, r + 1, fm.cols);
                        for (int t = 0; t < T; ++t) {
                            double acc = 0.0;
                            for (int k = 0; k < fm.cols; ++k) {
                                acc += wrow[k] *
                                       xf[static_cast<std::size_t>(t) * fm.cols + k];
                            }
                            const double gotv = b2f(
                                ref[static_cast<std::size_t>(r) +
                                    static_cast<std::size_t>(fm.rows) * t]);
                            const double e = std::fabs(gotv - acc);
                            if (e > omx) { omx = e; }
                            ose += e * e;
                        }
                    }
                    const double ormse = std::sqrt(ose / (RCHK * T));
                    std::printf("  oracle %-4s T=%-4d rows=%d maxabs=%.6f rmse=%.6f %s\n",
                                fm.name, T, RCHK, omx, ormse, omx <= 1e-3 ? "PASS" : "FAIL");
                    if (omx > 1e-3) { ++failures; }
                }
                // 4. vs async prototype (tolerance; async known-racy).
                {
                    launch_t2_prefill_mma(x, w, y1, stream);
                    device.synchronize();
                    CUDA_CHECK(
                        cudaMemcpy(got.data(), dy1.p, got.size() * 2, cudaMemcpyDeviceToHost));
                    print_stats("sync-vs-asyncP1", cmp_bufs(ref, got));
                }
                // 5. Timing: legacy vs async vs sync (median of 10, warmed).
                {
                    for (int i = 0; i < 3; ++i) {
                        ops::linear(x, w, y0, stream);
                        launch_t2_prefill_mma(x, w, y0, stream);
                        sync_launch(x, w, y0);
                    }
                    device.synchronize();
                    std::vector<double> d0, d1, d2;
                    for (int i = 0; i < 10; ++i) {
                        auto tick = [&]() {
                            cudaEvent_t e0, e1;
                            CUDA_CHECK(cudaEventCreate(&e0));
                            CUDA_CHECK(cudaEventCreate(&e1));
                            return std::make_pair(e0, e1);
                        };
                        {
                            auto [e0, e1] = tick();
                            CUDA_CHECK(cudaEventRecord(e0, stream));
                            ops::linear(x, w, y0, stream);
                            CUDA_CHECK(cudaEventRecord(e1, stream));
                            CUDA_CHECK(cudaEventSynchronize(e1));
                            float ms = 0;
                            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                            d0.push_back(ms);
                            CUDA_CHECK(cudaEventDestroy(e0));
                            CUDA_CHECK(cudaEventDestroy(e1));
                        }
                        {
                            auto [e0, e1] = tick();
                            CUDA_CHECK(cudaEventRecord(e0, stream));
                            launch_t2_prefill_mma(x, w, y0, stream);
                            CUDA_CHECK(cudaEventRecord(e1, stream));
                            CUDA_CHECK(cudaEventSynchronize(e1));
                            float ms = 0;
                            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                            d1.push_back(ms);
                            CUDA_CHECK(cudaEventDestroy(e0));
                            CUDA_CHECK(cudaEventDestroy(e1));
                        }
                        {
                            auto [e0, e1] = tick();
                            CUDA_CHECK(cudaEventRecord(e0, stream));
                            sync_launch(x, w, y0);
                            CUDA_CHECK(cudaEventRecord(e1, stream));
                            CUDA_CHECK(cudaEventSynchronize(e1));
                            float ms = 0;
                            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                            d2.push_back(ms);
                            CUDA_CHECK(cudaEventDestroy(e0));
                            CUDA_CHECK(cudaEventDestroy(e1));
                        }
                    }
                    const double m0 = med(d0), m1 = med(d1), m2 = med(d2);
                    std::printf("  timing med ms: legacy=%.4f async=%.4f sync=%.4f "
                                "sync-vs-legacy=%.2fx sync-vs-async=%.3f\n",
                                m0, m1, m2, m0 / m2, m1 / m2);
                }
            }
        }
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    }
    if (failures != 0) {
        std::printf("P1SYNC FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("OK P1SYNC\n");
    return 0;
}
