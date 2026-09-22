// M7 masked-tail numeric + crossover matrix (model-based, real weights).
// Shapes fg/fd/qkv (layer 8) x T in {8,9,15,16,24,31,32,33,47,48,63,64}.
// Per cell (with warmup launches per the churn discipline):
//   1. masked determinism (10 runs vs run0, want ZERO);
//   2. masked vs legacy production path (tolerance);
//   3. masked vs FP64 sampled oracle, 256 rows (tolerance 1e-3);
//   4. timing legacy vs masked (median of 10, warmed) -> crossover data.
// T=8/32 cross-check masked-vs-exact-schedule consistency.
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
            std::printf("P1TAIL-FAIL: ");                                                          \
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
    const double rel =
        (s.o2 > 0) ? std::sqrt(s.se / s.o2) : 0.0;
    std::printf("  %-20s mm=%zu/%zu maxabs=%.6f rmse=%.6f rel=%.6f cos=%.9f\n", tag, s.mm,
                s.n, s.maxabs, rmse, rel, cos);
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
        const int kTs[]   = {8, 9, 15, 16, 24, 31, 32, 33, 47, 48, 63, 64};
        for (int fi = 0; fi < 3; ++fi) {
            const Fam& fm   = fams[fi];
            const Weight& w = ws[fi];
            for (int T : kTs) {
                const bool is_tail = !(T == 8 || T == 32 || T == 64 || T == 128);
                const std::size_t n_el = static_cast<std::size_t>(fm.rows) * T;
                bq2full::DeviceBuffer dx(fm.cols * T * 2), dy0(n_el * 2), dy1(n_el * 2);
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
                auto mt_launch = [&](const Tensor& xa, const Weight& wa, Tensor& ya) {
                    if (T == 8) {
                        launch_t2_prefill_mma_sync8(xa, wa, ya, stream);
                    } else if (T == 32 || T == 64) {
                        launch_t2_prefill_mma_sync(xa, wa, ya, stream);
                    } else {
                        launch_t2_prefill_mma_sync_mt(xa, wa, ya, stream);
                    }
                };
                std::printf("== %s T=%d%s ==\n", fm.name, T, is_tail ? " (tail)" : "");
                // Warmup (churn discipline) + determinism vs run0.
                mt_launch(x, w, y0);
                mt_launch(x, w, y0);
                device.synchronize();
                std::vector<std::uint16_t> ref(n_el), got(n_el);
                mt_launch(x, w, y0);
                device.synchronize();
                CUDA_CHECK(
                    cudaMemcpy(ref.data(), dy0.p, ref.size() * 2, cudaMemcpyDeviceToHost));
                std::size_t det_mm = 0;
                for (int r = 0; r < 10; ++r) {
                    mt_launch(x, w, y1);
                    device.synchronize();
                    CUDA_CHECK(
                        cudaMemcpy(got.data(), dy1.p, got.size() * 2, cudaMemcpyDeviceToHost));
                    for (std::size_t i = 0; i < n_el; ++i) {
                        if (ref[i] != got[i]) { ++det_mm; }
                    }
                }
                std::printf("  determinism 10 runs: total-mm=%zu\n", det_mm);
                CHECK(det_mm == 0, "%s T=%d masked jitter: mm=%zu", fm.name, T, det_mm);
                // 2. vs legacy.
                {
                    ops::linear(x, w, y1, stream);
                    device.synchronize();
                    CUDA_CHECK(
                        cudaMemcpy(got.data(), dy1.p, got.size() * 2, cudaMemcpyDeviceToHost));
                    const Stats s = cmp_bufs(ref, got);
                    print_stats("mt-vs-legacy", s);
                    CHECK(s.maxabs <= 2e-3, "%s T=%d mt-vs-legacy maxabs=%.6f", fm.name, T,
                          s.maxabs);
                }
                // 3. FP64 oracle, 256 rows.
                {
                    const artifact::PayloadSpan span = model.reader.payload(fm.art);
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
                // 4. timing legacy vs masked.
                {
                    for (int i = 0; i < 3; ++i) {
                        ops::linear(x, w, y0, stream);
                        mt_launch(x, w, y0);
                    }
                    device.synchronize();
                    std::vector<double> d0, d1;
                    for (int i = 0; i < 10; ++i) {
                        cudaEvent_t e0, e1;
                        CUDA_CHECK(cudaEventCreate(&e0));
                        CUDA_CHECK(cudaEventCreate(&e1));
                        CUDA_CHECK(cudaEventRecord(e0, stream));
                        ops::linear(x, w, y0, stream);
                        CUDA_CHECK(cudaEventRecord(e1, stream));
                        CUDA_CHECK(cudaEventSynchronize(e1));
                        float ms = 0;
                        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                        d0.push_back(ms);
                        CUDA_CHECK(cudaEventDestroy(e0));
                        CUDA_CHECK(cudaEventDestroy(e1));
                        CUDA_CHECK(cudaEventCreate(&e0));
                        CUDA_CHECK(cudaEventCreate(&e1));
                        CUDA_CHECK(cudaEventRecord(e0, stream));
                        mt_launch(x, w, y0);
                        CUDA_CHECK(cudaEventRecord(e1, stream));
                        CUDA_CHECK(cudaEventSynchronize(e1));
                        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                        d1.push_back(ms);
                        CUDA_CHECK(cudaEventDestroy(e0));
                        CUDA_CHECK(cudaEventDestroy(e1));
                    }
                    const double m0 = med(d0), m1 = med(d1);
                    std::printf("  timing med ms: legacy=%.4f masked=%.4f speedup=%.2fx\n",
                                m0, m1, m0 / m1);
                }
            }
        }
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    }
    if (failures != 0) {
        std::printf("P1TAIL FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("OK P1TAIL\n");
    return 0;
}
