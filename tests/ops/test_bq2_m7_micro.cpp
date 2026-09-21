// M7 measurement experiments 1-4 (device-side, WDDM-safe cudaEvents).
// T2 GEMV per family x T in {1,8,32} with real artifact weights (ms +
// effective weight GB/s), FWHT per width, GDN recurrent core, null-launch /
// malloc-free / sync microbench, cold-cache vs warm prefill traffic probe.
// Env: NINFER_BQ2_FULL_ART (bonsai2_full_text.ninfer).
#include "ops/bq2_model_common.h"

#include "ninfer/ops/linear.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace bq2full;

struct M7Timer {
    cudaEvent_t a, b;
    M7Timer() {
        CUDA_CHECK(cudaEventCreate(&a));
        CUDA_CHECK(cudaEventCreate(&b));
    }
    ~M7Timer() {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    void start(cudaStream_t s) { CUDA_CHECK(cudaEventRecord(a, s)); }
    double stop_ms(cudaStream_t s) {
        CUDA_CHECK(cudaEventRecord(b, s));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        return ms;
    }
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
    try {
        Bq2Model model(art_path);
        DeviceContext& device = model.device;
        cudaStream_t stream   = model.stream;
        bool ok               = true;

        // ---- exp 2a: null-launch latency ----
        {
            M7Timer t;
            std::vector<double> ds;
            for (int i = 0; i < 200; ++i) {
                t.start(stream);
                // Zero-size memset as near-null launch proxy (no-op kernel).
                CUDA_CHECK(cudaMemsetAsync(nullptr, 0, 0, stream));
                ds.push_back(t.stop_ms(stream));
            }
            std::printf("micro null-launch med=%.3fus\n", med(ds) * 1000.0);
        }
        // ---- exp 2b: DeviceBuffer malloc/free ----
        {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 200; ++i) {
                DeviceBuffer b(10240);
                (void)b;
            }
            device.synchronize();
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            std::printf("micro malloc-free-pair 10KB: %.3fus/pair over 200\n", ms * 1000.0 / 200);
        }
        // ---- exp 2c: stream sync latency ----
        {
            M7Timer t;
            std::vector<double> ds;
            for (int i = 0; i < 200; ++i) {
                t.start(stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                ds.push_back(t.stop_ms(stream));
            }
            std::printf("micro stream-sync med=%.3fus\n", med(ds) * 1000.0);
        }

        // ---- exp 1+3: T2 GEMV per family (real weights), T in {1,8,32} ----
        struct Fam {
            const char* name;
            Weight w;
            int rows, cols;
        };
        const GdnLayerW& g8 = model.gdn[8];
        const FullLayerW& f7 = model.full[7];
        const Fam fams[] = {
            {"qkv", g8.qkv, 10240, 5120}, {"gate", g8.gate, 6144, 5120},
            {"so", g8.so, 5120, 6144},    {"fg", g8.fg, 17408, 5120},
            {"fu", g8.fu, 17408, 5120},   {"fd", g8.fd, 5120, 17408},
            {"attn_q", f7.q, 12288, 5120}, {"attn_k", f7.k, 1024, 5120},
            {"attn_v", f7.v, 1024, 5120}, {"attn_wo", f7.wo, 5120, 6144},
            {"head", model.head_t2, 248320, 5120},
        };
        for (int T : {1, 8, 32}) {
            for (const auto& fm : fams) {
                DeviceBuffer dx(fm.cols * T * 2), dy(fm.rows * T * 2);
                // Warm random-ish input (timing only): ascending pattern.
                {
                    std::vector<std::uint16_t> bits(fm.cols * T);
                    for (std::size_t i = 0; i < bits.size(); ++i) {
                        bits[i] = (std::uint16_t)(0x3C00 + (i % 64));
                    }
                    dx.copy_from_host(bits.data(), bits.size() * 2);
                }
                const Tensor x(dx.p, DType::BF16, {fm.cols, T});
                Tensor y(dy.p, DType::BF16, {fm.rows, T});
                M7Timer t;
                std::vector<double> ds;
                for (int i = 0; i < 5; ++i) { ops::linear(x, fm.w, y, stream); }
                device.synchronize();
                for (int i = 0; i < 20; ++i) {
                    t.start(stream);
                    ops::linear(x, fm.w, y, stream);
                    ds.push_back(t.stop_ms(stream));
                }
                const double ms = med(ds);
                const double wbytes =
                    (double)fm.rows * (fm.cols / 128) * 34.0;
                std::printf("gemv %-8s R=%-6d K=%-5d T=%-3d med=%.4fms wGB/s=%.1f\n", fm.name,
                            fm.rows, fm.cols, T, ms, wbytes / ms * 1e-6);
            }
        }

        // ---- FWHT per width ----
        for (int K : {5120, 6144, 17408}) {
            for (int T : {1, 8, 32}) {
                DeviceBuffer dx(K * T * 2), dy(K * T * 2), ds(K * 4);
                const Tensor x(dx.p, DType::BF16, {K, T});
                Tensor y(dy.p, DType::BF16, {K, T});
                M7Timer t;
                std::vector<double> dsv;
                for (int i = 0; i < 5; ++i) {
                    ops::t2_fwht_sign(x, static_cast<const float*>(ds.p), y, stream);
                }
                device.synchronize();
                for (int i = 0; i < 20; ++i) {
                    t.start(stream);
                    ops::t2_fwht_sign(x, static_cast<const float*>(ds.p), y, stream);
                    dsv.push_back(t.stop_ms(stream));
                }
                std::printf("fwht K=%-6d T=%-3d med=%.4fms\n", K, T, med(dsv));
            }
        }

        // ---- GDN recurrent core T=1 (random inputs, timing only) ----
        {
            DeviceBuffer dq(128 * 16 * 2), dk(128 * 16 * 2), dv(128 * 48 * 2);
            DeviceBuffer dg(48 * 4), db(48 * 4), dsi(128 * 128 * 48 * 4);
            DeviceBuffer dso(128 * 128 * 48 * 4), dout(128 * 48 * 2);
            dsi.fill(0);
            DeviceArena ws(std::max<std::size_t>(
                ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, 1, 1), 256));
            const Tensor q(dq.p, DType::BF16, {128, 16, 1});
            const Tensor k(dk.p, DType::BF16, {128, 16, 1});
            const Tensor v(dv.p, DType::BF16, {128, 48, 1});
            const Tensor g(dg.p, DType::FP32, {48, 1});
            const Tensor beta(db.p, DType::FP32, {48, 1});
            const Tensor si(dsi.p, DType::FP32, {128, 128, 48});
            Tensor so(dso.p, DType::FP32, {128, 128, 48});
            Tensor o(dout.p, DType::BF16, {128, 48, 1});
            M7Timer t;
            std::vector<double> ds;
            for (int i = 0; i < 5; ++i) {
                ops::gated_delta_net(q, k, v, g, beta, 0.08838834764831845f, true, ws, si, so,
                                     o, stream);
            }
            device.synchronize();
            for (int i = 0; i < 20; ++i) {
                t.start(stream);
                ops::gated_delta_net(q, k, v, g, beta, 0.08838834764831845f, true, ws, si, so,
                                     o, stream);
                ds.push_back(t.stop_ms(stream));
            }
            std::printf("gdn-core T=1 med=%.4fms\n", med(ds));
        }

        // ---- exp 4: cold-cache vs warm (fg 17408x5120, T=32) ----
        {
            const int T = 32, rows = 17408, cols = 5120;
            DeviceBuffer dx(cols * T * 2), dy(rows * T * 2);
            const Tensor x(dx.p, DType::BF16, {cols, T});
            Tensor y(dy.p, DType::BF16, {rows, T});
            // Warm.
            M7Timer t;
            std::vector<double> dw;
            for (int i = 0; i < 5; ++i) { ops::linear(x, model.gdn[8].fg, y, stream); }
            device.synchronize();
            for (int i = 0; i < 10; ++i) {
                t.start(stream);
                ops::linear(x, model.gdn[8].fg, y, stream);
                dw.push_back(t.stop_ms(stream));
            }
            // Cold: stream a 1GB dummy to evict L2 (~32-48MB), then time
            // immediately. (Weights stream from VRAM regardless; the cold
            // question is L2 residency of weights/activations.)
            DeviceBuffer flush(1ULL * 1024 * 1024 * 1024ULL);
            flush.fill(0x5a);
            device.synchronize();
            std::vector<double> dc;
            for (int i = 0; i < 3; ++i) {
                flush.fill((int)i);
                device.synchronize();
                t.start(stream);
                ops::linear(x, model.gdn[8].fg, y, stream);
                dc.push_back(t.stop_ms(stream));
            }
            std::printf("cold-cache fg T=32: warm=%.4fms cold=%.4fms ratio=%.2f\n", med(dw),
                        med(dc), med(dc) / med(dw));
        }

        std::printf("%s M7_MICRO\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    }
}
