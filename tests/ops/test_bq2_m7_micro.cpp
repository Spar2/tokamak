// M7 measurement experiments 1-4 (device-side, WDDM-safe cudaEvents).
// T2 GEMV per family x T in {1,8,32} with real artifact weights (ms +
// effective weight GB/s), FWHT per width, GDN recurrent core, null-launch /
// malloc-free / sync microbench, cold-cache vs warm prefill traffic probe.
// Env: NINFER_BQ2_FULL_ART (bonsai2_full_text.ninfer).
#include "ops/bq2_model_common.h"

#include "ninfer/ops/linear.h"
#include "ops/linear/t2/t2_launch.h"

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
                // T1-A side-by-side (T=1 only): same input, bit-compare + time.
                if (T == 1) {
                    DeviceBuffer dy2(fm.rows * 2);
                    Tensor y2(dy2.p, DType::BF16, {fm.rows, 1});
                    std::vector<double> da;
                    for (int i = 0; i < 5; ++i) {
                        ops::detail::launch_t2_gemv_t1a(x, fm.w, y2, stream);
                    }
                    device.synchronize();
                    for (int i = 0; i < 20; ++i) {
                        t.start(stream);
                        ops::detail::launch_t2_gemv_t1a(x, fm.w, y2, stream);
                        da.push_back(t.stop_ms(stream));
                    }
                    const double ma = med(da);
                    std::vector<std::uint16_t> b0(fm.rows), b1(fm.rows);
                    CUDA_CHECK(cudaMemcpy(b0.data(), dy.p, b0.size() * 2,
                                          cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(b1.data(), dy2.p, b1.size() * 2,
                                          cudaMemcpyDeviceToHost));
                    std::size_t mism = 0;
                    for (std::size_t i = 0; i < b0.size(); ++i) {
                        if (b0[i] != b1[i]) { ++mism; }
                    }
                    std::printf("  t1a  %-8s med=%.4fms speedup=%.3f biteq=%s\n", fm.name, ma,
                                ms / ma, mism == 0 ? "YES" : "NO");
                    if (mism != 0) { ok = false; }
                }
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

        // ---- M7-P1: tiled BF16-MMA prototype vs production ----
        // Representative shapes x T in {8,32,128}. Production baseline is
        // the CURRENT dispatch (row-persistent/chunked), never T1-A.
        {
            struct PF {
                const char* name;
                Weight w;
                const char* art;
                int rows, cols;
            };
            const PF pfs[] = {
                {"fg", model.gdn[8].fg, "text/layers/8/ffn_gate", 17408, 5120},
                {"fd", model.gdn[8].fd, "text/layers/8/ffn_down", 5120, 17408},
                {"qkv", model.gdn[8].qkv, "text/layers/8/attn_qkv", 10240, 5120},
            };
            for (int T : {8, 32, 128}) {
                for (const auto& pf : pfs) {
                    DeviceBuffer dx(pf.cols * T * 2), dy0(pf.rows * T * 2),
                        dy1(pf.rows * T * 2);
                    {
                        std::vector<std::uint16_t> bits(pf.cols * T);
                        for (std::size_t i = 0; i < bits.size(); ++i) {
                            bits[i] = (std::uint16_t)(0x3C00 + (i % 64));
                        }
                        dx.copy_from_host(bits.data(), bits.size() * 2);
                    }
                    const Tensor x(dx.p, DType::BF16, {pf.cols, T});
                    Tensor y0(dy0.p, DType::BF16, {pf.rows, T});
                    Tensor y1(dy1.p, DType::BF16, {pf.rows, T});
                    M7Timer t;
                    std::vector<double> d0, d1;
                    for (int i = 0; i < 3; ++i) {
                        ops::linear(x, pf.w, y0, stream);
                        ops::detail::launch_t2_prefill_mma(x, pf.w, y1, stream);
                    }
                    device.synchronize();
                    for (int i = 0; i < 10; ++i) {
                        t.start(stream);
                        ops::linear(x, pf.w, y0, stream);
                        d0.push_back(t.stop_ms(stream));
                    }
                    for (int i = 0; i < 10; ++i) {
                        t.start(stream);
                        ops::detail::launch_t2_prefill_mma(x, pf.w, y1, stream);
                        d1.push_back(t.stop_ms(stream));
                    }
                    const double m0 = med(d0), m1 = med(d1);
                    std::vector<std::uint16_t> b0(pf.rows * T), b1(pf.rows * T);
                    CUDA_CHECK(cudaMemcpy(b0.data(), dy0.p, b0.size() * 2,
                                          cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(b1.data(), dy1.p, b1.size() * 2,
                                          cudaMemcpyDeviceToHost));
                    double mx = 0.0, se = 0.0, r2 = 0.0, dot = 0.0, o2 = 0.0;
                    for (std::size_t i = 0; i < b0.size(); ++i) {
                        const double a = bf16_to_f32(b0[i]);
                        const double b = bf16_to_f32(b1[i]);
                        const double e = std::fabs(a - b);
                        if (e > mx) { mx = e; }
                        se += e * e;
                        r2 += b * b;
                        dot += a * b;
                        o2 += a * a;
                    }
                    const double rmse = std::sqrt(se / b0.size());
                    const double cos =
                        (o2 > 0.0 && r2 > 0.0) ? dot / std::sqrt(o2 * r2) : 0.0;
                    // Traffic accounting (prototype): codes+scales staged
                    // once per K-slab; activations staged once per CTA.
                    const double wbytes = (double)pf.rows * (pf.cols / 128) * 34.0;
                    const double abytes = (double)pf.cols * T * 2.0;
                    const double obytes = (double)pf.rows * T * 2.0;
                    std::printf("p1 %-4s T=%-4d base=%.4fms proto=%.4fms speedup=%.3f "
                                "delta-maxabs=%.6f delta-rmse=%.6f cos=%.8f "
                                "wMB=%.1f aMB=%.1f oMB=%.1f\n",
                                pf.name, T, m0, m1, m0 / m1, mx, rmse, cos,
                                wbytes / 1e6, abytes / 1e6, obytes / 1e6);
                    // FP64 oracle on subsampled rows (T<=32; T=128 fg only).
                    if (T <= 32 || (T == 128 && pf.rows == 17408)) {
                        const int RCHK = 256;
                        const artifact::PayloadSpan span = model.reader.payload(pf.art);
                        const std::uint8_t* ep =
                            reinterpret_cast<const std::uint8_t*>(span.data.data());
                        std::vector<float> xf(pf.cols * T);
                        {
                            std::vector<std::uint16_t> xb(pf.cols * T);
                            CUDA_CHECK(cudaMemcpy(xb.data(), dx.p, xb.size() * 2,
                                                  cudaMemcpyDeviceToHost));
                            for (std::size_t i = 0; i < xb.size(); ++i) {
                                xf[i] = bf16_to_f32(xb[i]);
                            }
                        }
                        double omx = 0.0, ose = 0.0;
                        for (int r = 0; r < RCHK; ++r) {
                            const std::vector<double> wrow =
                                t2_decode_rows(ep, pf.rows, r, r + 1, pf.cols);
                            for (int t = 0; t < T; ++t) {
                                double acc = 0.0;
                                for (int k = 0; k < pf.cols; ++k) {
                                    acc += wrow[k] *
                                           xf[static_cast<std::size_t>(t) * pf.cols + k];
                                }
                                const double got = bf16_to_f32(
                                    b1[static_cast<std::size_t>(t) * pf.rows + r]);
                                const double e = std::fabs(got - acc);
                                if (e > omx) { omx = e; }
                                ose += e * e;
                            }
                        }
                        const double ormse = std::sqrt(ose / (RCHK * T));
                        std::printf("  oracle %-4s T=%-4d rows=%d maxabs=%.6f rmse=%.6f %s\n",
                                    pf.name, T, RCHK, omx, ormse,
                                    omx <= 1e-3 ? "PASS" : "FAIL");
                        if (omx > 1e-3) { ok = false; }
                    }
                }
            }
        }

        std::printf("%s M7_MICRO\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    }
}
