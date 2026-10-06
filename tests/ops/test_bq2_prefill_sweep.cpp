// Full-model arbitrary-T prefill timing + smoke/soak runner (no Prism refs).
// Usage: prefill_sweep <T> [measured_runs=2] [class 0..3 =0] [seed=1]
// Classes: 0=tiled validation ids (prose-like); 1=uniform PRNG ids;
// 2=repeated token with rare swaps; 3=sequential id walk. All ids stay in
// the valid sampling domain [0,248077). One discarded warmup prefill
// (churn discipline), then measured runs. Reports per-run wall ms, CUDA
// event device ms, GPU telemetry, output hash, finiteness, and run-to-run
// self maxabs. Gate via NINFER_T2_PREFILL_P1 env (caller sets).
#include "ops/bq2_model_common.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
using namespace ninfer;
using namespace bq2full;

std::uint64_t rng_state = 0;
std::uint64_t rng_next() {
    std::uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

std::uint64_t fnv1a(const float* data, std::size_t n) {
    std::uint64_t h = 1469598103934665603ULL;
    const auto* p   = reinterpret_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n * 4; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// One nvidia-smi telemetry sample; never fails the test.
std::string gpu_telemetry() {
    char buf[256] = {0};
    FILE* f = popen("nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory,"
                    "power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,"
                    "noheader,nounits 2>/dev/null",
                    "r");
    if (f == nullptr) { return std::string("N/A"); }
    std::size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    pclose(f);
    if (n == 0) { return std::string("N/A"); }
    buf[n] = '\0';
    for (std::size_t i = 0; i < n; ++i) {
        if (buf[i] == '\n' || buf[i] == '\r') { buf[i] = ' '; }
    }
    return std::string(buf);
}

} // namespace

static double run_prefill(Bq2Model& model, Bq2SeqState& state, const std::vector<std::int32_t>& ids,
                          std::vector<float>& last_tok_out, double& dev_ms_out) {
    DeviceContext& device = model.device;
    cudaStream_t stream   = model.stream;
    const int T           = static_cast<int>(ids.size());
    state.reset();
    DeviceBuffer d_x(5120 * T * 2);
    embed_lookup(model, ids.data(), T, d_x.p, stream);
    device.synchronize();
    DeviceBuffer d_pos(T * 4), d_pos3(T * 3 * 4);
    {
        std::vector<std::int32_t> p1(T), p3(T * 3);
        for (int t = 0; t < T; ++t) {
            p1[t] = t;
            p3[t] = p3[T + t] = p3[2 * T + t] = t;
        }
        upload_i32(p1.data(), d_pos.p, T);
        upload_i32(p3.data(), d_pos3.p, T * 3);
    }
    const Tensor positions(d_pos.p, DType::I32, {T});
    const Tensor rope_positions(d_pos3.p, DType::I32, {T, 3});
    DeviceBuffer d_cur(5120 * T * 2), d_nxt(5120 * T * 2);
    CUDA_CHECK(cudaMemcpy(d_cur.p, d_x.p, 5120 * T * 2, cudaMemcpyDeviceToDevice));
    DeviceBuffer d_rows1(4);
    {
        const std::int32_t zero = 0;
        upload_i32(&zero, d_rows1.p, 1);
    }
    const Tensor table_rows(d_rows1.p, DType::I32, {1});
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    const auto t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(e0, stream));
    for (int il = 0; il < 64; ++il) {
        Tensor xc(d_cur.p, DType::BF16, {5120, T});
        Tensor xo(d_nxt.p, DType::BF16, {5120, T});
        if (is_full_layer(il)) {
            run_full_block(model, il, xc, xo, state.kv_view(il), positions, rope_positions,
                           table_rows, static_cast<std::uint32_t>(T), stream);
        } else {
            Tensor conv(state.gdn[il].conv.p, DType::BF16, {10240, 3});
            Tensor ssm(state.gdn[il].ssm.p, DType::FP32, {128, 128, 48});
            run_gdn_block(model, il, xc, xo, conv, ssm, stream);
        }
        device.synchronize();
        std::swap(d_cur, d_nxt);
    }
    DeviceBuffer d_n(5120 * T * 2);
    {
        const Tensor h(d_cur.p, DType::BF16, {5120, T});
        Tensor o(d_n.p, DType::BF16, {5120, T});
        ops::rmsnorm(h, model.final_norm, 1e-6f, false, o, stream);
    }
    CUDA_CHECK(cudaEventRecord(e1, stream));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float dev_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&dev_ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    device.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const std::vector<float> got = d2h_bf16(device, d_n.p, 5120 * T);
    last_tok_out.resize(5120);
    for (int i = 0; i < 5120; ++i) {
        last_tok_out[i] = got[static_cast<std::size_t>(T - 1) * 5120 + i];
    }
    dev_ms_out = dev_ms;
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
}

int main(int argc, char** argv) {
    const int T     = argc > 1 ? std::atoi(argv[1]) : 0;
    const int reps  = argc > 2 ? std::atoi(argv[2]) : 2;
    const int klass = argc > 3 ? std::atoi(argv[3]) : 0;
    rng_state       = static_cast<std::uint64_t>(argc > 4 ? std::atoll(argv[4]) : 1);
    const char* art_path = std::getenv("NINFER_BQ2_FULL_ART");
    if (T < 8 || T > 127 || klass < 0 || klass > 3 || art_path == nullptr) {
        std::cout << "usage: prefill_sweep <T 8..127> [reps] [class 0..3] [seed]; needs "
                     "NINFER_BQ2_FULL_ART\n";
        return 77;
    }
    const char* gate = std::getenv("NINFER_T2_PREFILL_P1");
    try {
        Bq2Model model(art_path);
        Bq2SeqState state(8);
        const std::vector<std::int32_t> ids32 = {
            760,   3841,  13477, 37550, 33075, 888,   279,   15217, 5388,  3043,  279,
            14367, 5883,  506,   41564, 1345,  19053, 7534,  13,    561,   3841,  13477,
            37550, 33075, 888,   279,   15217, 5388,  3043,  279,   14367, 5883};
        static constexpr std::int32_t kValidRows = 248077;
        std::vector<std::int32_t> ids;
        ids.reserve(T);
        if (klass == 0) {
            while (static_cast<int>(ids.size()) < T) {
                ids.insert(ids.end(), ids32.begin(), ids32.end());
            }
        } else if (klass == 1) {
            for (int t = 0; t < T; ++t) {
                ids.push_back(static_cast<std::int32_t>(rng_next() % kValidRows));
            }
        } else if (klass == 2) {
            const std::int32_t base = static_cast<std::int32_t>(15000 + rng_next() % 50000);
            const std::int32_t alt  = static_cast<std::int32_t>(rng_next() % kValidRows);
            for (int t = 0; t < T; ++t) {
                ids.push_back((rng_next() % 8 == 0) ? alt : base);
            }
        } else {
            const std::int32_t start = static_cast<std::int32_t>(rng_next() % kValidRows);
            const std::int32_t stride = 1 + static_cast<std::int32_t>(rng_next() % 97);
            for (int t = 0; t < T; ++t) {
                ids.push_back(static_cast<std::int32_t>((start + (std::int64_t)t * stride) %
                                                        kValidRows));
            }
        }
        ids.resize(T);
        std::vector<float> discard, prev;
        double dev_discard = 0;
        { // discarded warmup prefill (churn discipline)
            std::vector<float> tmp;
            run_prefill(model, state, ids, tmp, dev_discard);
        }
        bool ok = true;
        std::printf("SWEEP T=%d class=%d gate=%s", T, klass,
                    (gate != nullptr && gate[0] == '1') ? "ON" : "OFF");
        for (int r = 0; r < reps; ++r) {
            std::vector<float> out;
            double dev_ms   = 0;
            const double ms = run_prefill(model, state, ids, out, dev_ms);
            bool fin        = true;
            for (float f : out) {
                if (!std::isfinite(f)) { fin = false; break; }
            }
            const std::uint64_t h = fnv1a(out.data(), out.size());
            double mx             = 0;
            if (r > 0) {
                for (std::size_t i = 0; i < out.size(); ++i) {
                    const double e = std::fabs((double)out[i] - (double)prev[i]);
                    if (e > mx) { mx = e; }
                }
            }
            prev = out;
            const std::string tel = gpu_telemetry();
            std::printf(" run%d wall=%.1fms dev=%.1fms hash=%016llx%s", r, ms, dev_ms,
                        (unsigned long long)h, fin ? "" : "[NONFINITE]");
            if (!fin || (r > 0 && mx > 1e-2)) { ok = false; }
            if (r > 0) { std::printf("[selfmaxabs=%.6f]", mx); }
            std::printf("[tel:%s]", tel.c_str());
        }
        std::printf(" %s\n", ok ? "SWEEP-OK" : "SWEEP-FAIL");
        std::printf(ok ? "OK PREFILL_SWEEP\n" : "FAIL PREFILL_SWEEP\n");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    }
}
