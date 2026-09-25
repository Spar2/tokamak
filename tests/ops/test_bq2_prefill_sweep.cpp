// Full-model arbitrary-T prefill timing + smoke runner (no Prism refs).
// Usage: prefill_sweep <T> [measured_runs=2]
// Fixed token ids (T32 prefix tiled); one discarded warmup prefill (churn
// discipline), then measured runs. Reports per-run ms, finiteness, and
// run-to-run self maxabs. Gate via NINFER_T2_PREFILL_P1 env (caller sets).
#include "ops/bq2_model_common.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
using namespace ninfer;
using namespace bq2full;
} // namespace

static double run_prefill(Bq2Model& model, Bq2SeqState& state, const std::vector<std::int32_t>& ids,
                          std::vector<float>& last_tok_out) {
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
    const auto t0 = std::chrono::steady_clock::now();
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
    device.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const std::vector<float> got = d2h_bf16(device, d_n.p, 5120 * T);
    last_tok_out.resize(5120);
    for (int i = 0; i < 5120; ++i) {
        last_tok_out[i] = got[static_cast<std::size_t>(T - 1) * 5120 + i];
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
}

int main(int argc, char** argv) {
    const int T    = argc > 1 ? std::atoi(argv[1]) : 0;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 2;
    const char* art_path = std::getenv("NINFER_BQ2_FULL_ART");
    if (T < 8 || T > 127 || art_path == nullptr) {
        std::cout << "usage: prefill_sweep <T 8..127> [reps]; needs NINFER_BQ2_FULL_ART\n";
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
        std::vector<std::int32_t> ids;
        while (static_cast<int>(ids.size()) < T) {
            ids.insert(ids.end(), ids32.begin(), ids32.end());
        }
        ids.resize(T);
        std::vector<float> discard, prev;
        { // discarded warmup prefill (churn discipline)
            std::vector<float> tmp;
            run_prefill(model, state, ids, tmp);
        }
        bool ok = true;
        std::printf("SWEEP T=%d gate=%s", T, (gate != nullptr && gate[0] == '1') ? "ON" : "OFF");
        for (int r = 0; r < reps; ++r) {
            std::vector<float> out;
            const double ms = run_prefill(model, state, ids, out);
            bool fin        = true;
            for (float f : out) {
                if (!std::isfinite(f)) { fin = false; break; }
            }
            double mx = 0;
            if (r > 0) {
                for (std::size_t i = 0; i < out.size(); ++i) {
                    const double e = std::fabs((double)out[i] - (double)prev[i]);
                    if (e > mx) { mx = e; }
                }
            }
            prev = out;
            std::printf(" run%d=%.1fms%s", r, ms, fin ? "" : "[NONFINITE]");
            if (!fin || (r > 0 && mx > 1e-2)) { ok = false; }
            if (r > 0) { std::printf("[selfmaxabs=%.6f]", mx); }
        }
        std::printf(" %s\n", ok ? "SWEEP-OK" : "SWEEP-FAIL");
        std::printf(ok ? "OK PREFILL_SWEEP\n" : "FAIL PREFILL_SWEEP\n");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    }
}
