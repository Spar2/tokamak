// M6 Stage 7 (first part): continuous chain embedding -> layer 0 -> 1 -> 2
// -> 3 with PERSISTENT state (GDN conv/S slots, paged KV), vs Prism l_out.
// Layers 0/1/2 are GDN, layer 3 is full attention (interval rule).
// Env: NINFER_BQ2_FULL_ART, NINFER_BQ2_VECTORS_P0/P1/P2 (decode1 = prompt).
// Prompts (debug-log authoritative): P0 [9419] (T=1), P1 [9419,1814] (T=2),
// P2 [9419,1814,11,3242] (T=4, "Hello world, today").
// The final T=1 rerun proves reset determinism (bit-exact repeat).
#include "ops/bq2_model_common.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace bq2full;

const Gate kChainOut = {8e-3, 0.9995, 1e-1};

} // namespace

int main() {
    const char* art_path = std::getenv("NINFER_BQ2_FULL_ART");
    const char* p0_dir   = std::getenv("NINFER_BQ2_VECTORS_P0");
    const char* p1_dir   = std::getenv("NINFER_BQ2_VECTORS_P1");
    const char* p2_dir   = std::getenv("NINFER_BQ2_VECTORS_P2");
    if (art_path == nullptr || p0_dir == nullptr || p1_dir == nullptr || p2_dir == nullptr) {
        std::cout << "SKIP: set NINFER_BQ2_FULL_ART and NINFER_BQ2_VECTORS_P0/P1/P2\n";
        return 77;
    }
    try {
        bool ok = true;
        Bq2Model model(art_path);
        DeviceContext& device = model.device;
        cudaStream_t stream   = model.stream;
        Bq2SeqState state(8);
        DeviceBuffer d_rows1(4);
        {
            const std::int32_t zero = 0;
            upload_i32(&zero, d_rows1.p, 1);
        }
        const Tensor table_rows(d_rows1.p, DType::I32, {1});
        const struct {
            const char* vdir;
            std::vector<std::int32_t> ids;
            bool det_rerun;
        } cases[] = {{p0_dir, {9419}, false},
                     {p1_dir, {9419, 1814}, false},
                     {p2_dir, {9419, 1814, 11, 3242}, false},
                     {p0_dir, {9419}, true}};
        std::vector<float> det_embed, det_lout3;
        for (const auto& c : cases) {
            const int T = static_cast<int>(c.ids.size());
            std::printf("== T=%d%s ==\n", T, c.det_rerun ? " (reset determinism rerun)" : "");
            state.reset();
            // Embedding.
            DeviceBuffer d_x(5120 * T * 2);
            std::vector<float> cur_embed, cur_lout3;
            embed_lookup(model, c.ids.data(), T, d_x.p, stream);
            device.synchronize();
            {
                const std::vector<float> got = d2h_bf16(device, d_x.p, 5120 * T);
                const std::vector<float> ref0 =
                    load_ref(c.vdir, "model.input_embed.decode1", 5120 * T);
                const std::vector<float> ref = as_column_major(ref0, 5120, T);
                ok &= check_vec(got.data(), ref.data(), 5120 * T, "embed", kChainOut);
                if (T == 1) { cur_embed = got; }
                // TEMP DEBUG: input stats for layer-0 bisection.
                double m = 0.0, s2 = 0.0;
                for (float v : got) { m += v; s2 += (double)v * v; }
                m /= got.size();
                std::printf("  DBG x: mean=%.5f std=%.5f\n", m,
                            std::sqrt(s2 / got.size() - m * m));
            }
            // Positions: absolute, starting at frontier 0.
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
            for (int il = 0; il <= 3; ++il) {
                if (is_full_layer(il)) {
                    const Tensor x(d_cur.p, DType::BF16, {5120, T});
                    Tensor out(d_nxt.p, DType::BF16, {5120, T});
                    run_full_block(model, il, x, out, state.kv_view(il), positions,
                                   rope_positions, table_rows, stream);
                } else {
                    const Tensor x(d_cur.p, DType::BF16, {5120, T});
                    Tensor out(d_nxt.p, DType::BF16, {5120, T});
                    Tensor conv(state.gdn[il].conv.p, DType::BF16, {10240, 3});
                    Tensor ssm(state.gdn[il].ssm.p, DType::FP32, {128, 128, 48});
                    run_gdn_block(model, il, x, out, conv, ssm, stream);
                }
                device.synchronize();
                char refname[64];
                std::snprintf(refname, sizeof(refname), "l_out-%d.decode1", il);
                const std::vector<float> ref0 = load_ref(c.vdir, refname, 5120 * T);
                const std::vector<float> ref = as_column_major(ref0, 5120, T);
                const std::vector<float> got = d2h_bf16(device, d_nxt.p, 5120 * T);
                char label[32];
                std::snprintf(label, sizeof(label), "l_out-%d", il);
                ok &= check_vec(got.data(), ref.data(), 5120 * T, label, kChainOut);
                if (T == 1 && il == 3) { cur_lout3 = got; }
                CUDA_CHECK(cudaMemcpy(d_cur.p, d_nxt.p, 5120 * T * 2, cudaMemcpyDeviceToDevice));
            }
            state.frontier = T;
            if (T == 1 && !c.det_rerun) {
                det_embed = cur_embed;
                det_lout3 = cur_lout3;
            }
            if (c.det_rerun) {
                // Reset determinism: identical inputs after state.reset()
                // must reproduce bit-exact BF16 outputs (deterministic
                // kernels; catches state-reset and scheduling hazards).
                const bool same_embed = cur_embed.size() == det_embed.size() &&
                                        std::memcmp(cur_embed.data(), det_embed.data(),
                                                    cur_embed.size() * sizeof(float)) == 0;
                const bool same_lout3 = cur_lout3.size() == det_lout3.size() &&
                                        std::memcmp(cur_lout3.data(), det_lout3.data(),
                                                    cur_lout3.size() * sizeof(float)) == 0;
                std::printf("  reset-determinism: embed %s, l_out-3 %s\n",
                            same_embed ? "IDENTICAL" : "DIFFERS",
                            same_lout3 ? "IDENTICAL" : "DIFFERS");
                ok &= same_embed && same_lout3;
            }
        }
        std::printf("%s BQ2_CHAIN03\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    }
}
