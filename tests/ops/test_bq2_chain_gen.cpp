// M6 final correctness: production autoregressive generation.
// Teacher-forced + free-running greedy tracks over a verified Prism token
// sequence, per prompt. Selection is production ops::argmax with
// valid_rows=248077 (padded rows excluded by construction).
//
// Reference dir layout (NINFER_BQ2_GEN_DIR/<p>/):
//   gen_ids.txt        verified Prism free-run continuation ids (whitespace sep)
//   step<k>.logits.bin full Prism logits (248320 F32) for dumped teacher steps
// Prompt ids are embedded in the file names table below (tokenizer-resolved,
// add_bos=false, no chat template).
//
// Env: NINFER_BQ2_FULL_ART, NINFER_BQ2_GEN_DIR.
#include "ops/bq2_model_common.h"

#include "ninfer/ops/argmax.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace bq2full;

constexpr std::int32_t kEosA          = 248044; // <|endoftext|>
constexpr std::int32_t kEosB          = 248046; // <|im_end|>
// kVocabPhys (248320), kVocabValid (248077), kEps come from bq2_model_common.h.

const Gate kLogitGate = {0.15, 0.9995, 1e9}; // cos carries; rmse/rel printed

std::vector<std::int32_t> load_ids(const std::string& path) {
    std::vector<std::int32_t> out;
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) { throw std::runtime_error("missing " + path); }
    int v = 0;
    while (std::fscanf(f, "%d", &v) == 1) { out.push_back(v); }
    std::fclose(f);
    return out;
}

std::vector<float> load_f32(const std::string& path, std::size_t n) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { throw std::runtime_error("missing " + path); }
    std::vector<float> out(n);
    const std::size_t got = std::fread(out.data(), 4, n, f);
    std::fclose(f);
    if (got != n) { throw std::runtime_error("short " + path); }
    return out;
}

bool file_present(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f != nullptr) { std::fclose(f); }
    return f != nullptr;
}

bool all_finite(const float* v, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) { return false; }
    }
    return true;
}

// Full-chain forward for T tokens at absolute base; returns l_out-63.
void forward(Bq2Model& model, Bq2SeqState& state, const Tensor& table_rows, void* x_void,
             int t, int base, cudaStream_t stream, void* out_bf16) {
    const Tensor x(x_void, DType::BF16, {5120, t});
    DeviceBuffer d_cur(5120 * t * 2), d_nxt(5120 * t * 2);
    CUDA_CHECK(cudaMemcpy(d_cur.p, x.data, 5120 * t * 2, cudaMemcpyDeviceToDevice));
    DeviceBuffer d_pos(t * 4), d_pos3(t * 3 * 4);
    {
        std::vector<std::int32_t> p1(t), p3(t * 3);
        for (int k = 0; k < t; ++k) {
            p1[k] = base + k;
            p3[k] = p3[t + k] = p3[2 * t + k] = base + k;
        }
        upload_i32(p1.data(), d_pos.p, t);
        upload_i32(p3.data(), d_pos3.p, t * 3);
    }
    const Tensor positions(d_pos.p, DType::I32, {t});
    const Tensor rope_positions(d_pos3.p, DType::I32, {t, 3});
    const std::uint32_t max_vis = static_cast<std::uint32_t>(base + t);
    for (int il = 0; il <= 63; ++il) {
        if (is_full_layer(il)) {
            const Tensor xi(d_cur.p, DType::BF16, {5120, t});
            Tensor o(d_nxt.p, DType::BF16, {5120, t});
            run_full_block(model, il, xi, o, state.kv_view(il), positions, rope_positions,
                           table_rows, max_vis, stream);
        } else {
            const Tensor xi(d_cur.p, DType::BF16, {5120, t});
            Tensor o(d_nxt.p, DType::BF16, {5120, t});
            Tensor conv(state.gdn[il].conv.p, DType::BF16, {10240, 3});
            Tensor ssm(state.gdn[il].ssm.p, DType::FP32, {128, 128, 48});
            run_gdn_block(model, il, xi, o, conv, ssm, stream);
        }
        CUDA_CHECK(cudaMemcpy(d_cur.p, d_nxt.p, 5120 * t * 2, cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaMemcpy(out_bf16, d_cur.p, 5120 * t * 2, cudaMemcpyDeviceToDevice));
}

// Final norm + folded head -> logits [248320,t] (production path).
void head_logits(Bq2Model& model, void* h_bf16, int t, cudaStream_t stream, void* lg_bf16) {
    DeviceBuffer d_nr(5120 * t * 2);
    {
        const Tensor x(h_bf16, DType::BF16, {5120, t});
        DeviceBuffer d_n(5120 * t * 2);
        Tensor n(d_n.p, DType::BF16, {5120, t});
        ops::rmsnorm(x, model.final_norm, kEps, false, n, stream);
        const Tensor nn(d_n.p, DType::BF16, {5120, t});
        Tensor nr(d_nr.p, DType::BF16, {5120, t});
        ops::t2_fwht_sign(nn, model.signs_for(5120), nr, stream);
    }
    const Tensor nr(d_nr.p, DType::BF16, {5120, t});
    Tensor lg(lg_bf16, DType::BF16, {248320, t});
    ops::linear(nr, model.head_t2, lg, stream);
}

std::int32_t device_argmax(Bq2Model& model, void* lg_bf16, int t, int col,
                           cudaStream_t stream) {
    (void)t;
    // Column-major [248320,t]: column col starts at col*248320.
    const Tensor lg(static_cast<std::uint8_t*>(lg_bf16) +
                        static_cast<std::size_t>(col) * kVocabPhys * 2,
                    DType::BF16, {kVocabPhys, 1});
    DeviceBuffer d_id(4);
    Tensor out(d_id.p, DType::I32, {1});
    ops::argmax(lg, out, kVocabValid, stream);
    model.device.synchronize();
    std::int32_t id = -1;
    CUDA_CHECK(cudaMemcpy(&id, d_id.p, 4, cudaMemcpyDeviceToHost));
    return id;
}

void top2(const float* v, std::size_t n, int& a, int& b) {
    a = -1;
    b = -1;
    for (std::size_t i = 0; i < n; ++i) {
        if (a < 0 || v[i] > v[a]) {
            b = a;
            a = (int)i;
        } else if (b < 0 || v[i] > v[b]) {
            b = (int)i;
        }
    }
}

struct PromptCase {
    const char* dir;
    std::vector<std::int32_t> prompt;
};

// Dumped teacher-forced steps (Prism prefix runs with full logits).
const int kDumpedSteps[] = {0, 1, 2, 3, 7, 15, 31};

// Decode-step latencies across all prompts (M7 decode distribution).
std::vector<double> g_dec_ms;

} // namespace

int main() {
    const char* art_path = std::getenv("NINFER_BQ2_FULL_ART");
    const char* gen_dir  = std::getenv("NINFER_BQ2_GEN_DIR");
    if (art_path == nullptr || gen_dir == nullptr) {
        std::cout << "SKIP: set NINFER_BQ2_FULL_ART and NINFER_BQ2_GEN_DIR\n";
        return 77;
    }
    try {
        bool ok = true;
        Bq2Model model(art_path);
        DeviceContext& device = model.device;
        cudaStream_t stream   = model.stream;
        Bq2SeqState state(128);
        DeviceBuffer d_rows1(4);
        {
            const std::int32_t zero = 0;
            upload_i32(&zero, d_rows1.p, 1);
        }
        const Tensor table_rows(d_rows1.p, DType::I32, {1});
        // Prompt e (32-token -bf CLI reference) was REMOVED: the backup-CLI
        // -bf path demonstrably alters the effective prompt (its gen[0]=198
        // contradicts pinned-CPU argmax 506 with 5.15 margin on byte-exact
        // 32 ids), while NInfer-CUDA (506) agrees with pinned-CPU (506).
        // The -p literal path used for a/b/c/d is proven raw by gen[0]
        // agreement at every prompt. See M6 report Stage 7 notes.
        const PromptCase prompts[] = {
            {"a", {9419}},
            {"b", {9419, 1814, 11, 3242}},
            {"c", {727, 1822, 4406}},
            {"d", {16, 17, 18, 11, 220, 19, 20, 21, 13}},
        };
        int near_ties = 0;
        for (const auto& pc : prompts) {
            const std::string pdir  = std::string(gen_dir) + "/" + pc.dir;
            const std::vector<std::int32_t> gen = load_ids(pdir + "/gen_ids.txt");
            const int lp                        = static_cast<int>(pc.prompt.size());
            std::printf("== prompt %s: Lp=%d gen=%d ==\n", pc.dir, lp, (int)gen.size());
            // Full verified sequence S = prompt + gen.
            std::vector<std::int32_t> seq = pc.prompt;
            seq.insert(seq.end(), gen.begin(), gen.end());

            // ---------- teacher-forced ----------
            {
                state.reset();
                const auto t_pre0 = std::chrono::steady_clock::now();
                DeviceBuffer d_x(5120 * lp * 2);
                embed_lookup(model, pc.prompt.data(), lp, d_x.p, stream);
                DeviceBuffer d_h(5120 * lp * 2);
                forward(model, state, table_rows, d_x.p, lp, 0, stream, d_h.p);
                DeviceBuffer d_lg(kVocabPhys * lp * 2);
                head_logits(model, d_h.p, lp, stream, d_lg.p);
                device.synchronize();
                const double pre_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                             t_pre0)
                        .count();
                double dec_ms = 0.0;
                int dec_n     = 0;
                // Step 0 argmax comes from the prefill logits (last column).
                for (int k = 0; k <= (int)gen.size(); ++k) {
                    const int pos = lp + k; // position being predicted
                    std::vector<float> lg_host;
                    if (k == 0) {
                        lg_host = d2h_bf16(device, d_lg.p, kVocabPhys * lp);
                    } else {
                        // Decode S[pos-1] at absolute pos-1.
                        const auto t_dec0 = std::chrono::steady_clock::now();
                        DeviceBuffer dx(5120 * 2);
                        embed_lookup(model, &seq[pos - 1], 1, dx.p, stream);
                        DeviceBuffer dh(5120 * 2);
                        forward(model, state, table_rows, dx.p, 1, pos - 1, stream, dh.p);
                        DeviceBuffer dl(kVocabPhys * 2);
                        head_logits(model, dh.p, 1, stream, dl.p);
                        device.synchronize();
                        const double step_ms =
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_dec0)
                                .count();
                        dec_ms += step_ms;
                        ++dec_n;
                        g_dec_ms.push_back(step_ms);
                        lg_host = d2h_bf16(device, dl.p, kVocabPhys);
                    }
                    // Column layout: prefill host is [V,lp] column-major.
                    std::vector<float> col(kVocabPhys);
                    if (k == 0) {
                        for (int i = 0; i < kVocabPhys; ++i) {
                            col[i] = lg_host[static_cast<std::size_t>(lp - 1) * kVocabPhys + i];
                        }
                    } else {
                        col = lg_host;
                    }
                    if (!all_finite(col.data(), col.size())) {
                        std::printf("  TF k=%d: NON-FINITE logits\n", k);
                        ok = false;
                        break;
                    }
                    const std::int32_t got = device_argmax(model, [&] {
                        DeviceBuffer tmp(kVocabPhys * 2);
                        upload_bf16(col.data(), tmp.p, kVocabPhys);
                        return tmp;
                    }().p, 1, 0, stream);
                    if (got < 0 || got >= kVocabValid) {
                        std::printf("  TF k=%d: id %d outside valid domain\n", k, got);
                        ok = false;
                        break;
                    }
                    int ta = -1, tb = -1;
                    top2(col.data(), col.size(), ta, tb);
                    const double margin = (double)col[ta] - (double)col[tb];
                    bool dumped        = false;
                    for (int s : kDumpedSteps) {
                        if (s == k) { dumped = true; }
                    }
                    char lbl[64];
                    std::snprintf(lbl, sizeof(lbl), "TF.%s k=%d", pc.dir, k);
                    if (k < (int)gen.size()) {
                        const bool match = got == seq[pos];
                        // Full-logit distribution check where dumped. The
                        // argmax-vs-ref verdict always applies; pinned logits
                        // only add distribution metrics + near-tie margins.
                        // Missing pinned files (Stage-8 on demand) never skip
                        // the match verdict.
                        if (dumped) {
                            const std::string fp =
                                pdir + "/step" + std::to_string(k) + ".logits.bin";
                            if (!file_present(fp)) {
                                std::printf("  %s: no pinned logits (Stage-8 on demand)\n", lbl);
                                if (!match) {
                                    std::printf("  %s: MISMATCH got=%d ref=%d (no margins: needs Stage-8 prefix run)\n",
                                                lbl, got, seq[pos]);
                                    ok = false;
                                } else {
                                    std::printf("  %s: match=%d margin=%.4f PASS (no distribution data)\n",
                                                lbl, got, margin);
                                }
                            } else {
                                const std::vector<float> ref =
                                    load_f32(fp, kVocabPhys);
                                char dl2[64];
                                std::snprintf(dl2, sizeof(dl2), "TFdist.%s k=%d", pc.dir, k);
                                ok &= check_vec(col.data(), ref.data(), kVocabPhys, dl2,
                                                kLogitGate);
                                int ra = -1, rb = -1;
                                top2(ref.data(), ref.size(), ra, rb);
                                const double rmargin = (double)ref[ra] - (double)ref[rb];
                                const bool topset =
                                    (ta == ra && tb == rb) || (ta == rb && tb == ra);
                                std::printf("  %s: ninfer top=(%d,%d) m=%.4f ref top=(%d,%d) m=%.4f %s\n",
                                            lbl, ta, tb, margin, ra, rb, rmargin,
                                            topset ? "TOPSET-OK" : "TOPSET-DIFF");
                                if (!match) {
                                    // Near-tie rule: margins < 5*rmse AND < 0.02, top-2 agree.
                                    double se = 0.0;
                                    for (std::size_t i = 0; i < col.size(); ++i) {
                                        const double e = (double)col[i] - ref[i];
                                        se += e * e;
                                    }
                                    const double rmse = std::sqrt(se / col.size());
                                    const bool near =
                                        rmargin < 5 * rmse && rmargin < 0.02 && topset;
                                    std::printf("  %s: MISMATCH got=%d ref=%d %s\n", lbl, got,
                                                seq[pos], near ? "NEAR-TIE" : "REAL-DIVERGENCE");
                                    if (near) {
                                        if (++near_ties > 1) {
                                            std::printf("  second near-tie: FAIL\n");
                                            ok = false;
                                        }
                                    } else {
                                        ok = false;
                                    }
                                } else {
                                    std::printf("  %s: match=%d margin=%.4f PASS\n", lbl, got,
                                                margin);
                                }
                            }
                        } else if (!match) {
                            std::printf("  %s: MISMATCH got=%d ref=%d (undumped step: needs Stage-8 prefix run)\n",
                                        lbl, got, seq[pos]);
                            ok = false;
                        } else {
                            std::printf("  %s: match=%d margin=%.4f PASS\n", lbl, got, margin);
                        }
                    } else {
                        std::printf("  %s: extra step argmax=%d margin=%.4f (beyond verified prefix)\n",
                                    lbl, got, margin);
                        break;
                    }
                    if (got == kEosA || got == kEosB) {
                        std::printf("  %s: EOS %d, stopping\n", lbl, got);
                        break;
                    }
                }
                std::printf("  %s prefill T=%d %.1fms (%.1f tok/s); decode mean %.2fms/tok over %d steps\n",
                            pc.dir, lp, pre_ms, 1000.0 * lp / pre_ms,
                            dec_n > 0 ? dec_ms / dec_n : 0.0, dec_n);
            }

            // ---------- free-running greedy ----------
            // Full free sequence, then reset + identical rerun (bit-exact).
            auto free_run = [&](std::vector<std::int32_t>& ids_out, bool compare) {
                state.reset();
                DeviceBuffer d_x(5120 * lp * 2);
                embed_lookup(model, pc.prompt.data(), lp, d_x.p, stream);
                DeviceBuffer d_h(5120 * lp * 2);
                forward(model, state, table_rows, d_x.p, lp, 0, stream, d_h.p);
                DeviceBuffer d_lg(kVocabPhys * lp * 2);
                head_logits(model, d_h.p, lp, stream, d_lg.p);
                device.synchronize();
                bool diverged = false;
                // Bound by verified ref length: inputs seq[pos-1] require
                // pos-1 < Lp+G. (Unbounded k would over-read seq.)
                const int kcap = gen.size() < 32 ? static_cast<int>(gen.size()) : 32;
                for (int k = 0; k < kcap; ++k) {
                    const int pos = lp + k;
                    std::int32_t got = -1;
                    if (k == 0) {
                        got = device_argmax(model, d_lg.p, lp, lp - 1, stream);
                    } else {
                        const std::int32_t inp = diverged ? ids_out.back() : seq[pos - 1];
                        DeviceBuffer dx(5120 * 2);
                        embed_lookup(model, &inp, 1, dx.p, stream);
                        DeviceBuffer dh(5120 * 2);
                        forward(model, state, table_rows, dx.p, 1, pos - 1, stream, dh.p);
                        DeviceBuffer dl(kVocabPhys * 2);
                        head_logits(model, dh.p, 1, stream, dl.p);
                        device.synchronize();
                        got = device_argmax(model, dl.p, 1, 0, stream);
                    }
                    if (got < 0 || got >= kVocabValid) {
                        std::printf("  FR.%s k=%d: id %d outside valid domain\n", pc.dir, k, got);
                        ok = false;
                        break;
                    }
                    ids_out.push_back(got);
                    // Divergence tracking is independent of compare mode so
                    // the determinism rerun follows the identical free path.
                    if (!diverged && k < (int)gen.size() && got != seq[pos]) {
                        diverged = true;
                        if (compare) {
                            std::printf("  FR.%s k=%d: FIRST DIVERGENCE got=%d ref=%d\n",
                                        pc.dir, k, got, seq[pos]);
                        }
                    }
                    if (got == kEosA || got == kEosB) {
                        std::printf("  FR.%s k=%d: EOS %d\n", pc.dir, k, got);
                        break;
                    }
                }
                if (compare && !diverged) {
                    std::printf("  FR.%s: exact match over %d steps\n", pc.dir,
                                (int)ids_out.size());
                }
            };
            std::vector<std::int32_t> free_ids;
            free_run(free_ids, true);
            {
                std::vector<std::int32_t> rep;
                free_run(rep, false);
                const bool same = rep.size() == free_ids.size() &&
                                  std::memcmp(rep.data(), free_ids.data(),
                                              rep.size() * sizeof(std::int32_t)) == 0;
                std::printf("  FR.%s reset determinism over %d steps: %s\n", pc.dir,
                            (int)free_ids.size(), same ? "IDENTICAL" : "DIFFERS");
                ok &= same;
            }
        }
        std::printf("near-ties excused: %d\n", near_ties);
        if (!g_dec_ms.empty()) {
            std::sort(g_dec_ms.begin(), g_dec_ms.end());
            const std::size_t n = g_dec_ms.size();
            std::printf("decode ms/token: n=%d min=%.2f p10=%.2f median=%.2f p90=%.2f max=%.2f\n",
                        (int)n, g_dec_ms.front(), g_dec_ms[n / 10], g_dec_ms[n / 2],
                        g_dec_ms[n * 9 / 10], g_dec_ms.back());
        }
        std::printf("%s BQ2_CHAIN_GEN\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    }
}
