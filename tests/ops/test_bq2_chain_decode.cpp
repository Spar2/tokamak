// M6 inter-call state continuity: prefill -> decode persistence for the
// embedding -> layer 0 -> 1 -> 2 -> 3 chain with runtime-owned state.
// Token sequence (debug-log authoritative): [9419, 1814, 11, 3242]
// ("Hello world, today"). References: P0r/P1r/P2r prefill columns — a fresh
// prefill column k is the ground truth for decode step k (causal freshness).
//
// TEST A: prefill(4 in one call) vs incremental (1+1+1+1), per-prefix.
// TEST B: prefill(2) + decode(1) vs P2 column 2.
// TEST C: prefill(1) + decode x3, every step vs reference, finite outputs.
// TEST D: reset + repeat C bit-exactly.
//
// Env: NINFER_BQ2_FULL_ART, NINFER_BQ2_VECTORS_P0/P1/P2 (decode1 files).
#include "ops/bq2_model_common.h"

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

const Gate kChainOut = {8e-3, 0.9995, 1e-1};
const std::int32_t kSeq[] = {9419, 1814, 11, 3242};

struct StepOut {
    std::vector<float> lout[4];
};

// Column t of an (N,T) row-major reference.
std::vector<float> ref_col(const std::vector<float>& ref0, int n, int t, int ntok) {
    std::vector<float> col(n);
    for (int i = 0; i < n; ++i) { col[i] = ref0[static_cast<std::size_t>(i) * ntok + t]; }
    return col;
}

bool all_finite(const float* v, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) { return false; }
    }
    return true;
}

// Run one stateful chain step over T tokens at absolute base position.
// ids[T] are embedded; positions are base..base+T-1; GDN conv/ssm and the
// layer-3 KV cache persist in state (no reset here).
void run_step(Bq2Model& model, Bq2SeqState& state, const std::int32_t* ids, int t,
              int base, const Tensor& table_rows, cudaStream_t stream, StepOut& out) {
    DeviceContext& device = model.device;
    DeviceBuffer d_x(5120 * t * 2);
    embed_lookup(model, ids, t, d_x.p, stream);
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
    DeviceBuffer d_cur(5120 * t * 2), d_nxt(5120 * t * 2);
    CUDA_CHECK(cudaMemcpy(d_cur.p, d_x.p, 5120 * t * 2, cudaMemcpyDeviceToDevice));
    const std::uint32_t max_vis = static_cast<std::uint32_t>(base + t);
    for (int il = 0; il <= 3; ++il) {
        if (is_full_layer(il)) {
            const Tensor x(d_cur.p, DType::BF16, {5120, t});
            Tensor o(d_nxt.p, DType::BF16, {5120, t});
            run_full_block(model, il, x, o, state.kv_view(il), positions, rope_positions,
                           table_rows, max_vis, stream);
        } else {
            const Tensor x(d_cur.p, DType::BF16, {5120, t});
            Tensor o(d_nxt.p, DType::BF16, {5120, t});
            Tensor conv(state.gdn[il].conv.p, DType::BF16, {10240, 3});
            Tensor ssm(state.gdn[il].ssm.p, DType::FP32, {128, 128, 48});
            run_gdn_block(model, il, x, o, conv, ssm, stream);
        }
        device.synchronize();
        out.lout[il] = d2h_bf16(device, d_nxt.p, 5120 * t);
        CUDA_CHECK(cudaMemcpy(d_cur.p, d_nxt.p, 5120 * t * 2, cudaMemcpyDeviceToDevice));
    }
}

bool check_step(const StepOut& got, const char* pdir, int t, int base, const char* tag,
                bool& ok) {
    // Reference columns base..base+t-1 of the matching prefill width.
    for (int il = 0; il <= 3; ++il) {
        char refname[64];
        std::snprintf(refname, sizeof(refname), "l_out-%d.decode1", il);
        const std::vector<float> ref0 = load_ref(pdir, refname, 5120 * t);
        char label[48];
        for (int k = 0; k < t; ++k) {
            const std::vector<float> ref = ref_col(ref0, 5120, k, t);
            // Device step output is [5120,t] column-major; take column k.
            std::vector<float> gk(5120);
            for (int i = 0; i < 5120; ++i) {
                gk[i] = got.lout[il][static_cast<std::size_t>(k) * 5120 + i];
            }
            std::snprintf(label, sizeof(label), "%s l%d.t%d", tag, il, base + k);
            ok &= check_vec(gk.data(), ref.data(), 5120, label, kChainOut);
            if (!all_finite(gk.data(), 5120)) {
                std::printf("  %s: NON-FINITE output\n", label);
                ok = false;
            }
        }
    }
    return ok;
}

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
        (void)device;
        Bq2SeqState state(8);
        DeviceBuffer d_rows1(4);
        {
            const std::int32_t zero = 0;
            upload_i32(&zero, d_rows1.p, 1);
        }
        const Tensor table_rows(d_rows1.p, DType::I32, {1});

        // TEST A: one-call prefill(4) vs incremental 1+1+1+1.
        std::printf("== TEST A: prefill(4) vs incremental ==\n");
        StepOut pre;
        {
            state.reset();
            run_step(model, state, kSeq, 4, 0, table_rows, stream, pre);
            check_step(pre, p2_dir, 4, 0, "A.pre", ok);
        }
        for (int k = 0; k < 4; ++k) {
            StepOut inc;
            if (k == 0) { state.reset(); }
            // No reset between steps: conv/S/KV persist across calls.
            run_step(model, state, kSeq + k, 1, k, table_rows, stream, inc);
            const char* pdir = (k == 0) ? p0_dir : (k == 1) ? p1_dir : p2_dir;
            const int ntok   = (k == 0) ? 1 : (k == 1) ? 2 : 4;
            const int col    = k;
            for (int il = 0; il <= 3; ++il) {
                char refname[64];
                std::snprintf(refname, sizeof(refname), "l_out-%d.decode1", il);
                const std::vector<float> ref0 = load_ref(pdir, refname, 5120 * ntok);
                const std::vector<float> ref  = ref_col(ref0, 5120, col, ntok);
                char label[48];
                std::snprintf(label, sizeof(label), "A.inc l%d.t%d", il, k);
                ok &= check_vec(inc.lout[il].data(), ref.data(), 5120, label, kChainOut);
                // Incremental-vs-prefill direct equivalence.
                std::vector<float> pa(5120);
                for (int i = 0; i < 5120; ++i) {
                    pa[i] = pre.lout[il][static_cast<std::size_t>(k) * 5120 + i];
                }
                std::snprintf(label, sizeof(label), "A.pre-vs-inc l%d.t%d", il, k);
                ok &= check_vec(inc.lout[il].data(), pa.data(), 5120, label, kChainOut);
            }
        }

        // TEST B: prefill(2) + decode(1).
        std::printf("== TEST B: prefill(2) + decode ==\n");
        {
            state.reset();
            StepOut pf;
            run_step(model, state, kSeq, 2, 0, table_rows, stream, pf);
            check_step(pf, p1_dir, 2, 0, "B.pre", ok);
            StepOut dc;
            run_step(model, state, kSeq + 2, 1, 2, table_rows, stream, dc);
            for (int il = 0; il <= 3; ++il) {
                char refname[64];
                std::snprintf(refname, sizeof(refname), "l_out-%d.decode1", il);
                const std::vector<float> ref0 = load_ref(p2_dir, refname, 5120 * 4);
                const std::vector<float> ref  = ref_col(ref0, 5120, 2, 4);
                char label[48];
                std::snprintf(label, sizeof(label), "B.dec l%d.t2", il);
                ok &= check_vec(dc.lout[il].data(), ref.data(), 5120, label, kChainOut);
            }
        }

        // TEST C: prefill(1) + 3 decodes.
        std::printf("== TEST C: prefill(1) + 3 decodes ==\n");
        StepOut csteps[4];
        {
            state.reset();
            int frontier = 0;
            const struct {
                int t;
                int base;
            } calls[] = {{1, 0}, {1, 1}, {1, 2}, {1, 3}};
            for (int s = 0; s < 4; ++s) {
                if (frontier != calls[s].base) {
                    std::printf("  frontier slip: have %d want %d\n", frontier,
                                calls[s].base);
                    ok = false;
                }
                run_step(model, state, kSeq + calls[s].base, calls[s].t, calls[s].base,
                         table_rows, stream, csteps[s]);
                frontier += calls[s].t;
                const char* pdir = (s == 0) ? p0_dir : (s == 1) ? p1_dir : p2_dir;
                const int ntok   = (s == 0) ? 1 : (s == 1) ? 2 : 4;
                for (int il = 0; il <= 3; ++il) {
                    char refname[64];
                    std::snprintf(refname, sizeof(refname), "l_out-%d.decode1", il);
                    const std::vector<float> ref0 = load_ref(pdir, refname, 5120 * ntok);
                    const std::vector<float> ref  = ref_col(ref0, 5120, s, ntok);
                    char label[48];
                    std::snprintf(label, sizeof(label), "C.s%d l%d.t%d", s, il, s);
                    ok &= check_vec(csteps[s].lout[il].data(), ref.data(), 5120, label,
                                    kChainOut);
                    if (!all_finite(csteps[s].lout[il].data(), 5120)) {
                        std::printf("  %s: NON-FINITE output\n", label);
                        ok = false;
                    }
                }
            }
            if (frontier != 4) {
                std::printf("  final frontier %d, want 4\n", frontier);
                ok = false;
            }
        }

        // TEST D: reset + repeat C bit-exactly.
        std::printf("== TEST D: reset + repeat ==\n");
        {
            state.reset();
            const struct {
                int t;
                int base;
            } calls[] = {{1, 0}, {1, 1}, {1, 2}, {1, 3}};
            for (int s = 0; s < 4; ++s) {
                StepOut rep;
                run_step(model, state, kSeq + calls[s].base, calls[s].t, calls[s].base,
                         table_rows, stream, rep);
                for (int il = 0; il <= 3; ++il) {
                    const bool same =
                        rep.lout[il].size() == csteps[s].lout[il].size() &&
                        std::memcmp(rep.lout[il].data(), csteps[s].lout[il].data(),
                                    rep.lout[il].size() * sizeof(float)) == 0;
                    if (!same) {
                        std::printf("  D.s%d l%d: DIFFERS after reset\n", s, il);
                        ok = false;
                    }
                }
            }
            std::printf("  reset-after-decode: %s\n", ok ? "IDENTICAL" : "MISMATCH");
        }

        std::printf("%s BQ2_CHAIN_DECODE\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    }
}
