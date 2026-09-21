// M7-D2 in-situ timing: full-model T=1 decode replica with per-family
// cudaEvent brackets (2 reusable events, one sync per rep). Compares
// production dispatch (T1-A) vs direct old-kernel launch on the IDENTICAL
// call sequence (old path = same shapes/streams, launch_t2_gemv direct).
// Env: NINFER_BQ2_FULL_ART.
#include "ops/bq2_model_common.h"

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

struct Ev {
    cudaEvent_t a, b;
    Ev() {
        CUDA_CHECK(cudaEventCreate(&a));
        CUDA_CHECK(cudaEventCreate(&b));
    }
    ~Ev() {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
};

struct Fam {
    const char* name;
    double total = 0.0;
    int calls    = 0;
};

struct Probes {
    Ev ev;
    Fam lin_qkv{"qkv"}, lin_gate{"gate"}, lin_so{"so"}, lin_fg{"fg"}, lin_fu{"fu"}, lin_fd{"fd"};
    Fam lin_aq{"attn_q"}, lin_ak{"attn_k"}, lin_av{"attn_v"}, lin_wo{"attn_wo"}, lin_head{"head"};
    Fam fwht{"fwht"}, rmsnorm{"rmsnorm"}, conv{"conv"}, gdn{"gdn"}, attn{"attn"}, acts{"acts"};
    Fam resid{"resid"}, host{"host"}, d2misc{"d2misc"};
    Fam* all[20] = {&lin_qkv, &lin_gate, &lin_so, &lin_fg, &lin_fu, &lin_fd,
                    &lin_aq,  &lin_ak,  &lin_av,  &lin_wo, &lin_head,
                    &fwht,    &rmsnorm, &conv,    &gdn,    &attn,
                    &acts,    &resid,   &host,    &d2misc};
    void brief(const char* tag) {
        double sum = 0.0;
        for (auto* f : all) { sum += f->total; }
        std::printf("--- %s: event-sum=%.3fms ---\n", tag, sum);
        for (auto* f : all) {
            if (f->calls > 0) {
                std::printf("  %-9s total=%8.3fms calls=%4d mean=%7.3fus pct=%.1f%%\n", f->name,
                            f->total, f->calls, f->total * 1000.0 / f->calls,
                            100.0 * f->total / sum);
            }
        }
    }
};

inline void acc(Probes& p, Fam Probes::*fam, cudaStream_t s) {
    CUDA_CHECK(cudaEventSynchronize(p.ev.b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, p.ev.a, p.ev.b));
    (p.*fam).total += ms;
    (p.*fam).calls += 1;
}

#define BRK(probe, fam, stream, stmt)              \
    do {                                           \
        CUDA_CHECK(cudaEventRecord(probe.ev.a, stream)); \
        stmt;                                      \
        CUDA_CHECK(cudaEventRecord(probe.ev.b, stream)); \
        acc(probe, &Probes::fam, stream);          \
    } while (0)

// Full T=1 decode replica with per-family brackets. Mirrors run_gdn_block /
// run_full_block / embed / head exactly (shapes, streams, state use).
void decode_rep(Bq2Model& m, Bq2SeqState& state, const Tensor& table_rows, const void* x0,
                bool use_old, Probes& p, cudaStream_t s, void* out_bf16) {
    const int T = 1;
    DeviceBuffer d_cur(5120 * 2), d_nxt(5120 * 2);
    CUDA_CHECK(cudaMemcpy(d_cur.p, x0, 5120 * 2, cudaMemcpyDeviceToDevice));
    DeviceBuffer d_pos(4), d_pos3(12);
    {
        const std::int32_t z = 0;
        upload_i32(&z, d_pos.p, 1);
        upload_i32(&z, d_pos3.p, 1);
        upload_i32(&z, d_pos3.p + 4, 1);
        upload_i32(&z, d_pos3.p + 8, 1);
    }
    const Tensor positions(d_pos.p, DType::I32, {1});
    const Tensor rope_positions(d_pos3.p, DType::I32, {1, 3});
    for (int il = 0; il <= 63; ++il) {
        if (is_full_layer(il)) {
            const FullLayerW& w = m.full[il];
            DeviceBuffer d_h(5120 * 2), d_hr(17408 * 2);
            DeviceBuffer d_qfull(12288 * 2), d_k(1024 * 2), d_v(1024 * 2);
            DeviceBuffer d_q(6144 * 2), d_gate(6144 * 2);
            DeviceBuffer d_qn(6144 * 2), d_kn(1024 * 2);
            DeviceBuffer d_attn(6144 * 2);
            DeviceBuffer d_ao(5120 * 2), d_x1(5120 * 2), d_mh(5120 * 2);
            DeviceBuffer d_gv(17408 * 2), d_uv(17408 * 2), d_act(17408 * 2);
            DeviceBuffer d_d(5120 * 2);
            BRK(p, rmsnorm, s,
                { const Tensor xx(d_cur.p, DType::BF16, {5120, 1}); Tensor hh(d_h.p, DType::BF16, {5120, 1}); ops::rmsnorm(xx, w.an, kEps, false, hh, s); });
            BRK(p, fwht, s,
                { const Tensor hh(d_h.p, DType::BF16, {5120, 1}); Tensor hr(d_hr.p, DType::BF16, {5120, 1}); ops::t2_fwht_sign(hh, m.signs_for(5120), hr, s); });
            {
                const Tensor hr(d_hr.p, DType::BF16, {5120, 1});
                Tensor qf(d_qfull.p, DType::BF16, {12288, 1});
                Tensor kf(d_k.p, DType::BF16, {1024, 1});
                Tensor vf(d_v.p, DType::BF16, {1024, 1});
                CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                if (use_old) {
                    ops::detail::launch_t2_gemv(hr, w.q, qf, s);
                } else {
                    ops::linear(hr, w.q, qf, s);
                }
                CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                acc(p, &Probes::lin_aq, s);
                CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                if (use_old) {
                    ops::detail::launch_t2_gemv(hr, w.k, kf, s);
                } else {
                    ops::linear(hr, w.k, kf, s);
                }
                CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                acc(p, &Probes::lin_ak, s);
                CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                if (use_old) {
                    ops::detail::launch_t2_gemv(hr, w.v, vf, s);
                } else {
                    ops::linear(hr, w.v, vf, s);
                }
                CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                acc(p, &Probes::lin_av, s);
            }
            {
                // Host Q-split (timed as host drain, like production).
                auto t0 = std::chrono::steady_clock::now();
                m.device.synchronize();
                std::vector<std::uint16_t> qf_bits(12288);
                CUDA_CHECK(cudaMemcpy(qf_bits.data(), d_qfull.p, qf_bits.size() * 2,
                                      cudaMemcpyDeviceToHost));
                std::vector<std::uint16_t> q_bits(6144), gate_bits(6144);
                for (int hh = 0; hh < 24; ++hh) {
                    for (int d = 0; d < 256; ++d) {
                        q_bits[hh * 256 + d]    = qf_bits[hh * 512 + d];
                        gate_bits[hh * 256 + d] = qf_bits[hh * 512 + 256 + d];
                    }
                }
                CUDA_CHECK(cudaMemcpy(d_q.p, q_bits.data(), q_bits.size() * 2,
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_gate.p, gate_bits.data(), gate_bits.size() * 2,
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaStreamSynchronize(nullptr));
                const double ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                        .count();
                p.host.total += ms;
                p.host.calls += 1;
            }
            BRK(p, rmsnorm, s,
                { const Tensor qq(d_q.p, DType::BF16, {256, 24, 1}); const Tensor wq(w.qn.data, DType::BF16, {256}); Tensor qn(d_qn.p, DType::BF16, {256, 24, 1}); ops::rmsnorm(qq, wq, kEps, false, qn, s); });
            BRK(p, rmsnorm, s,
                { const Tensor kk(d_k.p, DType::BF16, {256, 4, 1}); const Tensor wk(w.kn.data, DType::BF16, {256}); Tensor kn(d_kn.p, DType::BF16, {256, 4, 1}); ops::rmsnorm(kk, wk, kEps, false, kn, s); });
            BRK(p, acts, s,
                { Tensor qq(d_qn.p, DType::BF16, {256, 24, 1}); Tensor kk(d_kn.p, DType::BF16, {256, 4, 1}); ops::rope(rope_positions, kRotaryDim, kRopeTheta, qq, kk, s); });
            const ops::CausalAttentionExecutionEnvelope env{1, 1};
            DeviceArena ws_attn(std::max<std::size_t>(
                ops::causal_softmax_attention_workspace_capacity_bytes(
                    ops::AttentionHeadGeometry{256, 24, 4}, KvCacheStorage::BFloat16, env, 1, 1, 1),
                256));
            BRK(p, attn, s,
                {
                    const Tensor qq(d_qn.p, DType::BF16, {256, 24, 1});
                    const Tensor kk(d_kn.p, DType::BF16, {256, 4, 1});
                    const Tensor vv(d_v.p, DType::BF16, {256, 4, 1});
                    const Tensor rows(table_rows);
                    Tensor out3(d_attn.p, DType::BF16, {256, 24, 1});
                    const Tensor empty;
                    ops::causal_softmax_attention(qq, kk, vv, positions, empty, rows,
                                                  ops::AttentionHeadGeometry{256, 24, 4},
                                                  kAttnScale, state.kv_view(il), env, ws_attn,
                                                  out3, s);
                });
            BRK(p, acts, s,
                { const Tensor gate(d_gate.p, DType::BF16, {6144, 1}); Tensor ax(d_attn.p, DType::BF16, {6144, 1}); ops::sigmoid_mul(gate, ax, s); });
            BRK(p, fwht, s,
                { const Tensor ax(d_attn.p, DType::BF16, {6144, 1}); Tensor axr(d_hr.p, DType::BF16, {6144, 1}); ops::t2_fwht_sign(ax, m.signs_for(6144), axr, s); });
            {
                const Tensor axr(d_hr.p, DType::BF16, {6144, 1});
                Tensor ao(d_ao.p, DType::BF16, {5120, 1});
                CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                if (use_old) {
                    ops::detail::launch_t2_gemv(axr, w.wo, ao, s);
                } else {
                    ops::linear(axr, w.wo, ao, s);
                }
                CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                acc(p, &Probes::lin_wo, s);
            }
            BRK(p, resid, s,
                {
                    CUDA_CHECK(cudaMemcpyAsync(d_x1.p, d_cur.p, 5120 * 2, cudaMemcpyDeviceToDevice, s));
                    const Tensor ao(d_ao.p, DType::BF16, {5120, 1});
                    Tensor x1(d_x1.p, DType::BF16, {5120, 1});
                    ops::residual_add(ao, x1, s);
                });
            BRK(p, rmsnorm, s,
                { const Tensor x1(d_x1.p, DType::BF16, {5120, 1}); const Tensor wpost(w.pn.data, DType::BF16, {5120}); Tensor mh(d_mh.p, DType::BF16, {5120, 1}); ops::rmsnorm(x1, wpost, kEps, false, mh, s); });
            BRK(p, fwht, s,
                { const Tensor mh(d_mh.p, DType::BF16, {5120, 1}); Tensor mhr(d_hr.p, DType::BF16, {5120, 1}); ops::t2_fwht_sign(mh, m.signs_for(5120), mhr, s); });
            {
                const Tensor mhr(d_hr.p, DType::BF16, {5120, 1});
                Tensor gv(d_gv.p, DType::BF16, {17408, 1});
                Tensor uv(d_uv.p, DType::BF16, {17408, 1});
                CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                if (use_old) {
                    ops::detail::launch_t2_gemv(mhr, w.fg, gv, s);
                    CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                    acc(p, &Probes::lin_fg, s);
                    CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                    ops::detail::launch_t2_gemv(mhr, w.fu, uv, s);
                } else {
                    ops::linear(mhr, w.fg, gv, s);
                    CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                    acc(p, &Probes::lin_fg, s);
                    CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                    ops::linear(mhr, w.fu, uv, s);
                }
                CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                acc(p, &Probes::lin_fu, s);
            }
            BRK(p, acts, s,
                { const Tensor gv(d_gv.p, DType::BF16, {17408, 1}); const Tensor uv(d_uv.p, DType::BF16, {17408, 1}); Tensor act(d_act.p, DType::BF16, {17408, 1}); ops::silu_mul(gv, uv, act, s); });
            BRK(p, fwht, s,
                { const Tensor act(d_act.p, DType::BF16, {17408, 1}); Tensor actr(d_hr.p, DType::BF16, {17408, 1}); ops::t2_fwht_sign(act, m.signs_for(17408), actr, s); });
            {
                const Tensor actr(d_hr.p, DType::BF16, {17408, 1});
                Tensor dd(d_d.p, DType::BF16, {5120, 1});
                CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                if (use_old) {
                    ops::detail::launch_t2_gemv(actr, w.fd, dd, s);
                } else {
                    ops::linear(actr, w.fd, dd, s);
                }
                CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                acc(p, &Probes::lin_fd, s);
            }
            BRK(p, resid, s,
                {
                    CUDA_CHECK(cudaMemcpyAsync(d_nxt.p, d_x1.p, 5120 * 2, cudaMemcpyDeviceToDevice, s));
                    const Tensor dd(d_d.p, DType::BF16, {5120, 1});
                    Tensor lo(d_nxt.p, DType::BF16, {5120, 1});
                    ops::residual_add(dd, lo, s);
                });
        } else {
            const GdnLayerW& w = m.gdn[il];
            DeviceBuffer d_h(5120 * 2), d_hr(17408 * 2);
            DeviceBuffer d_qkv(10240 * 2), d_z(6144 * 2);
            DeviceBuffer d_g(48 * 4), d_b(48 * 4);
            DeviceBuffer d_qc(2048 * 2), d_kc(2048 * 2), d_vc(6144 * 2);
            DeviceBuffer d_o(6144 * 2), d_on(6144 * 2), d_ong(6144 * 2);
            DeviceBuffer d_ao(5120 * 2), d_x1(5120 * 2), d_mh(5120 * 2);
            DeviceBuffer d_gv(17408 * 2), d_uv(17408 * 2), d_act(17408 * 2);
            DeviceBuffer d_d(5120 * 2);
            BRK(p, rmsnorm, s,
                { const Tensor xx(d_cur.p, DType::BF16, {5120, 1}); Tensor hh(d_h.p, DType::BF16, {5120, 1}); ops::rmsnorm(xx, w.an, kEps, false, hh, s); });
            BRK(p, fwht, s,
                { const Tensor hh(d_h.p, DType::BF16, {5120, 1}); Tensor hr(d_hr.p, DType::BF16, {5120, 1}); ops::t2_fwht_sign(hh, m.signs_for(5120), hr, s); });
            {
                const Tensor hr(d_hr.p, DType::BF16, {5120, 1});
                Tensor qkv(d_qkv.p, DType::BF16, {10240, 1});
                Tensor z(d_z.p, DType::BF16, {6144, 1});
                CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                if (use_old) {
                    ops::detail::launch_t2_gemv(hr, w.qkv, qkv, s);
                    CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                    acc(p, &Probes::lin_qkv, s);
                    CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                    ops::detail::launch_t2_gemv(hr, w.gate, z, s);
                } else {
                    ops::linear(hr, w.qkv, qkv, s);
                    CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                    acc(p, &Probes::lin_qkv, s);
                    CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                    ops::linear(hr, w.gate, z, s);
                }
                CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                acc(p, &Probes::lin_gate, s);
            }
            DeviceArena ws_gate(std::max<std::size_t>(
                ops::gdn_gating_proj_workspace_capacity_bytes(48, 5120, 1, 1), 256));
            BRK(p, acts, s,
                {
                    const Tensor hh(d_h.p, DType::BF16, {5120, 1});
                    Tensor gg(d_g.p, DType::FP32, {48, 1});
                    Tensor bb(d_b.p, DType::FP32, {48, 1});
                    ops::gdn_gating_proj(hh, w.alpha, w.beta, w.sa, w.sdt,
                                         ops::GdnGateFormula::RawMultiply, ws_gate, gg, bb,
                                         m.device.execution_view());
                });
            BRK(p, conv, s,
                {
                    const Tensor qkv(d_qkv.p, DType::BF16, {10240, 1});
                    const Tensor wcw(m.conv_weight_of(il), DType::BF16, {10240, 4});
                    Tensor o0(d_qc.p, DType::BF16, {2048, 1});
                    Tensor o1(d_kc.p, DType::BF16, {2048, 1});
                    Tensor o2(d_vc.p, DType::BF16, {6144, 1});
                    Tensor csi(state.gdn[il].conv.p, DType::BF16, {10240, 3});
                    ops::causal_conv1d_silu_split(qkv, wcw, csi, csi, o0, o1, o2, s);
                });
            DeviceArena ws_gdn(std::max<std::size_t>(
                ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, 1, 1), 256));
            BRK(p, gdn, s,
                {
                    const Tensor qq(d_qc.p, DType::BF16, {128, 16, 1});
                    const Tensor kk(d_kc.p, DType::BF16, {128, 16, 1});
                    const Tensor vv(d_vc.p, DType::BF16, {128, 48, 1});
                    const Tensor gg(d_g.p, DType::FP32, {48, 1});
                    const Tensor bb(d_b.p, DType::FP32, {48, 1});
                    Tensor oo(d_o.p, DType::BF16, {128, 48, 1});
                    Tensor ssm(state.gdn[il].ssm.p, DType::FP32, {128, 128, 48});
                    ops::gated_delta_net(qq, kk, vv, gg, bb, kGdnScale, true, ws_gdn, ssm, oo, s);
                });
            BRK(p, rmsnorm, s,
                {
                    const Tensor oo(d_o.p, DType::BF16, {128, 48, 1});
                    const Tensor zz(d_z.p, DType::BF16, {128, 48, 1});
                    Tensor onn(d_on.p, DType::BF16, {128, 48, 1});
                    const Tensor wsn(w.sn.data, DType::BF16, {128});
                    ops::gated_rmsnorm(oo, wsn, zz, kEps, onn, s);
                });
            BRK(p, acts, s,
                {
                    const Tensor ung(d_on.p, DType::BF16, {6144, 1});
                    Tensor grouped(d_ong.p, DType::BF16, {6144, 1});
                    ops::t2_gdn_v_group(ung, 16, 48, 128, grouped, s);
                });
            BRK(p, fwht, s,
                { const Tensor onn(d_ong.p, DType::BF16, {6144, 1}); Tensor onr(d_hr.p, DType::BF16, {6144, 1}); ops::t2_fwht_sign(onn, m.signs_for(6144), onr, s); });
            {
                const Tensor onr(d_hr.p, DType::BF16, {6144, 1});
                Tensor ao(d_ao.p, DType::BF16, {5120, 1});
                CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                if (use_old) {
                    ops::detail::launch_t2_gemv(onr, w.so, ao, s);
                } else {
                    ops::linear(onr, w.so, ao, s);
                }
                CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                acc(p, &Probes::lin_so, s);
            }
            BRK(p, resid, s,
                {
                    CUDA_CHECK(cudaMemcpyAsync(d_x1.p, d_cur.p, 5120 * 2, cudaMemcpyDeviceToDevice, s));
                    const Tensor ao(d_ao.p, DType::BF16, {5120, 1});
                    Tensor x1(d_x1.p, DType::BF16, {5120, 1});
                    ops::residual_add(ao, x1, s);
                });
            BRK(p, rmsnorm, s,
                { const Tensor x1(d_x1.p, DType::BF16, {5120, 1}); const Tensor wpost(w.pn.data, DType::BF16, {5120}); Tensor mh(d_mh.p, DType::BF16, {5120, 1}); ops::rmsnorm(x1, wpost, kEps, false, mh, s); });
            BRK(p, fwht, s,
                { const Tensor mh(d_mh.p, DType::BF16, {5120, 1}); Tensor mhr(d_hr.p, DType::BF16, {5120, 1}); ops::t2_fwht_sign(mh, m.signs_for(5120), mhr, s); });
            {
                const Tensor mhr(d_hr.p, DType::BF16, {5120, 1});
                Tensor gv(d_gv.p, DType::BF16, {17408, 1});
                Tensor uv(d_uv.p, DType::BF16, {17408, 1});
                CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                if (use_old) {
                    ops::detail::launch_t2_gemv(mhr, w.fg, gv, s);
                    CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                    acc(p, &Probes::lin_fg, s);
                    CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                    ops::detail::launch_t2_gemv(mhr, w.fu, uv, s);
                } else {
                    ops::linear(mhr, w.fg, gv, s);
                    CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                    acc(p, &Probes::lin_fg, s);
                    CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                    ops::linear(mhr, w.fu, uv, s);
                }
                CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                acc(p, &Probes::lin_fu, s);
            }
            BRK(p, acts, s,
                { const Tensor gv(d_gv.p, DType::BF16, {17408, 1}); const Tensor uv(d_uv.p, DType::BF16, {17408, 1}); Tensor act(d_act.p, DType::BF16, {17408, 1}); ops::silu_mul(gv, uv, act, s); });
            BRK(p, fwht, s,
                { const Tensor act(d_act.p, DType::BF16, {17408, 1}); Tensor actr(d_hr.p, DType::BF16, {17408, 1}); ops::t2_fwht_sign(act, m.signs_for(17408), actr, s); });
            {
                const Tensor actr(d_hr.p, DType::BF16, {17408, 1});
                Tensor dd(d_d.p, DType::BF16, {5120, 1});
                CUDA_CHECK(cudaEventRecord(p.ev.a, s));
                if (use_old) {
                    ops::detail::launch_t2_gemv(actr, w.fd, dd, s);
                } else {
                    ops::linear(actr, w.fd, dd, s);
                }
                CUDA_CHECK(cudaEventRecord(p.ev.b, s));
                acc(p, &Probes::lin_fd, s);
            }
            BRK(p, resid, s,
                {
                    CUDA_CHECK(cudaMemcpyAsync(d_nxt.p, d_x1.p, 5120 * 2, cudaMemcpyDeviceToDevice, s));
                    const Tensor dd(d_d.p, DType::BF16, {5120, 1});
                    Tensor lo(d_nxt.p, DType::BF16, {5120, 1});
                    ops::residual_add(dd, lo, s);
                });
        }
    }
    CUDA_CHECK(cudaMemcpy(out_bf16, d_cur.p, 5120 * 2, cudaMemcpyDeviceToDevice));
}

// Head: norm + fwht + 248320 linear, bracketed.
void head_rep(Bq2Model& m, void* h, Probes& p, cudaStream_t s, void* lg) {
    DeviceBuffer d_n(5120 * 2), d_nr(5120 * 2);
    BRK(p, rmsnorm, s,
        { const Tensor xx(h, DType::BF16, {5120, 1}); Tensor nn(d_n.p, DType::BF16, {5120, 1}); ops::rmsnorm(xx, m.final_norm, kEps, false, nn, s); });
    BRK(p, fwht, s,
        { const Tensor nn(d_n.p, DType::BF16, {5120, 1}); Tensor nr(d_nr.p, DType::BF16, {5120, 1}); ops::t2_fwht_sign(nn, m.signs_for(5120), nr, s); });
    {
        const Tensor nr(d_nr.p, DType::BF16, {5120, 1});
        Tensor lo(static_cast<void*>(lg), DType::BF16, {248320, 1});
        CUDA_CHECK(cudaEventRecord(p.ev.a, s));
        ops::linear(nr, m.head_t2, lo, s);
        CUDA_CHECK(cudaEventRecord(p.ev.b, s));
        acc(p, &Probes::lin_head, s);
    }
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
        Bq2SeqState state(8);
        DeviceBuffer d_rows1(4);
        {
            const std::int32_t zero = 0;
            upload_i32(&zero, d_rows1.p, 1);
        }
        const Tensor table_rows(d_rows1.p, DType::I32, {1});
        // Fixed T=1 input: embed [9419] once, reuse across reps.
        DeviceBuffer d_x(5120 * 2);
        {
            const std::int32_t id = 9419;
            embed_lookup(model, &id, 1, d_x.p, stream);
            device.synchronize();
        }
        DeviceBuffer d_pos(4), d_pos3(12);
        {
            const std::int32_t z = 0;
            upload_i32(&z, d_pos.p, 1);
            upload_i32(&z, d_pos3.p, 1);
            upload_i32(&z, d_pos3.p + 4, 1);
            upload_i32(&z, d_pos3.p + 8, 1);
        }
        // NOTE: positions/table_rows tensors are built per call inside
        // decode_rep from these buffers (views only, no allocs).
        for (bool use_old : {false, true}) {
            Probes p;
            // Warmup (untimed, also settles clocks/allocators).
            {
                DeviceBuffer d_h(5120 * 2), d_lg(248320 * 2);
                state.reset();
                decode_rep(model, state, table_rows, d_x.p, use_old, p, stream, d_h.p);
                head_rep(model, d_h.p, p, stream, d_lg.p);
                device.synchronize();
            }
            Probes q;
            std::printf("== use_old=%d ==\n", use_old ? 1 : 0);
            auto t0 = std::chrono::steady_clock::now();
            const int reps = 6;
            for (int r = 0; r < reps; ++r) {
                DeviceBuffer d_h(5120 * 2), d_lg(248320 * 2);
                state.reset();
                decode_rep(model, state, table_rows, d_x.p, use_old, q, stream, d_h.p);
                head_rep(model, d_h.p, q, stream, d_lg.p);
                device.synchronize();
            }
            const double wall =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            std::printf("wall %.1fms for %d reps (%.2f ms/rep)\n", wall, reps, wall / reps);
            q.brief(use_old ? "OLD" : "T1-A");
        }
        std::printf("OK D2_TIMING\n");
        return 0;
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    }
}
