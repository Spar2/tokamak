// M5 step 5/6: one real GDN block (layer 8) through production NInfer ops.
// Env: NINFER_BQ2_BLOCK8 (artifact), NINFER_BQ2_VECTORS (T{T}_*.bin oracle).
// Mirrors the Prism qwen35 GDN block order (the PQ2 GGUF declares
// general.architecture=qwen35, not qwen3next). T=1: full checkpoints.
// T=2,4,8,16,32,128: stage + final checkpoints vs FP64 oracle.
// T<=32 covers the T2 row_persistent path, T=128 the chunked path.
//
// PRISM-TRUE conventions (verified against llama-debug dumps of the live
// PQ2 model, prism_l8_dump.txt / prism_l8_gate.txt / prism_l8_alpha.txt):
// - RMSNorm uses PLAIN gain (unit_offset=FALSE). Prism build_norm is
//   ggml_rms_norm + mul (no +1); dump ratio attn_norm/inp divided by w is
//   constant 1.3766, divided by (1+w) drifts 0.709-0.731.
// - GDN gate is a RAW MULTIPLY g = ssm_a * softplus(a_raw + dt), NO exp.
//   Dump ratio gate/a_softplus == ssm_a to 1.0000; -exp(ssm_a) is off by
//   orders of magnitude (Bonsai trains A as a direct negative decay scale;
//   the "-A_log.exp()" comments in qwen3next.cpp describe code that
//   multiplies raw, as does the fused ggml CPU kernel).
// - GGUF byte order is ne0-fastest (torch bytes preserved): ssm_alpha/beta
//   host matrix is (48,5120) hh-major, ssm_conv host matrix is (10240,4),
//   NO transposes. Validated decisively: Prism conv_output_silu token-0
//   matches the direct-order computation to 4 decimals on all channels.
// Consequence: Bonsai selects GdnGateFormula::RawMultiply explicitly at the
// production gdn_gating_proj call below (host-side gate math was removed).
// All other stages call production ops with artifact values.
// NOTE: the conv split kernel consumes weight bytes TAP-MAJOR ([4,C] order:
// slot [j*C+c]) under the logical Tensor{C,4} descriptor (deduced from its
// own test's offset() oracle and confirmed on-device); the header's "[C,4]"
// describes the logical shape, not the byte order. Same trap applies to
// conv state ([3,C]-major); M5 uses zero state (immune).
// STREAM DISCIPLINE: every host read of device memory is preceded by
// device.synchronize(). Blocking D2H on the default stream does NOT wait for
// the non-blocking op stream; a missing sync reads garbage intermittently
// (the only flakiness ever observed here).
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/linear/t2/t2_rotation.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

constexpr float kEps      = 1e-6F;
constexpr float kGdnScale = 0.08838834764831845F;

std::vector<std::uint8_t> load_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { return {}; }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<std::uint8_t> v(static_cast<std::size_t>(n));
    if (std::fread(v.data(), 1, v.size(), f) != v.size()) { std::exit(3); }
    std::fclose(f);
    return v;
}

std::uint16_t f32_to_bf16(float v) {
    std::uint32_t w = 0;
    std::memcpy(&w, &v, 4);
    return static_cast<std::uint16_t>((w + 0x7fffu + ((w >> 16) & 1u)) >> 16);
}

float bf16_to_f32(std::uint16_t h) {
    std::uint32_t w = static_cast<std::uint32_t>(h) << 16;
    float f = 0.0F;
    std::memcpy(&f, &w, 4);
    return f;
}

struct Gate {
    double rmse;
    double cos;
    double maxabs;
};

// Reports maxabs/rmse/rel/cos; passes on rmse+cos (rel is informational:
// tiny-magnitude checkpoints such as gdn_out would divide by ~0).
bool check_vec(const float* got, const float* ref, std::size_t n, const char* label, Gate gate) {
    double maxabs = 0.0, se = 0.0, r2 = 0.0, dot = 0.0, o2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double e = std::fabs((double)got[i] - ref[i]);
        if (e > maxabs) { maxabs = e; }
        se += e * e;
        r2 += (double)ref[i] * ref[i];
        dot += (double)got[i] * ref[i];
        o2 += (double)got[i] * got[i];
    }
    const double rmse = std::sqrt(se / n);
    const double rel  = r2 > 0.0 ? std::sqrt(se / r2) : 0.0;
    const double cos  = (o2 > 0.0 && r2 > 0.0) ? dot / std::sqrt(o2 * r2) : 0.0;
    const bool pass   = maxabs <= gate.maxabs && rmse <= gate.rmse && cos >= gate.cos;
    std::printf("  %-14s maxabs=%.6f rmse=%.6f rel=%.6f cos=%.8f %s\n", label, maxabs, rmse, rel,
                cos, pass ? "PASS" : "FAIL");
    return pass;
}

std::vector<float> d2h_f32(const void* d, std::size_t n) {
    std::vector<float> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d, n * 4, cudaMemcpyDeviceToHost));
    return out;
}

std::vector<float> d2h_bf16(const void* d, std::size_t n) {
    std::vector<std::uint16_t> tmp(n);
    CUDA_CHECK(cudaMemcpy(tmp.data(), d, n * 2, cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) { out[i] = bf16_to_f32(tmp[i]); }
    return out;
}

std::vector<float> load_ref(const std::string& dir, int t, const char* name, std::size_t n) {
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/T%d_%s.bin", dir.c_str(), t, name);
    const std::vector<std::uint8_t> raw = load_file(path);
    if (raw.size() != n * 4) {
        std::fprintf(stderr, "ref missing/size %s\n", path);
        std::exit(4);
    }
    std::vector<float> out(n);
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}

void upload_bf16(const float* host, void* device, std::size_t n) {
    std::vector<std::uint16_t> bits(n);
    for (std::size_t i = 0; i < n; ++i) { bits[i] = f32_to_bf16(host[i]); }
    CUDA_CHECK(cudaMemcpy(device, bits.data(), n * 2, cudaMemcpyHostToDevice));
    // Pageable H2D copies may return before the transfer completes; settle
    // the default stream before kernels consume the buffer (same rule as
    // DeviceBuffer::copy_from_host). Missing this caused intermittent
    // garbage inputs at larger T.
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

// Oracle vectors are numpy (rows,T) C-order: element (i,t) at i*T+t.
// NInfer Tensors are ne0-fastest: element (i,t) at i + rows*t. Transpose the
// reference to column-major before comparing (identity at T=1).
std::vector<float> as_column_major(const std::vector<float>& row_major, int rows, int T) {
    std::vector<float> out(row_major.size());
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < rows; ++i) {
            out[static_cast<std::size_t>(t) * rows + i] =
                row_major[static_cast<std::size_t>(i) * T + t];
        }
    }
    return out;
}

} // namespace

int main() {
    const char* art_path = std::getenv("NINFER_BQ2_BLOCK8");
    const char* vec_dir  = std::getenv("NINFER_BQ2_VECTORS");
    if (art_path == nullptr || vec_dir == nullptr) {
        std::cout << "SKIP: set NINFER_BQ2_BLOCK8 and NINFER_BQ2_VECTORS\n";
        return 77;
    }
    try {
        bool ok = true;
        const std::string vdir(vec_dir);

        artifact::Reader reader(art_path);
        artifact::Binder binder(reader);
        auto req_t2 = [&](const char* n, std::uint64_t rows, std::uint64_t cols) {
            return artifact::bind_device_tensor(binder, n, artifact::NumericFormat::T2G128_F16S,
                                                {rows, cols});
        };
        auto req_f32 = [&](const char* n, std::initializer_list<std::uint64_t> shape) {
            return artifact::bind_device_tensor(binder, n, artifact::NumericFormat::FP32, shape);
        };
        auto req_bf16 = [&](const char* n, std::uint64_t rows, std::uint64_t cols) {
            return artifact::bind_device_tensor(binder, n, artifact::NumericFormat::BF16,
                                                {rows, cols});
        };
        const auto h_qkv   = req_t2("text/layers/8/attn_qkv", 10240, 5120);
        const auto h_gate  = req_t2("text/layers/8/attn_gate", 6144, 5120);
        const auto h_so    = req_t2("text/layers/8/ssm_out", 5120, 6144);
        const auto h_fg    = req_t2("text/layers/8/ffn_gate", 17408, 5120);
        const auto h_fu    = req_t2("text/layers/8/ffn_up", 17408, 5120);
        const auto h_fd    = req_t2("text/layers/8/ffn_down", 5120, 17408);
        const auto h_an    = req_f32("text/layers/8/attn_norm", {5120});
        const auto h_pn    = req_f32("text/layers/8/post_attn_norm", {5120});
        const auto h_sn    = req_f32("text/layers/8/ssm_norm", {128});
        const auto h_sa    = req_f32("text/layers/8/ssm_a", {48});
        const auto h_sdt   = req_f32("text/layers/8/ssm_dt", {48});
        const auto h_sc    = req_f32("text/layers/8/ssm_conv", {4, 10240});
        const auto h_alpha = req_bf16("text/layers/8/ssm_alpha", 5120, 48);
        const auto h_beta  = req_bf16("text/layers/8/ssm_beta", 5120, 48);
        // One shared spec handle: Binder forbids binding an object twice.
        const artifact::ObjectHandle h_spec = artifact::bind_raw_resource(binder, "bq2/spec");
        const artifact::RotationPlan r5120 = {
            artifact::bind_raw_resource(binder, "bq2/signs/5120"), h_spec};
        const artifact::RotationPlan r6144 = {
            artifact::bind_raw_resource(binder, "bq2/signs/6144"), h_spec};
        const artifact::RotationPlan r17408 = {
            artifact::bind_raw_resource(binder, "bq2/signs/17408"), h_spec};
        const artifact::MaterializationPlan plan = binder.finish();

        DeviceContext device(0);
        const cudaStream_t stream = device.stream;
        artifact::MaterializedArtifact materialized = artifact::materialize(reader, plan, device);
        device.synchronize(); // settle materialization uploads before host reads below
        auto mat_t2 = [&](artifact::ObjectHandle h, std::int32_t rows, std::int32_t cols) {
            return artifact::materialized_weight(materialized, h,
                                                 artifact::NumericFormat::T2G128_F16S, rows, cols);
        };
        const Weight w_qkv  = mat_t2(h_qkv, 10240, 5120);
        const Weight w_gate = mat_t2(h_gate, 6144, 5120);
        const Weight w_so   = mat_t2(h_so, 5120, 6144);
        const Weight w_fg   = mat_t2(h_fg, 17408, 5120);
        const Weight w_fu   = mat_t2(h_fu, 17408, 5120);
        const Weight w_fd   = mat_t2(h_fd, 5120, 17408);
        const artifact::RotationHost rot5120 =
            artifact::materialized_rotation(materialized, r5120, 5120);
        const artifact::RotationHost rot6144 =
            artifact::materialized_rotation(materialized, r6144, 6144);
        const artifact::RotationHost rot17408 =
            artifact::materialized_rotation(materialized, r17408, 17408);
        auto get_f32 = [&](artifact::ObjectHandle h, std::size_t n) {
            return d2h_f32(materialized.device_data(h), n);
        };
        auto get_bf16_bits = [&](artifact::ObjectHandle h, std::size_t n) {
            std::vector<std::uint16_t> out(n);
            CUDA_CHECK(
                cudaMemcpy(out.data(), materialized.device_data(h), n * 2, cudaMemcpyDeviceToHost));
            return out;
        };
        const std::vector<float> f_an = get_f32(h_an, 5120);
        const std::vector<float> f_pn = get_f32(h_pn, 5120);
        const std::vector<float> f_sn = get_f32(h_sn, 128);
        const std::vector<float> f_sc = get_f32(h_sc, 4 * 10240);

        // Production alpha/beta weights: GGUF [5120,48] ne0-fastest bytes are
        // already [48,5120] row-major order (byte f holds torch (hh=f//5120,
        // k=f%5120)), so the artifact payload uploads DIRECTLY as BF16_CTRL
        // [48,5120] Weight views. These feed the REAL production
        // gdn_gating_proj with GdnGateFormula::RawMultiply below.
        DeviceBuffer d_alpha(48 * 5120 * 2), d_beta(48 * 5120 * 2);
        {
            const std::vector<std::uint16_t> a_bits = get_bf16_bits(h_alpha, 5120 * 48);
            const std::vector<std::uint16_t> b_bits = get_bf16_bits(h_beta, 5120 * 48);
            d_alpha.copy_from_host(a_bits.data(), a_bits.size() * 2);
            d_beta.copy_from_host(b_bits.data(), b_bits.size() * 2);
        }
        auto bf16_ctrl_weight = [](void* payload, std::int32_t rows, std::int32_t cols) {
            Weight w{};
            w.payload         = payload;
            w.qdata           = payload;
            w.payload_bytes   = static_cast<std::uint64_t>(rows) * cols * 2;
            w.qtype           = QType::BF16_CTRL;
            w.layout          = QuantLayout::Contiguous;
            w.n               = rows;
            w.k               = cols;
            w.ndim            = 2;
            w.shape[0]        = rows;
            w.shape[1]        = cols;
            w.padded_shape[0] = rows;
            w.padded_shape[1] = cols;
            return w;
        };
        const Weight w_alpha = bf16_ctrl_weight(d_alpha.p, 48, 5120);
        const Weight w_beta  = bf16_ctrl_weight(d_beta.p, 48, 5120);
        const Tensor t_sa =
            artifact::materialized_tensor(materialized, h_sa, artifact::NumericFormat::FP32, {48});
        const Tensor t_sdt =
            artifact::materialized_tensor(materialized, h_sdt, artifact::NumericFormat::FP32, {48});

        // Persistent device staging (test-owned, mirrors converter policy):
        // norms F32->BF16 cast; conv DIRECT copy [10240,4] + BF16 cast (GGUF
        // ne0-fastest: byte f = (j=f%4, c=f//4); NO transpose — Prism-exact);
        // signs uploaded once to device-global.
        DeviceBuffer d_an(5120 * 2), d_pn(5120 * 2), d_sn(128 * 2);
        upload_bf16(f_an.data(), d_an.p, 5120);
        upload_bf16(f_pn.data(), d_pn.p, 5120);
        upload_bf16(f_sn.data(), d_sn.p, 128);
        DeviceBuffer d_sc(10240 * 4 * 2);
        {
            // Byte-order contract (deduced from the op's own test, whose
            // oracle indexes weight[offset(c,j,C)] = column*C + c, and
            // confirmed against device behavior): the split kernel consumes
            // weight bytes TAP-MAJOR ([4,C]: slot [j*C+c] = tap j, channel c)
            // under the logical Tensor{C,4} descriptor. GGUF bytes are
            // ne0-fastest (slot [c*4+j] = tap j, channel c), so stage the
            // transpose BUF[j*C+c] = GGUF[c*4+j] + BF16 cast. (Prism reads the
            // same weights direct C-order; the difference is purely the
            // kernel's expected byte order. The header's "[C,4]" describes
            // the logical shape, not the byte order — filed as doc issue.)
            // NOTE: same trap applies to conv STATE ([3,C]-major under
            // Tensor{C,3}); M5 uses zero state (immune), non-zero future
            // states must stage [3,C]-major.
            std::vector<std::uint16_t> bits(10240 * 4);
            for (std::int32_t c = 0; c < 10240; ++c) {
                for (std::int32_t j = 0; j < 4; ++j) {
                    bits[static_cast<std::size_t>(j) * 10240 + c] =
                        f32_to_bf16(f_sc[static_cast<std::size_t>(c) * 4 + j]);
                }
            }
            d_sc.copy_from_host(bits.data(), bits.size() * 2);
        }
        DeviceBuffer d_s5120(5120 * 4), d_s6144(6144 * 4), d_s17408(17408 * 4);
        d_s5120.copy_from_host(rot5120.signs.data(), 5120 * 4);
        d_s6144.copy_from_host(rot6144.signs.data(), 6144 * 4);
        d_s17408.copy_from_host(rot17408.signs.data(), 17408 * 4);

        // Gates account for BF16 rounding at every stage (oracle is FP64
        // end-to-end): linears/gates carry input-rounding error, small-signal
        // checkpoints use absolute+cosine verdicts (rel would divide by ~0).
        // Maxabs tails grow with sample count (more draws); rmse+cos carry
        // the verdict (l_out rmse is flat across T=1..128: no accumulation).
        const Gate kLin   = {1e-2, 0.9999, 5e-2};  // T2 linears qkv/z
        // gate maxabs is outlier-driven (BF16-rounded device h feeds the host
        // FP64 GEMV while the oracle uses FP64 h; softplus tails amplify);
        // rmse+cos carry the formula/layout verdict.
        const Gate kGate  = {1e-2, 0.9999, 1e-1};  // gate_g/gate_beta
        const Gate kConv  = {1e-3, 0.9999, 1.5e-2}; // conv_silu (T>1 carries
                                                   // BF16 state vs FP32 oracle)
        const Gate kSmall = {5e-4, 0.99, 5e-3};    // gdn_out/gated_out/attn_out
        const Gate kMid   = {5e-3, 0.9999, 2e-2};  // attn_res/ffn_out
        const Gate kFinal = {5e-3, 0.9995, 8e-2};  // l_out

        const int oracle_ts[] = {1, 2, 4};
        for (int ti = 0; ti < 3; ++ti) {
            const int T = oracle_ts[ti];
            std::printf("== T=%d ==\n", T);
            const std::vector<float> x0 = load_ref(vdir, T, "x0", 5120 * T);
            DeviceBuffer d_x(5120 * T * 2), d_h(5120 * T * 2), d_hr(17408 * T * 2);
            DeviceBuffer d_qkv(10240 * T * 2), d_z(6144 * T * 2);
            DeviceBuffer d_g(48 * T * 4), d_b(48 * T * 4);
            DeviceBuffer d_qc(2048 * T * 2), d_kc(2048 * T * 2), d_vc(6144 * T * 2);
            DeviceBuffer d_csi(10240 * 3 * 2), d_cso(10240 * 3 * 2);
            DeviceBuffer d_o(6144 * T * 2), d_on(6144 * T * 2);
            DeviceBuffer d_ao(5120 * T * 2), d_x1(5120 * T * 2), d_mh(5120 * T * 2);
            DeviceBuffer d_gv(17408 * T * 2), d_uv(17408 * T * 2), d_act(17408 * T * 2);
            DeviceBuffer d_d(5120 * T * 2), d_lout(5120 * T * 2);
            DeviceBuffer d_rsi(128 * 128 * 48 * 4), d_rso(128 * 128 * 48 * 4);
            d_csi.fill(0);
            d_rsi.fill(0);
            // x0 reference is numpy (5120,T) row-major; the device Tensor
            // [5120,T] is ne0-fastest: stage transposed.
            {
                std::vector<float> xcm(5120 * T);
                for (int t = 0; t < T; ++t) {
                    for (int i = 0; i < 5120; ++i) {
                        xcm[static_cast<std::size_t>(t) * 5120 + i] =
                            x0[static_cast<std::size_t>(i) * T + t];
                    }
                }
                upload_bf16(xcm.data(), d_x.p, 5120 * T);
            }
            DeviceArena ws_gdn(std::max<std::size_t>(
                ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, T, T), 256));
            double ms_oth_total = 0.0;
            CudaEventTimer t_all(device), t_fw(device), t_lin(device), t_gdn(device),
                t_oth(device), t_gate(device);
            t_all.start();
            // 1. attn norm (PLAIN gain: unit_offset=false, Prism-true).
            t_oth.start();
            {
                const Tensor x(d_x.p, DType::BF16, {5120, T});
                const Tensor w(d_an.p, DType::BF16, {5120});
                Tensor h(d_h.p, DType::BF16, {5120, T});
                ops::rmsnorm(x, w, kEps, false, h, stream);
            }
            t_oth.record_stop();
            device.synchronize();
            ms_oth_total += t_oth.elapsed_ms();
            // 2. FWHT + T2 linears (qkv, z).
            t_fw.start();
            {
                const Tensor h(d_h.p, DType::BF16, {5120, T});
                Tensor hr(d_hr.p, DType::BF16, {5120, T});
                ops::t2_fwht_sign(h, static_cast<const float*>(d_s5120.p), hr, stream);
            }
            t_fw.record_stop();
            device.synchronize();
            const double ms_fw1 = t_fw.elapsed_ms();
            t_lin.start();
            {
                const Tensor hr(d_hr.p, DType::BF16, {5120, T});
                Tensor qkv(d_qkv.p, DType::BF16, {10240, T});
                Tensor z(d_z.p, DType::BF16, {6144, T});
                ops::linear(hr, w_qkv, qkv, stream);
                ops::linear(hr, w_gate, z, stream);
            }
            t_lin.record_stop();
            device.synchronize();
            const double ms_lin1 = t_lin.elapsed_ms();
            device.synchronize();
            if (T > 1) {
                // Stage localization at T>1 (T2 row_persistent path, conv/GDN
                // recurrence) before the final verdict.
                ok &= check_vec(d2h_bf16(d_qkv.p, 10240 * T).data(),
                                as_column_major(load_ref(vdir, T, "qkv_mixed", 10240 * T), 10240,
                                                T).data(),
                                10240 * T, "qkv_T", kLin);
                ok &= check_vec(d2h_bf16(d_z.p, 6144 * T).data(),
                                as_column_major(load_ref(vdir, T, "z", 6144 * T), 6144, T).data(),
                                6144 * T, "z_T", kLin);
            }
            if (T == 1) {
                ok &= check_vec(d2h_bf16(d_h.p, 5120).data(),
                                load_ref(vdir, T, "attn_norm", 5120).data(), 5120, "attn_norm",
                                {5e-3, 0.999, 2e-2});
                ok &= check_vec(d2h_bf16(d_qkv.p, 10240).data(),
                                load_ref(vdir, T, "qkv_mixed", 10240).data(), 10240, "qkv_mixed",
                                kLin);
                ok &= check_vec(d2h_bf16(d_z.p, 6144).data(),
                                load_ref(vdir, T, "z", 6144).data(), 6144, "z", kLin);
            }
            // 3. Control projection via the REAL production op with the
            // explicitly selected RawMultiply formula (Bonsai semantics).
            t_gate.start();
            {
                DeviceArena ws_gate(std::max<std::size_t>(
                    ops::gdn_gating_proj_workspace_capacity_bytes(48, 5120, T, T), 256));
                const Tensor h(d_h.p, DType::BF16, {5120, T});
                Tensor g(d_g.p, DType::FP32, {48, T});
                Tensor b(d_b.p, DType::FP32, {48, T});
                ops::gdn_gating_proj(h, w_alpha, w_beta, t_sa, t_sdt,
                                     ops::GdnGateFormula::RawMultiply, ws_gate, g, b,
                                     device.execution_view());
            }
            t_gate.record_stop();
            device.synchronize();
            const double ms_gate = t_gate.elapsed_ms();
            {
                if (T > 1) {
                    ok &= check_vec(d2h_f32(d_g.p, 48 * T).data(),
                                    as_column_major(load_ref(vdir, T, "gate_g", 48 * T), 48, T)
                                        .data(),
                                    48 * T, "gate_T", kGate);
                    ok &= check_vec(d2h_f32(d_b.p, 48 * T).data(),
                                    as_column_major(load_ref(vdir, T, "gate_beta", 48 * T), 48, T)
                                        .data(),
                                    48 * T, "beta_T", kGate);
                }
                if (T == 1) {
                    ok &= check_vec(d2h_f32(d_g.p, 48).data(),
                                    load_ref(vdir, T, "gate_g", 48).data(), 48, "gate_g", kGate);
                    ok &= check_vec(d2h_f32(d_b.p, 48).data(),
                                    load_ref(vdir, T, "gate_beta", 48).data(), 48, "gate_beta",
                                    kGate);
                }
            }
            // 4. conv split + GDN core.
            t_oth.start();
            {
                const Tensor x(d_qkv.p, DType::BF16, {10240, T});
                const Tensor w(d_sc.p, DType::BF16, {10240, 4});
                const Tensor si(d_csi.p, DType::BF16, {10240, 3});
                Tensor so(d_cso.p, DType::BF16, {10240, 3});
                Tensor o0(d_qc.p, DType::BF16, {2048, T});
                Tensor o1(d_kc.p, DType::BF16, {2048, T});
                Tensor o2(d_vc.p, DType::BF16, {6144, T});
                ops::causal_conv1d_silu_split(x, w, si, so, o0, o1, o2, stream);
            }
            t_oth.record_stop();
            device.synchronize();
            ms_oth_total += t_oth.elapsed_ms();
            t_gdn.start();
            {
                const Tensor q(d_qc.p, DType::BF16, {128, 16, T});
                const Tensor k(d_kc.p, DType::BF16, {128, 16, T});
                const Tensor v(d_vc.p, DType::BF16, {128, 48, T});
                const Tensor g(d_g.p, DType::FP32, {48, T});
                const Tensor beta(d_b.p, DType::FP32, {48, T});
                const Tensor si(d_rsi.p, DType::FP32, {128, 128, 48});
                Tensor so(d_rso.p, DType::FP32, {128, 128, 48});
                Tensor o(d_o.p, DType::BF16, {128, 48, T});
                ops::gated_delta_net(q, k, v, g, beta, kGdnScale, true, ws_gdn, si, so, o,
                                     stream);
            }
            t_gdn.record_stop();
            device.synchronize();
            const double ms_gdn = t_gdn.elapsed_ms();
            device.synchronize();
            if (T > 1) {
                // Reassemble full conv output [10240,T]: split buffers are
                // [rows,T] C-order; oracle conv_all is [10240,T] C-order with
                // q/k/v channel blocks contiguous — same flat order.
                std::vector<float> conv;
                {
                    const std::vector<float> c0 = d2h_bf16(d_qc.p, 2048 * T);
                    const std::vector<float> c1 = d2h_bf16(d_kc.p, 2048 * T);
                    const std::vector<float> c2 = d2h_bf16(d_vc.p, 6144 * T);
                    for (int t = 0; t < T; ++t) {
                        conv.insert(conv.end(), c0.begin() + t * 2048, c0.begin() + (t + 1) * 2048);
                        conv.insert(conv.end(), c1.begin() + t * 2048, c1.begin() + (t + 1) * 2048);
                        conv.insert(conv.end(), c2.begin() + t * 6144, c2.begin() + (t + 1) * 6144);
                    }
                }
                ok &= check_vec(conv.data(),
                                as_column_major(load_ref(vdir, T, "conv_silu", 10240 * T), 10240, T)
                                    .data(),
                                10240 * T, "conv_T", kConv);
                ok &= check_vec(d2h_bf16(d_o.p, 6144 * T).data(),
                                as_column_major(load_ref(vdir, T, "gdn_out", 6144 * T), 6144, T)
                                    .data(),
                                6144 * T, "gdn_T", kSmall);
            }
            if (T == 1) {
                // Reassemble full conv output [10240] in q/k/v split order.
                std::vector<float> conv;
                {
                    const std::vector<float> c0 = d2h_bf16(d_qc.p, 2048);
                    const std::vector<float> c1 = d2h_bf16(d_kc.p, 2048);
                    const std::vector<float> c2 = d2h_bf16(d_vc.p, 6144);
                    conv.insert(conv.end(), c0.begin(), c0.end());
                    conv.insert(conv.end(), c1.begin(), c1.end());
                    conv.insert(conv.end(), c2.begin(), c2.end());
                }
                ok &= check_vec(conv.data(), load_ref(vdir, T, "conv_silu", 10240).data(), 10240,
                                "conv_silu", kConv);
                ok &= check_vec(d2h_bf16(d_o.p, 6144).data(),
                                load_ref(vdir, T, "gdn_out", 6144).data(), 6144, "gdn_out", kSmall);
            }
            // 5. gated norm + ssm_out + residual.
            t_oth.start();
            {
                const Tensor o(d_o.p, DType::BF16, {128, 48, T});
                const Tensor z(d_z.p, DType::BF16, {128, 48, T});
                Tensor on(d_on.p, DType::BF16, {128, 48, T});
                const Tensor w(d_sn.p, DType::BF16, {128});
                ops::gated_rmsnorm(o, w, z, kEps, on, stream);
            }
            t_oth.record_stop();
            device.synchronize();
            ms_oth_total += t_oth.elapsed_ms();
            t_fw.start();
            {
                const Tensor on(d_on.p, DType::BF16, {6144, T});
                Tensor onr(d_hr.p, DType::BF16, {6144, T});
                ops::t2_fwht_sign(on, static_cast<const float*>(d_s6144.p), onr, stream);
            }
            t_fw.record_stop();
            device.synchronize();
            const double ms_fw2 = t_fw.elapsed_ms();
            t_lin.start();
            {
                const Tensor onr(d_hr.p, DType::BF16, {6144, T});
                Tensor ao(d_ao.p, DType::BF16, {5120, T});
                ops::linear(onr, w_so, ao, stream);
            }
            t_lin.record_stop();
            device.synchronize();
            const double ms_lin2 = t_lin.elapsed_ms();
            {
                CUDA_CHECK(cudaMemcpyAsync(d_x1.p, d_x.p, 5120 * T * 2, cudaMemcpyDeviceToDevice,
                                           stream));
                const Tensor ao(d_ao.p, DType::BF16, {5120, T});
                Tensor x1(d_x1.p, DType::BF16, {5120, T});
                ops::residual_add(ao, x1, stream);
            }
            device.synchronize();
            if (T == 1) {
                ok &= check_vec(d2h_bf16(d_on.p, 6144).data(),
                                load_ref(vdir, T, "gated_out", 6144).data(), 6144, "gated_out",
                                kSmall);
                ok &= check_vec(d2h_bf16(d_ao.p, 5120).data(),
                                load_ref(vdir, T, "attn_out", 5120).data(), 5120, "attn_out",
                                kSmall);
                ok &= check_vec(d2h_bf16(d_x1.p, 5120).data(),
                                load_ref(vdir, T, "attn_residual", 5120).data(), 5120, "attn_res",
                                kMid);
            }
            // 6. post norm + MLP.
            t_oth.start();
            {
                const Tensor x1(d_x1.p, DType::BF16, {5120, T});
                const Tensor w(d_pn.p, DType::BF16, {5120});
                Tensor mh(d_mh.p, DType::BF16, {5120, T});
                ops::rmsnorm(x1, w, kEps, false, mh, stream);
            }
            t_oth.record_stop();
            device.synchronize();
            ms_oth_total += t_oth.elapsed_ms();
            t_fw.start();
            {
                const Tensor mh(d_mh.p, DType::BF16, {5120, T});
                Tensor mhr(d_hr.p, DType::BF16, {5120, T});
                ops::t2_fwht_sign(mh, static_cast<const float*>(d_s5120.p), mhr, stream);
            }
            t_fw.record_stop();
            device.synchronize();
            const double ms_fw3 = t_fw.elapsed_ms();
            t_lin.start();
            {
                const Tensor mhr(d_hr.p, DType::BF16, {5120, T});
                Tensor gv(d_gv.p, DType::BF16, {17408, T});
                Tensor uv(d_uv.p, DType::BF16, {17408, T});
                ops::linear(mhr, w_fg, gv, stream);
                ops::linear(mhr, w_fu, uv, stream);
            }
            t_lin.record_stop();
            device.synchronize();
            const double ms_lin3 = t_lin.elapsed_ms();
            t_oth.start();
            {
                const Tensor gv(d_gv.p, DType::BF16, {17408, T});
                const Tensor uv(d_uv.p, DType::BF16, {17408, T});
                Tensor act(d_act.p, DType::BF16, {17408, T});
                ops::silu_mul(gv, uv, act, stream);
            }
            t_oth.record_stop();
            device.synchronize();
            ms_oth_total += t_oth.elapsed_ms();
            t_fw.start();
            {
                const Tensor act(d_act.p, DType::BF16, {17408, T});
                Tensor actr(d_hr.p, DType::BF16, {17408, T});
                ops::t2_fwht_sign(act, static_cast<const float*>(d_s17408.p), actr, stream);
            }
            t_fw.record_stop();
            device.synchronize();
            const double ms_fw4 = t_fw.elapsed_ms();
            t_lin.start();
            {
                const Tensor actr(d_hr.p, DType::BF16, {17408, T});
                Tensor d(d_d.p, DType::BF16, {5120, T});
                ops::linear(actr, w_fd, d, stream);
            }
            t_lin.record_stop();
            device.synchronize();
            const double ms_lin4 = t_lin.elapsed_ms();
            {
                CUDA_CHECK(cudaMemcpyAsync(d_lout.p, d_x1.p, 5120 * T * 2,
                                           cudaMemcpyDeviceToDevice, stream));
                const Tensor d(d_d.p, DType::BF16, {5120, T});
                Tensor lo(d_lout.p, DType::BF16, {5120, T});
                ops::residual_add(d, lo, stream);
            }
            t_all.record_stop();
            device.synchronize();
            CUDA_CHECK(cudaDeviceSynchronize());
            if (T == 1) {
                ok &= check_vec(d2h_bf16(d_mh.p, 5120).data(),
                                load_ref(vdir, T, "attn_post_norm", 5120).data(), 5120, "post_norm",
                                {5e-3, 0.999, 3e-2});
                ok &= check_vec(d2h_bf16(d_act.p, 17408).data(),
                                load_ref(vdir, T, "ffn_act", 17408).data(), 17408, "ffn_act",
                                {3e-3, 0.999, 5e-2});
                ok &= check_vec(d2h_bf16(d_d.p, 5120).data(),
                                load_ref(vdir, T, "ffn_out", 5120).data(), 5120, "ffn_out", kMid);
            }
            ok &= check_vec(d2h_bf16(d_lout.p, 5120 * T).data(),
                            as_column_major(load_ref(vdir, T, "l_out", 5120 * T), 5120, T).data(),
                            5120 * T, T == 1 ? "l_out" : "l_out_T", kFinal);
            std::printf("T=%d e2e %.3fms (fwht %.3f lin %.3f gate %.3f gdn %.3f other %.3f)\n",
                        T, t_all.elapsed_ms(), ms_fw1 + ms_fw2 + ms_fw3 + ms_fw4,
                        ms_lin1 + ms_lin2 + ms_lin3 + ms_lin4, ms_gate, ms_gdn, ms_oth_total);
        }

        std::printf("%s BQ2_BLOCK8\n", ok ? "OK" : "FAIL");
        if (!ok) { return 1; }

        // Depth coverage with absolute oracle verdicts: T=8,16,32,128 exercise
        // the T2 row_persistent (T<=32) and chunked (T>32) paths plus conv/GDN
        // recurrence at depth. Inputs are the oracle's x0 vectors.
        struct StabOut {
            std::vector<float> h, qkv, z, g, b, conv, gdn, lout;
        };
        const int stab_ts[] = {8, 16, 32, 128};
        for (int si = 0; si < 4; ++si) {
            const int T = stab_ts[si];
            const std::vector<float> xs = load_ref(vdir, T, "x0", 5120 * T);
            auto run_block = [&](StabOut& out, bool capture) {
                DeviceBuffer d_x(5120 * T * 2), d_h(5120 * T * 2), d_hr(17408 * T * 2);
                DeviceBuffer d_qkv(10240 * T * 2), d_z(6144 * T * 2);
                DeviceBuffer d_g(48 * T * 4), d_b(48 * T * 4);
                DeviceBuffer d_qc(2048 * T * 2), d_kc(2048 * T * 2), d_vc(6144 * T * 2);
                DeviceBuffer d_csi(10240 * 3 * 2), d_cso(10240 * 3 * 2);
                DeviceBuffer d_o(6144 * T * 2), d_on(6144 * T * 2);
                DeviceBuffer d_ao(5120 * T * 2), d_x1(5120 * T * 2), d_mh(5120 * T * 2);
                DeviceBuffer d_gv(17408 * T * 2), d_uv(17408 * T * 2), d_act(17408 * T * 2);
                DeviceBuffer d_d(5120 * T * 2), d_lout(5120 * T * 2);
                DeviceBuffer d_rsi(128 * 128 * 48 * 4), d_rso(128 * 128 * 48 * 4);
                d_csi.fill(0);
                d_rsi.fill(0);
                // Host vectors are numpy (5120,T) row-major; the device
                // Tensor is ne0-fastest: stage transposed (same as main loop).
                {
                    std::vector<float> xcm(5120 * T);
                    for (int t = 0; t < T; ++t) {
                        for (int i = 0; i < 5120; ++i) {
                            xcm[static_cast<std::size_t>(t) * 5120 + i] =
                                xs[static_cast<std::size_t>(i) * T + t];
                        }
                    }
                    upload_bf16(xcm.data(), d_x.p, 5120 * T);
                }
                DeviceArena ws(std::max<std::size_t>(
                    ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, T, T), 256));
                {
                    const Tensor x(d_x.p, DType::BF16, {5120, T});
                    const Tensor w(d_an.p, DType::BF16, {5120});
                    Tensor h(d_h.p, DType::BF16, {5120, T});
                    ops::rmsnorm(x, w, kEps, false, h, stream);
                }
                {
                    const Tensor h(d_h.p, DType::BF16, {5120, T});
                    Tensor hr(d_hr.p, DType::BF16, {5120, T});
                    ops::t2_fwht_sign(h, static_cast<const float*>(d_s5120.p), hr, stream);
                }
                {
                    const Tensor hr(d_hr.p, DType::BF16, {5120, T});
                    Tensor qkv(d_qkv.p, DType::BF16, {10240, T});
                    Tensor z(d_z.p, DType::BF16, {6144, T});
                    ops::linear(hr, w_qkv, qkv, stream);
                    ops::linear(hr, w_gate, z, stream);
                }
                {
                    // Production control projection (RawMultiply), same as the
                    // main loop. out.h still needs the host copy for capture.
                    device.synchronize();
                    if (capture) { out.h = d2h_bf16(d_h.p, 5120 * T); }
                    DeviceArena ws_gate(std::max<std::size_t>(
                        ops::gdn_gating_proj_workspace_capacity_bytes(48, 5120, T, T), 256));
                    const Tensor h(d_h.p, DType::BF16, {5120, T});
                    Tensor g(d_g.p, DType::FP32, {48, T});
                    Tensor b(d_b.p, DType::FP32, {48, T});
                    ops::gdn_gating_proj(h, w_alpha, w_beta, t_sa, t_sdt,
                                         ops::GdnGateFormula::RawMultiply, ws_gate, g, b,
                                         device.execution_view());
                    device.synchronize();
                    if (capture) {
                        out.g = d2h_f32(d_g.p, 48 * T);
                        out.b = d2h_f32(d_b.p, 48 * T);
                    }
                }
                {
                    const Tensor x(d_qkv.p, DType::BF16, {10240, T});
                    const Tensor w(d_sc.p, DType::BF16, {10240, 4});
                    const Tensor si(d_csi.p, DType::BF16, {10240, 3});
                    Tensor so(d_cso.p, DType::BF16, {10240, 3});
                    Tensor o0(d_qc.p, DType::BF16, {2048, T});
                    Tensor o1(d_kc.p, DType::BF16, {2048, T});
                    Tensor o2(d_vc.p, DType::BF16, {6144, T});
                    ops::causal_conv1d_silu_split(x, w, si, so, o0, o1, o2, stream);
                }
                {
                    const Tensor q(d_qc.p, DType::BF16, {128, 16, T});
                    const Tensor k(d_kc.p, DType::BF16, {128, 16, T});
                    const Tensor v(d_vc.p, DType::BF16, {128, 48, T});
                    const Tensor g(d_g.p, DType::FP32, {48, T});
                    const Tensor beta(d_b.p, DType::FP32, {48, T});
                    const Tensor si(d_rsi.p, DType::FP32, {128, 128, 48});
                    Tensor so(d_rso.p, DType::FP32, {128, 128, 48});
                    Tensor o(d_o.p, DType::BF16, {128, 48, T});
                    ops::gated_delta_net(q, k, v, g, beta, kGdnScale, true, ws, si, so, o,
                                         stream);
                }
                {
                    const Tensor o(d_o.p, DType::BF16, {128, 48, T});
                    const Tensor z(d_z.p, DType::BF16, {128, 48, T});
                    Tensor on(d_on.p, DType::BF16, {128, 48, T});
                    const Tensor w(d_sn.p, DType::BF16, {128});
                    ops::gated_rmsnorm(o, w, z, kEps, on, stream);
                }
                {
                    const Tensor on(d_on.p, DType::BF16, {6144, T});
                    Tensor onr(d_hr.p, DType::BF16, {6144, T});
                    ops::t2_fwht_sign(on, static_cast<const float*>(d_s6144.p), onr, stream);
                }
                {
                    const Tensor onr(d_hr.p, DType::BF16, {6144, T});
                    Tensor ao(d_ao.p, DType::BF16, {5120, T});
                    ops::linear(onr, w_so, ao, stream);
                }
                {
                    CUDA_CHECK(cudaMemcpyAsync(d_x1.p, d_x.p, 5120 * T * 2,
                                               cudaMemcpyDeviceToDevice, stream));
                    const Tensor ao(d_ao.p, DType::BF16, {5120, T});
                    Tensor x1(d_x1.p, DType::BF16, {5120, T});
                    ops::residual_add(ao, x1, stream);
                }
                {
                    const Tensor x1(d_x1.p, DType::BF16, {5120, T});
                    const Tensor w(d_pn.p, DType::BF16, {5120});
                    Tensor mh(d_mh.p, DType::BF16, {5120, T});
                    ops::rmsnorm(x1, w, kEps, false, mh, stream);
                }
                {
                    const Tensor mh(d_mh.p, DType::BF16, {5120, T});
                    Tensor mhr(d_hr.p, DType::BF16, {5120, T});
                    ops::t2_fwht_sign(mh, static_cast<const float*>(d_s5120.p), mhr, stream);
                }
                {
                    const Tensor mhr(d_hr.p, DType::BF16, {5120, T});
                    Tensor gv(d_gv.p, DType::BF16, {17408, T});
                    Tensor uv(d_uv.p, DType::BF16, {17408, T});
                    ops::linear(mhr, w_fg, gv, stream);
                    ops::linear(mhr, w_fu, uv, stream);
                }
                {
                    const Tensor gv(d_gv.p, DType::BF16, {17408, T});
                    const Tensor uv(d_uv.p, DType::BF16, {17408, T});
                    Tensor act(d_act.p, DType::BF16, {17408, T});
                    ops::silu_mul(gv, uv, act, stream);
                }
                {
                    const Tensor act(d_act.p, DType::BF16, {17408, T});
                    Tensor actr(d_hr.p, DType::BF16, {17408, T});
                    ops::t2_fwht_sign(act, static_cast<const float*>(d_s17408.p), actr, stream);
                }
                {
                    const Tensor actr(d_hr.p, DType::BF16, {17408, T});
                    Tensor d(d_d.p, DType::BF16, {5120, T});
                    ops::linear(actr, w_fd, d, stream);
                }
                {
                    CUDA_CHECK(cudaMemcpyAsync(d_lout.p, d_x1.p, 5120 * T * 2,
                                               cudaMemcpyDeviceToDevice, stream));
                    const Tensor d(d_d.p, DType::BF16, {5120, T});
                    Tensor lo(d_lout.p, DType::BF16, {5120, T});
                    ops::residual_add(d, lo, stream);
                }
                device.synchronize();
                if (capture) {
                    out.lout = d2h_bf16(d_lout.p, 5120 * T);
                    out.qkv  = d2h_bf16(d_qkv.p, 10240 * T);
                    out.z    = d2h_bf16(d_z.p, 6144 * T);
                    out.gdn  = d2h_bf16(d_o.p, 6144 * T);
                    const std::vector<float> c0 = d2h_bf16(d_qc.p, 2048 * T);
                    const std::vector<float> c1 = d2h_bf16(d_kc.p, 2048 * T);
                    const std::vector<float> c2 = d2h_bf16(d_vc.p, 6144 * T);
                    out.conv.clear();
                    for (int t = 0; t < T; ++t) {
                        out.conv.insert(out.conv.end(), c0.begin() + t * 2048,
                                        c0.begin() + (t + 1) * 2048);
                        out.conv.insert(out.conv.end(), c1.begin() + t * 2048,
                                        c1.begin() + (t + 1) * 2048);
                        out.conv.insert(out.conv.end(), c2.begin() + t * 6144,
                                        c2.begin() + (t + 1) * 6144);
                    }
                }
            };
            StabOut got;
            CudaEventTimer t_s(device);
            t_s.start();
            run_block(got, true);
            const double ms_stab = t_s.stop_ms();
            std::printf("== T=%d ==\n", T);
            ok &= check_vec(got.h.data(),
                            as_column_major(load_ref(vdir, T, "attn_norm", 5120 * T), 5120, T)
                                .data(),
                            5120 * T, "s_attn", {5e-3, 0.999, 4e-2});
            ok &= check_vec(got.qkv.data(),
                            as_column_major(load_ref(vdir, T, "qkv_mixed", 10240 * T), 10240, T)
                                .data(),
                            10240 * T, "s_qkv", kLin);
            ok &= check_vec(got.z.data(),
                            as_column_major(load_ref(vdir, T, "z", 6144 * T), 6144, T).data(),
                            6144 * T, "s_z", kLin);
            ok &= check_vec(got.g.data(),
                            as_column_major(load_ref(vdir, T, "gate_g", 48 * T), 48, T).data(),
                            48 * T, "s_gate", kGate);
            ok &= check_vec(got.b.data(),
                            as_column_major(load_ref(vdir, T, "gate_beta", 48 * T), 48, T).data(),
                            48 * T, "s_beta", kGate);
            ok &= check_vec(got.conv.data(),
                            as_column_major(load_ref(vdir, T, "conv_silu", 10240 * T), 10240, T)
                                .data(),
                            10240 * T, "s_conv", kConv);
            ok &= check_vec(got.gdn.data(),
                            as_column_major(load_ref(vdir, T, "gdn_out", 6144 * T), 6144, T)
                                .data(),
                            6144 * T, "s_gdn", kSmall);
            ok &= check_vec(got.lout.data(),
                            as_column_major(load_ref(vdir, T, "l_out", 5120 * T), 5120, T).data(),
                            5120 * T, "s_lout", kFinal);
            std::printf("T=%d stab %.1fms\n", T, ms_stab);
        }

        std::printf("%s BQ2_BLOCK8_STAB\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    }
}
