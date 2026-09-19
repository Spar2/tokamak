// M5 step 6/6: one real full-attention block (layer 7) through production NInfer ops.
// Env: NINFER_BQ2_BLOCK7 (artifact), NINFER_BQ2_BLOCK7_VECTORS (T{T}_*.bin oracle).
// Mirrors the Prism qwen35 full-attention block order (GGUF arch=qwen35).
// n_head=24, n_kv=4, head_dim=256, rotary_dim=64, theta=1e7, scale=1/16.
// T=1: full checkpoints. T=2,4,8,16,32,128: stage + final checkpoints.
//
// PRISM-TRUE conventions (verified against llama-debug dumps, prism_l7_*.txt):
// - RMSNorm PLAIN gain everywhere (attn/q/k/post norms): MUL nodes, no +1.
// - Q projection outputs [Q;gate] interleaved per head: rows [512h:512h+256]
//   are Q_h, rows [512h+256:512h+512] are gate_h. K/V are head-major [256,4].
// - RoPE: split-half NeoX, R=64, phi(i,t)=t*theta^(-2i/64), dims [64,256)
//   unchanged. Text positions are (t,t,t[,0]) per llama-graph.cpp, so MRoPE
//   sections vs pair%3 give identical angles; NInfer rope covers the 24/4
//   D256/R64 MRoPE domain natively (tested).
// - Attention: scale=1/sqrt(256) (no attention.scale key), full causal over a
//   fresh paged BF16 cache (no SWA keys in GGUF), GQA group 6.
// - gate_sigmoid = sigmoid(gate); attn_gated = pregate * gate_sigmoid (MUL);
//   production sigmoid_mul(gate, attn) in place. attn_output = Wo @ FWHT.
// - FFN is the same dense SwiGLU T2 path as layer 8.
// - l_out = ffn_out + attn_residual (post_ffn == l_out validated by checkpoint).
//
// Staging notes: Q/gate de-interleave is a host bit-exact BF16 repack (test
// staging; T2 row ranges are per-head strided, not viewable). K/V buffers are
// already head-major and viewed directly. Norms are F32->BF16 converter-cast
// previews. T2 weights come straight from the artifact.
// STREAM DISCIPLINE (lesson from layer 8): every host read is preceded by
// device.synchronize(); every H2D upload settles the default stream.
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/softmax_attention.h"
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

constexpr float kEps       = 1e-6F;
constexpr float kAttnScale = 0.0625F;
constexpr float kRopeTheta = 1.0e7F;
constexpr int kRotaryDim   = 64;

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
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

void upload_i32(const std::int32_t* host, void* device, std::size_t n) {
    CUDA_CHECK(cudaMemcpy(device, host, n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

// Oracle vectors are numpy (rows,T) C-order: element (i,t) at i*T+t.
// NInfer Tensors are ne0-fastest: element (i,t) at i + rows*t.
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
    const char* art_path = std::getenv("NINFER_BQ2_BLOCK7");
    const char* vec_dir  = std::getenv("NINFER_BQ2_BLOCK7_VECTORS");
    if (art_path == nullptr || vec_dir == nullptr) {
        std::cout << "SKIP: set NINFER_BQ2_BLOCK7 and NINFER_BQ2_BLOCK7_VECTORS\n";
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
        const auto h_q    = req_t2("text/layers/7/attn_q", 12288, 5120);
        const auto h_k    = req_t2("text/layers/7/attn_k", 1024, 5120);
        const auto h_v    = req_t2("text/layers/7/attn_v", 1024, 5120);
        const auto h_wo   = req_t2("text/layers/7/attn_out", 5120, 6144);
        const auto h_fg   = req_t2("text/layers/7/ffn_gate", 17408, 5120);
        const auto h_fu   = req_t2("text/layers/7/ffn_up", 17408, 5120);
        const auto h_fd   = req_t2("text/layers/7/ffn_down", 5120, 17408);
        const auto h_an   = req_f32("text/layers/7/attn_norm", {5120});
        const auto h_pn   = req_f32("text/layers/7/post_attn_norm", {5120});
        const auto h_qn   = req_f32("text/layers/7/attn_q_norm", {256});
        const auto h_kn   = req_f32("text/layers/7/attn_k_norm", {256});
        const artifact::ObjectHandle h_spec = artifact::bind_raw_resource(binder, "bq2/spec");
        const artifact::RotationPlan r5120 = {artifact::bind_raw_resource(binder, "bq2/signs/5120"),
                                              h_spec};
        const artifact::RotationPlan r6144 = {artifact::bind_raw_resource(binder, "bq2/signs/6144"),
                                              h_spec};
        const artifact::RotationPlan r17408 = {
            artifact::bind_raw_resource(binder, "bq2/signs/17408"), h_spec};
        const artifact::MaterializationPlan plan = binder.finish();

        DeviceContext device(0);
        const cudaStream_t stream = device.stream;
        artifact::MaterializedArtifact materialized = artifact::materialize(reader, plan, device);
        device.synchronize();
        auto mat_t2 = [&](artifact::ObjectHandle h, std::int32_t rows, std::int32_t cols) {
            return artifact::materialized_weight(materialized, h,
                                                 artifact::NumericFormat::T2G128_F16S, rows, cols);
        };
        const Weight w_q  = mat_t2(h_q, 12288, 5120);
        const Weight w_k  = mat_t2(h_k, 1024, 5120);
        const Weight w_v  = mat_t2(h_v, 1024, 5120);
        const Weight w_wo = mat_t2(h_wo, 5120, 6144);
        const Weight w_fg = mat_t2(h_fg, 17408, 5120);
        const Weight w_fu = mat_t2(h_fu, 17408, 5120);
        const Weight w_fd = mat_t2(h_fd, 5120, 17408);
        const artifact::RotationHost rot5120 =
            artifact::materialized_rotation(materialized, r5120, 5120);
        const artifact::RotationHost rot6144 =
            artifact::materialized_rotation(materialized, r6144, 6144);
        const artifact::RotationHost rot17408 =
            artifact::materialized_rotation(materialized, r17408, 17408);

        auto get_f32 = [&](artifact::ObjectHandle h, std::size_t n) {
            return d2h_f32(materialized.device_data(h), n);
        };
        const std::vector<float> f_an = get_f32(h_an, 5120);
        const std::vector<float> f_pn = get_f32(h_pn, 5120);
        const std::vector<float> f_qn = get_f32(h_qn, 256);
        const std::vector<float> f_kn = get_f32(h_kn, 256);

        DeviceBuffer d_an(5120 * 2), d_pn(5120 * 2), d_qnw(256 * 2), d_knw(256 * 2);
        upload_bf16(f_an.data(), d_an.p, 5120);
        upload_bf16(f_pn.data(), d_pn.p, 5120);
        upload_bf16(f_qn.data(), d_qnw.p, 256);
        upload_bf16(f_kn.data(), d_knw.p, 256);
        DeviceBuffer d_s5120(5120 * 4), d_s6144(6144 * 4), d_s17408(17408 * 4);
        d_s5120.copy_from_host(rot5120.signs.data(), 5120 * 4);
        d_s6144.copy_from_host(rot6144.signs.data(), 6144 * 4);
        d_s17408.copy_from_host(rot17408.signs.data(), 17408 * 4);

        // Attention-path noise floors (measured, structural, all cos>=0.99997):
        // Q/K norms + rope carry input BF16-rounding direction noise (~0.4%);
        // attention adds reduction noise, FP16 V-cache quantization, and rare
        // softmax tie-flips (large maxabs outliers, flat rmse). rmse+cos carry
        // every verdict; maxabs only guards against catastrophic divergence.
        const Gate kLin   = {1e-2, 0.9999, 5e-2};  // T2 linears
        const Gate kNormA = {8e-3, 0.999, 4e-2};   // Q/K norms + rope
        const Gate kAttn  = {1e-2, 0.999, 1e-1};   // attention core (pregate/gated)
        const Gate kMid   = {5e-3, 0.9999, 5e-2};  // attn_res/ffn_out
        const Gate kFinal = {8e-3, 0.9995, 2e-1};  // l_out
        const ops::AttentionHeadGeometry kGeo{256, 24, 4};

        const int oracle_ts[] = {1, 2, 4, 8, 16, 32, 128};
        for (int ti = 0; ti < 7; ++ti) {
            const int T = oracle_ts[ti];
            std::printf("== T=%d ==\n", T);
            const std::vector<float> x0 = load_ref(vdir, T, "x0", 5120 * T);
            // Transpose numpy (5120,T) row-major to ne0-fastest for upload.
            std::vector<float> xcm(5120 * T);
            for (int t = 0; t < T; ++t) {
                for (int i = 0; i < 5120; ++i) {
                    xcm[static_cast<std::size_t>(t) * 5120 + i] =
                        x0[static_cast<std::size_t>(i) * T + t];
                }
            }
            DeviceBuffer d_x(5120 * T * 2), d_h(5120 * T * 2), d_hr(17408 * T * 2);
            DeviceBuffer d_qfull(12288 * T * 2), d_k(1024 * T * 2), d_v(1024 * T * 2);
            DeviceBuffer d_q(6144 * T * 2), d_gate(6144 * T * 2);
            DeviceBuffer d_qn(6144 * T * 2), d_kn(1024 * T * 2);
            DeviceBuffer d_attn(6144 * T * 2);
            DeviceBuffer d_ao(5120 * T * 2), d_x1(5120 * T * 2), d_mh(5120 * T * 2);
            DeviceBuffer d_gv(17408 * T * 2), d_uv(17408 * T * 2), d_act(17408 * T * 2);
            DeviceBuffer d_d(5120 * T * 2), d_lout(5120 * T * 2);
            DeviceBuffer d_pos3(T * 3 * 4), d_pos(T * 4), d_rows(4);
            upload_bf16(xcm.data(), d_x.p, 5120 * T);
            {
                // Rope positions are [T,3] column-major (axis outer, like the
                // rope op's own test): slot [a*T+t] = t (text: all axes equal).
                std::vector<std::int32_t> p3(T * 3), p1(T);
                for (int t = 0; t < T; ++t) {
                    p3[t] = p3[T + t] = p3[2 * T + t] = t;
                    p1[t]                             = t;
                }
                upload_i32(p3.data(), d_pos3.p, T * 3);
                upload_i32(p1.data(), d_pos.p, T);
                const std::int32_t zero = 0;
                upload_i32(&zero, d_rows.p, 1);
            }
            CudaEventTimer t_all(device), t_fw(device), t_lin(device), t_attn(device),
                t_oth(device);
            double ms_oth_total = 0.0;
            t_all.start();
            // 1. attn norm (PLAIN gain).
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
            // 2. FWHT + T2 Q/K/V projections.
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
                Tensor qf(d_qfull.p, DType::BF16, {12288, T});
                Tensor kf(d_k.p, DType::BF16, {1024, T});
                Tensor vf(d_v.p, DType::BF16, {1024, T});
                ops::linear(hr, w_q, qf, stream);
                ops::linear(hr, w_k, kf, stream);
                ops::linear(hr, w_v, vf, stream);
            }
            t_lin.record_stop();
            device.synchronize();
            const double ms_lin1 = t_lin.elapsed_ms();
            // 3. Q/gate de-interleave (host bit-exact BF16 repack, test staging:
            // Q rows are per-head strided, not viewable). K/V are head-major.
            t_oth.start();
            {
                const std::vector<std::uint16_t> qf_bits = [&] {
                    std::vector<std::uint16_t> tmp(12288 * T);
                    CUDA_CHECK(
                        cudaMemcpy(tmp.data(), d_qfull.p, tmp.size() * 2, cudaMemcpyDeviceToHost));
                    return tmp;
                }();
                std::vector<std::uint16_t> q_bits(6144 * T), gate_bits(6144 * T);
                for (int t = 0; t < T; ++t) {
                    for (int hh = 0; hh < 24; ++hh) {
                        for (int d = 0; d < 256; ++d) {
                            q_bits[static_cast<std::size_t>(t) * 6144 + hh * 256 + d] =
                                qf_bits[static_cast<std::size_t>(t) * 12288 + hh * 512 + d];
                            gate_bits[static_cast<std::size_t>(t) * 6144 + hh * 256 + d] =
                                qf_bits[static_cast<std::size_t>(t) * 12288 + hh * 512 + 256 + d];
                        }
                    }
                }
                CUDA_CHECK(cudaMemcpy(d_q.p, q_bits.data(), q_bits.size() * 2,
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_gate.p, gate_bits.data(), gate_bits.size() * 2,
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaStreamSynchronize(nullptr));
            }
            t_oth.record_stop();
            device.synchronize();
            ms_oth_total += t_oth.elapsed_ms();
            // 4. Q/K norms (PLAIN gain) on head views.
            t_oth.start();
            {
                const Tensor q(d_q.p, DType::BF16, {256, 24, T});
                const Tensor w(d_qnw.p, DType::BF16, {256});
                Tensor qn(d_qn.p, DType::BF16, {256, 24, T});
                ops::rmsnorm(q, w, kEps, false, qn, stream);
            }
            {
                const Tensor k(d_k.p, DType::BF16, {256, 4, T});
                const Tensor w(d_knw.p, DType::BF16, {256});
                Tensor kn(d_kn.p, DType::BF16, {256, 4, T});
                ops::rmsnorm(k, w, kEps, false, kn, stream);
            }
            t_oth.record_stop();
            device.synchronize();
            ms_oth_total += t_oth.elapsed_ms();
            // Norm outputs are checked here: rope mutates d_qn/d_kn in place.
            if (T == 1) {
                ok &= check_vec(d2h_bf16(d_qn.p, 6144).data(),
                                load_ref(vdir, T, "q_normed", 6144).data(), 6144, "q_normed",
                                kNormA);
                ok &= check_vec(d2h_bf16(d_kn.p, 1024).data(),
                                load_ref(vdir, T, "k_normed", 1024).data(), 1024, "k_normed",
                                kNormA);
            }
            // 5. MRoPE in place (R=64, theta=1e7, positions (t,t,t)).
            t_oth.start();
            {
                const Tensor pos(d_pos3.p, DType::I32, {T, 3});
                Tensor q(d_qn.p, DType::BF16, {256, 24, T});
                Tensor k(d_kn.p, DType::BF16, {256, 4, T});
                ops::rope(pos, kRotaryDim, kRopeTheta, q, k, stream);
            }
            t_oth.record_stop();
            device.synchronize();
            ms_oth_total += t_oth.elapsed_ms();
            if (T == 1) {
                ok &= check_vec(d2h_bf16(d_qn.p, 6144).data(),
                                load_ref(vdir, T, "rope_q", 6144).data(), 6144, "rope_q",
                                kNormA);
                ok &= check_vec(d2h_bf16(d_kn.p, 1024).data(),
                                load_ref(vdir, T, "rope_k", 1024).data(), 1024, "rope_k",
                                kNormA);
            }
            // 6. Causal GQA over a fresh paged BF16 cache (append form).
            // Cache planes are allocated BEFORE the timer (cudaMalloc cost is
            // not kernel time).
            const int pages = (T + 63) / 64;
            DeviceBuffer d_kp(256 * 64 * 4 * pages * 2), d_vp(256 * 64 * 4 * pages * 2);
            DeviceBuffer d_bt(pages * 1 * 4);
            d_kp.fill(0);
            d_vp.fill(0);
            t_attn.start();
            {
                std::vector<std::int32_t> bt(pages);
                for (int i = 0; i < pages; ++i) { bt[i] = i; }
                upload_i32(bt.data(), d_bt.p, pages);
                PagedKVBatchLayerView cache;
                cache.k_pages = Tensor(d_kp.p, DType::BF16, {256, 64, 4, pages});
                cache.v_pages = Tensor(d_vp.p, DType::FP16, {256, 64, 4, pages});
                cache.block_tables  = Tensor(d_bt.p, DType::I32, {pages, 1});
                cache.head_dim      = 256;
                cache.num_kv_heads  = 4;
                cache.storage       = KvCacheStorage::BFloat16;
                const Tensor q(d_qn.p, DType::BF16, {256, 24, T});
                const Tensor k(d_kn.p, DType::BF16, {256, 4, T});
                const Tensor v(d_v.p, DType::BF16, {256, 4, T});
                const Tensor pos(d_pos.p, DType::I32, {T});
                const Tensor rows(d_rows.p, DType::I32, {1});
                Tensor out(d_attn.p, DType::BF16, {256, 24, T});
                const Tensor empty;
                const ops::CausalAttentionExecutionEnvelope env{static_cast<std::uint32_t>(T),
                                                               static_cast<std::uint32_t>(T)};
                DeviceArena ws_attn(std::max<std::size_t>(
                    ops::causal_softmax_attention_workspace_capacity_bytes(
                        kGeo, KvCacheStorage::BFloat16, env, 1, T, T),
                    256));
                ops::causal_softmax_attention(q, k, v, pos, empty, rows, kGeo, kAttnScale, cache,
                                              env, ws_attn, out, stream);
            }
            t_attn.record_stop();
            device.synchronize();
            const double ms_attn = t_attn.elapsed_ms();
            if (T == 1) {
                // (q_normed/k_normed/rope_q/rope_k were checked before rope
                // mutated d_qn/d_kn; d_h/d_qfull/d_k/d_v are never mutated.)
                ok &= check_vec(d2h_bf16(d_h.p, 5120).data(),
                                load_ref(vdir, T, "attn_norm", 5120).data(), 5120, "attn_norm",
                                kNormA);
                ok &= check_vec(d2h_bf16(d_qfull.p, 12288).data(),
                                load_ref(vdir, T, "q_full", 12288).data(), 12288, "q_full", kLin);
                ok &= check_vec(d2h_bf16(d_k.p, 1024).data(),
                                load_ref(vdir, T, "k_full", 1024).data(), 1024, "k_full", kLin);
                ok &= check_vec(d2h_bf16(d_v.p, 1024).data(),
                                load_ref(vdir, T, "v", 1024).data(), 1024, "v", kLin);
                ok &= check_vec(d2h_bf16(d_attn.p, 6144).data(),
                                load_ref(vdir, T, "pregate", 6144).data(), 6144, "pregate", kAttn);
            } else {
                ok &= check_vec(d2h_bf16(d_qfull.p, 12288 * T).data(),
                                as_column_major(load_ref(vdir, T, "q_full", 12288 * T), 12288, T)
                                    .data(),
                                12288 * T, "q_full_T", kLin);
                ok &= check_vec(d2h_bf16(d_attn.p, 6144 * T).data(),
                                as_column_major(load_ref(vdir, T, "pregate", 6144 * T), 6144, T)
                                    .data(),
                                6144 * T, "pregate_T", kAttn);
            }
            // 7. Sigmoid gate (in place on a [6144,T] view) + ssm_out + residual.
            t_oth.start();
            {
                const Tensor gate(d_gate.p, DType::BF16, {6144, T});
                Tensor ax(d_attn.p, DType::BF16, {6144, T});
                ops::sigmoid_mul(gate, ax, stream);
            }
            t_oth.record_stop();
            device.synchronize();
            ms_oth_total += t_oth.elapsed_ms();
            t_fw.start();
            {
                const Tensor ax(d_attn.p, DType::BF16, {6144, T});
                Tensor axr(d_hr.p, DType::BF16, {6144, T});
                ops::t2_fwht_sign(ax, static_cast<const float*>(d_s6144.p), axr, stream);
            }
            t_fw.record_stop();
            device.synchronize();
            const double ms_fw2 = t_fw.elapsed_ms();
            t_lin.start();
            {
                const Tensor axr(d_hr.p, DType::BF16, {6144, T});
                Tensor ao(d_ao.p, DType::BF16, {5120, T});
                ops::linear(axr, w_wo, ao, stream);
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
                ok &= check_vec(d2h_bf16(d_attn.p, 6144).data(),
                                load_ref(vdir, T, "gated", 6144).data(), 6144, "gated", kAttn);
                ok &= check_vec(d2h_bf16(d_ao.p, 5120).data(),
                                load_ref(vdir, T, "attn_out", 5120).data(), 5120, "attn_out",
                                kLin);
                ok &= check_vec(d2h_bf16(d_x1.p, 5120).data(),
                                load_ref(vdir, T, "attn_residual", 5120).data(), 5120, "attn_res",
                                kMid);
            }
            // 8. post norm + MLP.
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
            if (T == 1) {
                ok &= check_vec(d2h_bf16(d_mh.p, 5120).data(),
                                load_ref(vdir, T, "attn_post_norm", 5120).data(), 5120, "post_norm",
                                kNormA);
                ok &= check_vec(d2h_bf16(d_act.p, 17408).data(),
                                load_ref(vdir, T, "ffn_act", 17408).data(), 17408, "ffn_act",
                                {3e-3, 0.999, 5e-2});
                ok &= check_vec(d2h_bf16(d_d.p, 5120).data(),
                                load_ref(vdir, T, "ffn_out", 5120).data(), 5120, "ffn_out", kMid);
            }
            ok &= check_vec(d2h_bf16(d_lout.p, 5120 * T).data(),
                            as_column_major(load_ref(vdir, T, "l_out", 5120 * T), 5120, T).data(),
                            5120 * T, T == 1 ? "l_out" : "l_out_T", kFinal);
            std::printf("T=%d e2e %.3fms (fwht %.3f lin %.3f attn %.3f other %.3f)\n", T,
                        t_all.elapsed_ms(), ms_fw1 + ms_fw2 + ms_fw3 + ms_fw4,
                        ms_lin1 + ms_lin2 + ms_lin3 + ms_lin4, ms_attn, ms_oth_total);
        }

        std::printf("%s BQ2_BLOCK7\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    }
}
