// M6 shared full-model harness: Bonsai-2 27B text model in NInfer ops.
// Loads the full text artifact (bonsai2_full_text.ninfer), binds all 64
// layers + endpoints + rotation resources, owns single-sequence persistent
// state (paged KV per full layer, conv+S slots per GDN layer), and executes
// embedding -> layer stack -> final norm -> head -> logits with greedy steps.
//
// Provenance: block equations mirror tests/ops/test_bq2_block7.cpp (full)
// and test_bq2_block8.cpp (GDN), themselves validated vs Prism-true oracles.
// Conventions (see those files): plain RMS gains (unit_offset=false),
// RawMultiply gates, GGUF ne0-fastest aux order, tap-major conv bytes,
// T2 RowSplit weights straight from artifact, signs device-global.
// Stream discipline: every host read preceded by device.synchronize(); every
// H2D upload settles the default stream.
#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating_proj.h"
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
#include <utility>
#include <vector>

namespace bq2full {

using namespace ninfer;

constexpr float kEps       = 1e-6F;
constexpr float kGdnScale  = 0.08838834764831845F;
constexpr float kAttnScale = 0.0625F;
constexpr float kRopeTheta = 1.0e7F;
constexpr int kRotaryDim   = 64;
constexpr int kVocabPhys   = 248320;
constexpr int kVocabValid  = 248077;

inline bool is_full_layer(int il) { return (il + 1) % 4 == 0; }

struct Gate {
    double rmse;
    double cos;
    double maxabs;
};

inline bool check_vec(const float* got, const float* ref, std::size_t n, const char* label,
                      Gate gate) {
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
    std::printf("  %-16s maxabs=%.6f rmse=%.6f rel=%.6f cos=%.8f %s\n", label, maxabs, rmse, rel,
                cos, pass ? "PASS" : "FAIL");
    return pass;
}

inline std::uint16_t f32_to_bf16(float v) {
    std::uint32_t w = 0;
    std::memcpy(&w, &v, 4);
    return static_cast<std::uint16_t>((w + 0x7fffu + ((w >> 16) & 1u)) >> 16);
}

inline float bf16_to_f32(std::uint16_t h) {
    std::uint32_t w = static_cast<std::uint32_t>(h) << 16;
    float f = 0.0F;
    std::memcpy(&f, &w, 4);
    return f;
}

inline std::vector<float> d2h_f32(DeviceContext& device, const void* d, std::size_t n) {
    device.synchronize();
    std::vector<float> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d, n * 4, cudaMemcpyDeviceToHost));
    return out;
}

inline std::vector<float> d2h_bf16(DeviceContext& device, const void* d, std::size_t n) {
    device.synchronize();
    std::vector<std::uint16_t> tmp(n);
    CUDA_CHECK(cudaMemcpy(tmp.data(), d, n * 2, cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) { out[i] = bf16_to_f32(tmp[i]); }
    return out;
}

inline void upload_bf16(const float* host, void* device, std::size_t n) {
    std::vector<std::uint16_t> bits(n);
    for (std::size_t i = 0; i < n; ++i) { bits[i] = f32_to_bf16(host[i]); }
    CUDA_CHECK(cudaMemcpy(device, bits.data(), n * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

inline void upload_i32(const std::int32_t* host, void* device, std::size_t n) {
    CUDA_CHECK(cudaMemcpy(device, host, n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

inline float f16_to_f32(std::uint16_t h) {
    std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    std::uint32_t exp  = (static_cast<std::uint32_t>(h) >> 10) & 0x1fu;
    std::uint32_t mant = static_cast<std::uint32_t>(h) & 0x03ffu;
    std::uint32_t w;
    if (exp == 0) {
        if (mant == 0) {
            w = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x03ffu;
            w = sign | ((e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        w = sign | 0x7f800000u | (mant << 13);
    } else {
        w = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f = 0.0F;
    std::memcpy(&f, &w, 4);
    return f;
}

// Decode T2 rows [row_begin, row_end) of an [N,K] NInfer row-split payload to
// FP64 (host test staging, exact). Layout: code plane + 256B pad + F16 scales.
inline std::vector<double> t2_decode_rows(const std::uint8_t* payload, int N, int row_begin,
                                          int row_end, int K) {
    const int G                 = K / 128;
    const std::size_t code_size = static_cast<std::size_t>(N) * G * 32;
    const std::size_t scale_off = (code_size + 255) / 256 * 256;
    std::vector<double> out(static_cast<std::size_t>(row_end - row_begin) * K);
    for (int n = row_begin; n < row_end; ++n) {
        for (int g = 0; g < G; ++g) {
            const std::size_t gi    = static_cast<std::size_t>(n) * G + g;
            const std::uint8_t* codes = payload + gi * 32;
            std::uint16_t sh = static_cast<std::uint16_t>(payload[scale_off + gi * 2] |
                                                          (payload[scale_off + gi * 2 + 1] << 8));
            const double sc = f16_to_f32(sh);
            for (int j = 0; j < 128; ++j) {
                const int code = (codes[j / 4] >> ((j % 4) * 2)) & 3;
                out[(static_cast<std::size_t>(n - row_begin)) * K + g * 128 + j] =
                    (code - 1) * sc;
            }
        }
    }
    return out;
}

inline std::vector<std::uint8_t> load_file(const std::string& path) {
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

inline std::vector<float> load_ref(const std::string& dir, const char* name, std::size_t n) {
    const std::string path = dir + "/" + name + ".bin";
    const std::vector<std::uint8_t> raw = load_file(path);
    if (raw.size() != n * 4) {
        std::fprintf(stderr, "ref missing/size %s\n", path.c_str());
        std::exit(4);
    }
    std::vector<float> out(n);
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}

// Prism dumps are numpy (rows,T) C-order: element (i,t) at i*T+t.
// NInfer Tensors are ne0-fastest: element (i,t) at i + rows*t.
inline std::vector<float> as_column_major(const std::vector<float>& row_major, int rows, int T) {
    std::vector<float> out(row_major.size());
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < rows; ++i) {
            out[static_cast<std::size_t>(t) * rows + i] =
                row_major[static_cast<std::size_t>(i) * T + t];
        }
    }
    return out;
}

// BF16_CTRL Weight view over a device buffer holding [rows,cols] row-major BF16.
inline Weight bf16_ctrl_weight(void* payload, std::int32_t rows, std::int32_t cols) {
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
}

struct GdnLayerW {
    Weight qkv, gate, so, fg, fu, fd;
    Tensor an, pn, sn, sa, sdt;
    Weight alpha, beta; // BF16_CTRL [48,5120], direct artifact bytes
};

struct FullLayerW {
    Weight q, k, v, wo, fg, fu, fd;
    Tensor an, pn, qn, kn;
};

struct Bq2Model {
    DeviceContext device{0};
    cudaStream_t stream = nullptr;
    artifact::Reader reader;
    artifact::MaterializedArtifact materialized;
    std::vector<GdnLayerW> gdn;   // indexed by il (empty for full layers)
    std::vector<FullLayerW> full; // indexed by il (empty for GDN layers)
    Weight embed_t2;              // T2 [248320,5120]
    Weight head_t2;               // T2 [248320,5120]
    Tensor final_norm;            // BF16 [5120]
    DeviceBuffer signs5120, signs6144, signs17408;

    Bq2Model(const char* art_path) : reader(art_path) {
        stream = device.stream;
        artifact::Binder binder(reader);
        auto req_t2 = [&](const char* n, std::uint64_t rows, std::uint64_t cols) {
            return artifact::bind_device_tensor(binder, n, artifact::NumericFormat::T2G128_F16S,
                                                {rows, cols});
        };
        // Norms/conv are BF16 in the full artifact (converter RNE-cast,
        // verified); ssm_a/ssm_dt stay F32 (consumed as FP32).
        auto req_f32 = [&](const char* n, std::initializer_list<std::uint64_t> shape) {
            return artifact::bind_device_tensor(binder, n, artifact::NumericFormat::FP32, shape);
        };
        auto req_bf16 = [&](const char* n, std::initializer_list<std::uint64_t> shape) {
            return artifact::bind_device_tensor(binder, n, artifact::NumericFormat::BF16, shape);
        };
        struct Handles {
            artifact::ObjectHandle qkv, gate, so, fg, fu, fd;
            artifact::ObjectHandle an, pn, extra[4];
            artifact::ObjectHandle ab[2];
            bool full = false;
        };
        std::vector<Handles> hs(64);
        char name[128];
        for (int il = 0; il < 64; ++il) {
            Handles h;
            h.full = is_full_layer(il);
            std::snprintf(name, sizeof(name), "text/layers/%d/ffn_gate", il);
            h.fg = req_t2(name, 17408, 5120);
            std::snprintf(name, sizeof(name), "text/layers/%d/ffn_up", il);
            h.fu = req_t2(name, 17408, 5120);
            std::snprintf(name, sizeof(name), "text/layers/%d/ffn_down", il);
            h.fd = req_t2(name, 5120, 17408);
            std::snprintf(name, sizeof(name), "text/layers/%d/attn_norm", il);
            h.an = req_bf16(name, {5120});
            std::snprintf(name, sizeof(name), "text/layers/%d/post_attn_norm", il);
            h.pn = req_bf16(name, {5120});
            if (h.full) {
                std::snprintf(name, sizeof(name), "text/layers/%d/attn_q", il);
                h.qkv = req_t2(name, 12288, 5120);
                std::snprintf(name, sizeof(name), "text/layers/%d/attn_k", il);
                h.gate = req_t2(name, 1024, 5120);
                std::snprintf(name, sizeof(name), "text/layers/%d/attn_v", il);
                h.so = req_t2(name, 1024, 5120);
                std::snprintf(name, sizeof(name), "text/layers/%d/attn_q_norm", il);
                h.extra[0] = req_bf16(name, {256});
                std::snprintf(name, sizeof(name), "text/layers/%d/attn_k_norm", il);
                h.extra[1] = req_bf16(name, {256});
                std::snprintf(name, sizeof(name), "text/layers/%d/attn_out", il);
                h.extra[2] = req_t2(name, 5120, 6144);
            } else {
                std::snprintf(name, sizeof(name), "text/layers/%d/attn_qkv", il);
                h.qkv = req_t2(name, 10240, 5120);
                std::snprintf(name, sizeof(name), "text/layers/%d/attn_gate", il);
                h.gate = req_t2(name, 6144, 5120);
                std::snprintf(name, sizeof(name), "text/layers/%d/ssm_out", il);
                h.so = req_t2(name, 5120, 6144);
                std::snprintf(name, sizeof(name), "text/layers/%d/ssm_norm", il);
                h.extra[0] = req_bf16(name, {128});
                std::snprintf(name, sizeof(name), "text/layers/%d/ssm_a", il);
                h.extra[1] = req_f32(name, {48});
                std::snprintf(name, sizeof(name), "text/layers/%d/ssm_dt", il);
                h.extra[2] = req_f32(name, {48});
                std::snprintf(name, sizeof(name), "text/layers/%d/ssm_conv", il);
                h.extra[3] = req_bf16(name, {4, 10240});
                std::snprintf(name, sizeof(name), "text/layers/%d/ssm_alpha", il);
                h.ab[0] = req_bf16(name, {5120, 48});
                std::snprintf(name, sizeof(name), "text/layers/%d/ssm_beta", il);
                h.ab[1] = req_bf16(name, {5120, 48});
            }
            hs[il] = h;
        }
        const auto h_emb  = req_t2("text/token_embedding", 248320, 5120);
        const auto h_head = req_t2("text/output_head", 248320, 5120);
        const auto h_fn   = req_bf16("text/final_norm", {5120});
        const artifact::ObjectHandle h_spec = artifact::bind_raw_resource(binder, "bq2/spec");
        const artifact::RotationPlan r5120 = {artifact::bind_raw_resource(binder, "bq2/signs/5120"),
                                              h_spec};
        const artifact::RotationPlan r6144 = {artifact::bind_raw_resource(binder, "bq2/signs/6144"),
                                              h_spec};
        const artifact::RotationPlan r17408 = {
            artifact::bind_raw_resource(binder, "bq2/signs/17408"), h_spec};
        const artifact::MaterializationPlan plan = binder.finish();
        materialized = artifact::materialize(reader, plan, device);
        device.synchronize();
        auto mat_t2 = [&](artifact::ObjectHandle h, std::int32_t rows, std::int32_t cols) {
            return artifact::materialized_weight(materialized, h,
                                                 artifact::NumericFormat::T2G128_F16S, rows, cols);
        };
        auto mat_bf16 = [&](artifact::ObjectHandle h, std::initializer_list<std::int32_t> sh) {
            return artifact::materialized_tensor(materialized, h, artifact::NumericFormat::BF16,
                                                 sh);
        };
        auto mat_f32 = [&](artifact::ObjectHandle h, std::initializer_list<std::int32_t> sh) {
            return artifact::materialized_tensor(materialized, h, artifact::NumericFormat::FP32,
                                                 sh);
        };
        gdn.resize(64);
        full.resize(64);
        for (int il = 0; il < 64; ++il) {
            const Handles& h = hs[il];
            if (h.full) {
                FullLayerW w;
                w.q  = mat_t2(h.qkv, 12288, 5120);
                w.k  = mat_t2(h.gate, 1024, 5120);
                w.v  = mat_t2(h.so, 1024, 5120);
                w.wo = mat_t2(h.extra[2], 5120, 6144);
                w.fg = mat_t2(h.fg, 17408, 5120);
                w.fu = mat_t2(h.fu, 17408, 5120);
                w.fd = mat_t2(h.fd, 5120, 17408);
                // Norms are BF16 in artifact (converter-cast, verified).
                w.an = mat_bf16(h.an, {5120});
                w.pn = mat_bf16(h.pn, {5120});
                w.qn = mat_bf16(h.extra[0], {256});
                w.kn = mat_bf16(h.extra[1], {256});
                full[il] = w;
            } else {
                GdnLayerW w;
                w.qkv = mat_t2(h.qkv, 10240, 5120);
                w.gate = mat_t2(h.gate, 6144, 5120);
                w.so  = mat_t2(h.so, 5120, 6144);
                w.fg  = mat_t2(h.fg, 17408, 5120);
                w.fu  = mat_t2(h.fu, 17408, 5120);
                w.fd  = mat_t2(h.fd, 5120, 17408);
                w.an  = mat_bf16(h.an, {5120});
                w.pn  = mat_bf16(h.pn, {5120});
                w.sn  = mat_bf16(h.extra[0], {128});
                w.sa  = mat_f32(h.extra[1], {48});
                w.sdt = mat_f32(h.extra[2], {48});
                // alpha/beta: direct BF16 [5120,48] bytes are already [48,5120]
                // row-major (GGUF ne0-fastest); reinterpret as [48,5120] views.
                w.alpha = artifact::materialized_weight(materialized, h.ab[0],
                                                        artifact::NumericFormat::BF16, 48, 5120);
                w.beta  = artifact::materialized_weight(materialized, h.ab[1],
                                                        artifact::NumericFormat::BF16, 48, 5120);
                // Conv: artifact holds BF16 [4,10240]-order bytes (GGUF
                // ne0-fastest, converter-cast); stage tap-major [j*C+c]
                // (kernel contract; see causal_conv1d_silu.h).
                std::vector<std::uint16_t> cbits(4 * 10240);
                CUDA_CHECK(cudaMemcpy(cbits.data(), materialized.device_data(h.extra[3]),
                                      cbits.size() * 2, cudaMemcpyDeviceToHost));
                gdn_conv_bufs_.emplace_back(10240 * 4 * 2);
                std::vector<std::uint16_t> bits(10240 * 4);
                for (std::int32_t c = 0; c < 10240; ++c) {
                    for (std::int32_t j = 0; j < 4; ++j) {
                        bits[static_cast<std::size_t>(j) * 10240 + c] =
                            cbits[static_cast<std::size_t>(c) * 4 + j];
                    }
                }
                gdn_conv_bufs_.back().copy_from_host(bits.data(), bits.size() * 2);
                gdn[il] = w;
            }
        }
        embed_t2   = mat_t2(h_emb, 248320, 5120);
        head_t2    = mat_t2(h_head, 248320, 5120);
        final_norm = mat_bf16(h_fn, {5120});
        const artifact::RotationHost s5120 =
            artifact::materialized_rotation(materialized, r5120, 5120);
        const artifact::RotationHost s6144 =
            artifact::materialized_rotation(materialized, r6144, 6144);
        const artifact::RotationHost s17408 =
            artifact::materialized_rotation(materialized, r17408, 17408);
        signs5120  = DeviceBuffer(5120 * 4);
        signs6144  = DeviceBuffer(6144 * 4);
        signs17408 = DeviceBuffer(17408 * 4);
        signs5120.copy_from_host(s5120.signs.data(), 5120 * 4);
        signs6144.copy_from_host(s6144.signs.data(), 6144 * 4);
        signs17408.copy_from_host(s17408.signs.data(), 17408 * 4);
    }

    // Per-layer conv weight views (tap-major staging owned here).
    std::vector<DeviceBuffer> gdn_conv_bufs_;
    void* conv_weight_of(int il) {
        // gdn_conv_bufs_[k] aligns with the k-th GDN layer in index order.
        int k = 0;
        for (int i = 0; i < il; ++i) {
            if (!is_full_layer(i)) { ++k; }
        }
        return gdn_conv_bufs_[k].p;
    }

    const float* signs_for(int width) const {
        if (width == 5120) { return static_cast<const float*>(signs5120.p); }
        if (width == 6144) { return static_cast<const float*>(signs6144.p); }
        return static_cast<const float*>(signs17408.p);
    }
};

// Minimal single-sequence state owner: paged KV per full layer, conv+S slots
// per GDN layer. Reset zeroes everything (never rely on prior contents).
struct Bq2SeqState {
    struct FullLayerState {
        DeviceBuffer kp, vp, bt;
        int pages = 0;
    };
    struct GdnLayerState {
        DeviceBuffer conv; // BF16 [10240,3]
        DeviceBuffer ssm;  // FP32 [128,128,48]
    };
    std::vector<FullLayerState> full;
    std::vector<GdnLayerState> gdn;
    std::int32_t frontier = 0; // next absolute position

    Bq2SeqState(int max_tokens) {
        const int pages = (max_tokens + 63) / 64;
        full.resize(64);
        gdn.resize(64);
        std::vector<std::int32_t> bt(pages);
        for (int i = 0; i < pages; ++i) { bt[i] = i; }
        for (int il = 0; il < 64; ++il) {
            if (is_full_layer(il)) {
                FullLayerState s;
                s.pages = pages;
                s.kp    = DeviceBuffer(256 * 64 * 4 * pages * 2);
                s.vp    = DeviceBuffer(256 * 64 * 4 * pages * 2);
                s.bt    = DeviceBuffer(pages * 1 * 4);
                s.kp.fill(0);
                s.vp.fill(0);
                s.bt.copy_from_host(bt.data(), bt.size() * 4);
                full[il] = std::move(s);
            } else {
                GdnLayerState s;
                s.conv = DeviceBuffer(10240 * 3 * 2);
                s.ssm  = DeviceBuffer(128 * 128 * 48 * 4);
                s.conv.fill(0);
                s.ssm.fill(0);
                gdn[il] = std::move(s);
            }
        }
    }

    void reset() {
        for (int il = 0; il < 64; ++il) {
            if (is_full_layer(il)) {
                full[il].kp.fill(0);
                full[il].vp.fill(0);
            } else {
                gdn[il].conv.fill(0);
                gdn[il].ssm.fill(0);
            }
        }
        frontier = 0;
    }

    PagedKVBatchLayerView kv_view(int il) {
        PagedKVBatchLayerView v;
        v.k_pages      = Tensor(full[il].kp.p, DType::BF16, {256, 64, 4, full[il].pages});
        v.v_pages      = Tensor(full[il].vp.p, DType::FP16, {256, 64, 4, full[il].pages});
        v.block_tables = Tensor(full[il].bt.p, DType::I32, {full[il].pages, 1});
        v.head_dim     = 256;
        v.num_kv_heads = 4;
        v.storage      = KvCacheStorage::BFloat16;
        return v;
    }
};

// Production GDN block (layer il): x [5120,T] -> out [5120,T] (disjoint).
// conv_state BF16 [10240,3] and ssm_state FP32 [128,128,48] are persistent
// (in-place update); zero them at session reset, never per call.
inline void run_gdn_block(Bq2Model& m, int il, const Tensor& x, Tensor& out, Tensor& conv_state,
                          Tensor& ssm_state, cudaStream_t s) {
    const GdnLayerW& w = m.gdn[il];
    const int T        = x.ne[1];
    DeviceBuffer d_h(5120 * T * 2), d_hr(17408 * T * 2);
    DeviceBuffer d_qkv(10240 * T * 2), d_z(6144 * T * 2);
    DeviceBuffer d_g(48 * T * 4), d_b(48 * T * 4);
    DeviceBuffer d_qc(2048 * T * 2), d_kc(2048 * T * 2), d_vc(6144 * T * 2);
    DeviceBuffer d_o(6144 * T * 2), d_on(6144 * T * 2), d_ong(6144 * T * 2);
    DeviceBuffer d_ao(5120 * T * 2), d_x1(5120 * T * 2), d_mh(5120 * T * 2);
    DeviceBuffer d_gv(17408 * T * 2), d_uv(17408 * T * 2), d_act(17408 * T * 2);
    DeviceBuffer d_d(5120 * T * 2);
    {
        Tensor h(d_h.p, DType::BF16, {5120, T});
        ops::rmsnorm(x, w.an, kEps, false, h, s);
    }
    {
        const Tensor h(d_h.p, DType::BF16, {5120, T});
        Tensor hr(d_hr.p, DType::BF16, {5120, T});
        ops::t2_fwht_sign(h, m.signs_for(5120), hr, s);
    }
    {
        const Tensor hr(d_hr.p, DType::BF16, {5120, T});
        Tensor qkv(d_qkv.p, DType::BF16, {10240, T});
        Tensor z(d_z.p, DType::BF16, {6144, T});
        ops::linear(hr, w.qkv, qkv, s);
        ops::linear(hr, w.gate, z, s);
    }
    {
        DeviceArena ws_gate(std::max<std::size_t>(
            ops::gdn_gating_proj_workspace_capacity_bytes(48, 5120, T, T), 256));
        const Tensor h(d_h.p, DType::BF16, {5120, T});
        Tensor g(d_g.p, DType::FP32, {48, T});
        Tensor b(d_b.p, DType::FP32, {48, T});
        ops::gdn_gating_proj(h, w.alpha, w.beta, w.sa, w.sdt,
                             ops::GdnGateFormula::RawMultiply, ws_gate, g, b,
                             m.device.execution_view());
    }
    {
        const Tensor qkv(d_qkv.p, DType::BF16, {10240, T});
        const Tensor wcw(m.conv_weight_of(il), DType::BF16, {10240, 4});
        Tensor o0(d_qc.p, DType::BF16, {2048, T});
        Tensor o1(d_kc.p, DType::BF16, {2048, T});
        Tensor o2(d_vc.p, DType::BF16, {6144, T});
        // In-place state update: persistent conv_state is both the initial
        // window and the receiver of the trailing window.
        ops::causal_conv1d_silu_split(qkv, wcw, conv_state, conv_state, o0, o1, o2, s);
    }
    {
        DeviceArena ws_gdn(std::max<std::size_t>(
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, T, T), 256));
        const Tensor q(d_qc.p, DType::BF16, {128, 16, T});
        const Tensor k(d_kc.p, DType::BF16, {128, 16, T});
        const Tensor v(d_vc.p, DType::BF16, {128, 48, T});
        const Tensor g(d_g.p, DType::FP32, {48, T});
        const Tensor beta(d_b.p, DType::FP32, {48, T});
        Tensor o(d_o.p, DType::BF16, {128, 48, T});
        // In-place state update via the same-storage overload.
        ops::gated_delta_net(q, k, v, g, beta, kGdnScale, true, ws_gdn, ssm_state, o, s);
    }
    {
        const Tensor o(d_o.p, DType::BF16, {128, 48, T});
        const Tensor z(d_z.p, DType::BF16, {128, 48, T});
        Tensor on(d_on.p, DType::BF16, {128, 48, T});
        const Tensor wsn(w.sn.data, DType::BF16, {128});
        ops::gated_rmsnorm(o, wsn, z, kEps, on, s);
    }
    {
        // GDN-V grouped ssm_out rotation on device ([128,16,3]->[128,3,16]);
        // required by folded ssm_out weights (Prism hadamard_gdn_v_grouped).
        const Tensor ung(d_on.p, DType::BF16, {6144, T});
        Tensor grouped(d_ong.p, DType::BF16, {6144, T});
        ops::t2_gdn_v_group(ung, 16, 48, 128, grouped, s);
    }
    {
        const Tensor on(d_ong.p, DType::BF16, {6144, T});
        Tensor onr(d_hr.p, DType::BF16, {6144, T});
        ops::t2_fwht_sign(on, m.signs_for(6144), onr, s);
    }
    {
        const Tensor onr(d_hr.p, DType::BF16, {6144, T});
        Tensor ao(d_ao.p, DType::BF16, {5120, T});
        ops::linear(onr, w.so, ao, s);
    }
    {
        CUDA_CHECK(
            cudaMemcpyAsync(d_x1.p, x.data, 5120 * T * 2, cudaMemcpyDeviceToDevice, s));
        const Tensor ao(d_ao.p, DType::BF16, {5120, T});
        Tensor x1(d_x1.p, DType::BF16, {5120, T});
        ops::residual_add(ao, x1, s);
    }
    {
        const Tensor x1(d_x1.p, DType::BF16, {5120, T});
        const Tensor wpost(w.pn.data, DType::BF16, {5120});
        Tensor mh(d_mh.p, DType::BF16, {5120, T});
        ops::rmsnorm(x1, wpost, kEps, false, mh, s);
    }
    {
        const Tensor mh(d_mh.p, DType::BF16, {5120, T});
        Tensor mhr(d_hr.p, DType::BF16, {5120, T});
        ops::t2_fwht_sign(mh, m.signs_for(5120), mhr, s);
    }
    {
        const Tensor mhr(d_hr.p, DType::BF16, {5120, T});
        Tensor gv(d_gv.p, DType::BF16, {17408, T});
        Tensor uv(d_uv.p, DType::BF16, {17408, T});
        ops::linear(mhr, w.fg, gv, s);
        ops::linear(mhr, w.fu, uv, s);
    }
    {
        const Tensor gv(d_gv.p, DType::BF16, {17408, T});
        const Tensor uv(d_uv.p, DType::BF16, {17408, T});
        Tensor act(d_act.p, DType::BF16, {17408, T});
        ops::silu_mul(gv, uv, act, s);
    }
    {
        const Tensor act(d_act.p, DType::BF16, {17408, T});
        Tensor actr(d_hr.p, DType::BF16, {17408, T});
        ops::t2_fwht_sign(act, m.signs_for(17408), actr, s);
    }
    {
        const Tensor actr(d_hr.p, DType::BF16, {17408, T});
        Tensor d(d_d.p, DType::BF16, {5120, T});
        ops::linear(actr, w.fd, d, s);
    }
    {
        CUDA_CHECK(
            cudaMemcpyAsync(out.data, d_x1.p, 5120 * T * 2, cudaMemcpyDeviceToDevice, s));
        const Tensor d(d_d.p, DType::BF16, {5120, T});
        Tensor lo(out.data, DType::BF16, {5120, T});
        ops::residual_add(d, lo, s);
    }
}

// Production full-attention block (layer il): x [5120,T] -> out [5120,T].
// KV cache (per-layer, persistent, append form), positions I32 [T] (absolute),
// rope_positions I32 [T,3] (t,t,t with absolute t), table_rows I32 [1].
inline void run_full_block(Bq2Model& m, int il, const Tensor& x, Tensor& out,
                           PagedKVBatchLayerView cache, const Tensor& positions,
                           const Tensor& rope_positions, const Tensor& table_rows,
                           cudaStream_t s) {
    const FullLayerW& w = m.full[il];
    const int T         = x.ne[1];
    DeviceBuffer d_h(5120 * T * 2), d_hr(17408 * T * 2);
    DeviceBuffer d_qfull(12288 * T * 2), d_k(1024 * T * 2), d_v(1024 * T * 2);
    DeviceBuffer d_q(6144 * T * 2), d_gate(6144 * T * 2);
    DeviceBuffer d_qn(6144 * T * 2), d_kn(1024 * T * 2);
    DeviceBuffer d_attn(6144 * T * 2);
    DeviceBuffer d_ao(5120 * T * 2), d_x1(5120 * T * 2), d_mh(5120 * T * 2);
    DeviceBuffer d_gv(17408 * T * 2), d_uv(17408 * T * 2), d_act(17408 * T * 2);
    DeviceBuffer d_d(5120 * T * 2);
    {
        Tensor h(d_h.p, DType::BF16, {5120, T});
        ops::rmsnorm(x, w.an, kEps, false, h, s);
    }
    {
        const Tensor h(d_h.p, DType::BF16, {5120, T});
        Tensor hr(d_hr.p, DType::BF16, {5120, T});
        ops::t2_fwht_sign(h, m.signs_for(5120), hr, s);
    }
    {
        const Tensor hr(d_hr.p, DType::BF16, {5120, T});
        Tensor qf(d_qfull.p, DType::BF16, {12288, T});
        Tensor kf(d_k.p, DType::BF16, {1024, T});
        Tensor vf(d_v.p, DType::BF16, {1024, T});
        ops::linear(hr, w.q, qf, s);
        ops::linear(hr, w.k, kf, s);
        ops::linear(hr, w.v, vf, s);
    }
    // Q/gate de-interleave: host bit-exact BF16 repack (test staging; Q rows
    // are per-head strided). K/V buffers are already head-major.
    m.device.synchronize();
    {
        std::vector<std::uint16_t> qf_bits(12288 * T);
        CUDA_CHECK(
            cudaMemcpy(qf_bits.data(), d_qfull.p, qf_bits.size() * 2, cudaMemcpyDeviceToHost));
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
        CUDA_CHECK(
            cudaMemcpy(d_q.p, q_bits.data(), q_bits.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gate.p, gate_bits.data(), gate_bits.size() * 2,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    {
        const Tensor q(d_q.p, DType::BF16, {256, 24, T});
        const Tensor wq(w.qn.data, DType::BF16, {256});
        Tensor qn(d_qn.p, DType::BF16, {256, 24, T});
        ops::rmsnorm(q, wq, kEps, false, qn, s);
    }
    {
        const Tensor k(d_k.p, DType::BF16, {256, 4, T});
        const Tensor wk(w.kn.data, DType::BF16, {256});
        Tensor kn(d_kn.p, DType::BF16, {256, 4, T});
        ops::rmsnorm(k, wk, kEps, false, kn, s);
    }
    {
        Tensor q(d_qn.p, DType::BF16, {256, 24, T});
        Tensor k(d_kn.p, DType::BF16, {256, 4, T});
        ops::rope(rope_positions, kRotaryDim, kRopeTheta, q, k, s);
    }
    {
        const Tensor q(d_qn.p, DType::BF16, {256, 24, T});
        const Tensor k(d_kn.p, DType::BF16, {256, 4, T});
        const Tensor v(d_v.p, DType::BF16, {256, 4, T});
        const Tensor rows(table_rows);
        Tensor out3(d_attn.p, DType::BF16, {256, 24, T});
        const Tensor empty;
        const ops::CausalAttentionExecutionEnvelope env{static_cast<std::uint32_t>(T),
                                                       static_cast<std::uint32_t>(T)};
        DeviceArena ws_attn(std::max<std::size_t>(
            ops::causal_softmax_attention_workspace_capacity_bytes(
                ops::AttentionHeadGeometry{256, 24, 4}, KvCacheStorage::BFloat16, env, 1, T, T),
            256));
        ops::causal_softmax_attention(q, k, v, positions, empty, rows,
                                      ops::AttentionHeadGeometry{256, 24, 4}, kAttnScale, cache,
                                      env, ws_attn, out3, s);
    }
    {
        const Tensor gate(d_gate.p, DType::BF16, {6144, T});
        Tensor ax(d_attn.p, DType::BF16, {6144, T});
        ops::sigmoid_mul(gate, ax, s);
    }
    {
        const Tensor ax(d_attn.p, DType::BF16, {6144, T});
        Tensor axr(d_hr.p, DType::BF16, {6144, T});
        ops::t2_fwht_sign(ax, m.signs_for(6144), axr, s);
    }
    {
        const Tensor axr(d_hr.p, DType::BF16, {6144, T});
        Tensor ao(d_ao.p, DType::BF16, {5120, T});
        ops::linear(axr, w.wo, ao, s);
    }
    {
        CUDA_CHECK(
            cudaMemcpyAsync(d_x1.p, x.data, 5120 * T * 2, cudaMemcpyDeviceToDevice, s));
        const Tensor ao(d_ao.p, DType::BF16, {5120, T});
        Tensor x1(d_x1.p, DType::BF16, {5120, T});
        ops::residual_add(ao, x1, s);
    }
    {
        const Tensor x1(d_x1.p, DType::BF16, {5120, T});
        const Tensor wpost(w.pn.data, DType::BF16, {5120});
        Tensor mh(d_mh.p, DType::BF16, {5120, T});
        ops::rmsnorm(x1, wpost, kEps, false, mh, s);
    }
    {
        const Tensor mh(d_mh.p, DType::BF16, {5120, T});
        Tensor mhr(d_hr.p, DType::BF16, {5120, T});
        ops::t2_fwht_sign(mh, m.signs_for(5120), mhr, s);
    }
    {
        const Tensor mhr(d_hr.p, DType::BF16, {5120, T});
        Tensor gv(d_gv.p, DType::BF16, {17408, T});
        Tensor uv(d_uv.p, DType::BF16, {17408, T});
        ops::linear(mhr, w.fg, gv, s);
        ops::linear(mhr, w.fu, uv, s);
    }
    {
        const Tensor gv(d_gv.p, DType::BF16, {17408, T});
        const Tensor uv(d_uv.p, DType::BF16, {17408, T});
        Tensor act(d_act.p, DType::BF16, {17408, T});
        ops::silu_mul(gv, uv, act, s);
    }
    {
        const Tensor act(d_act.p, DType::BF16, {17408, T});
        Tensor actr(d_hr.p, DType::BF16, {17408, T});
        ops::t2_fwht_sign(act, m.signs_for(17408), actr, s);
    }
    {
        const Tensor actr(d_hr.p, DType::BF16, {17408, T});
        Tensor d(d_d.p, DType::BF16, {5120, T});
        ops::linear(actr, w.fd, d, s);
    }
    {
        CUDA_CHECK(
            cudaMemcpyAsync(out.data, d_x1.p, 5120 * T * 2, cudaMemcpyDeviceToDevice, s));
        const Tensor d(d_d.p, DType::BF16, {5120, T});
        Tensor lo(out.data, DType::BF16, {5120, T});
        ops::residual_add(d, lo, s);
    }
}

// Embedding lookup: token ids -> PQ2 row gather (host exact FP64 decode) ->
// BF16 upload -> production t2_fwht_sign_inverse (H-then-signs). out is
// [5120,T] ne0-fastest (column t at t*5120).
inline void embed_lookup(Bq2Model& m, const std::int32_t* ids, int T, void* out_bf16,
                         cudaStream_t s) {
    const artifact::PayloadSpan span =
        m.reader.payload("text/token_embedding");
    const std::uint8_t* epayload = reinterpret_cast<const std::uint8_t*>(span.data.data());
    std::vector<float> latent(5120 * T);
    for (int t = 0; t < T; ++t) {
        const std::vector<double> row = t2_decode_rows(epayload, kVocabPhys, ids[t], ids[t] + 1,
                                                       5120);
        for (int d = 0; d < 5120; ++d) {
            latent[static_cast<std::size_t>(t) * 5120 + d] = static_cast<float>(row[d]);
        }
    }
    DeviceBuffer d_lat(5120 * T * 2);
    upload_bf16(latent.data(), d_lat.p, 5120 * T);
    const Tensor x(d_lat.p, DType::BF16, {5120, T});
    Tensor out(out_bf16, DType::BF16, {5120, T});
    ops::t2_fwht_sign_inverse(x, static_cast<const float*>(m.signs5120.p), out, s);
}

} // namespace bq2full
