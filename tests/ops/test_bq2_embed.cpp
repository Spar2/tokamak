// M6 Stage 6: embedding endpoint — token id -> PQ2 row gather ->
// inverse Bonsai transform -> hidden activation, vs Prism model.input_embed.
// Env: NINFER_BQ2_FULL_ART (artifact), NINFER_BQ2_VECTORS_P0/P1 (Prism dumps).
// Token ids are ground truth from the debug runs' own logs + logits sidecars
// (llama-tokenize with stdin disagrees due to trailing newline — the debug
// log is authoritative):
//   "Hello"       -> [9419]          (P0, T=1)
//   "Hello world" -> [9419, 1814]    (P1, T=2)
// Prism dump eval mapping (proven by NINFER_DECODE instrumentation + exact
// embedding matches): the FIRST eval of each run is a fixed [BOS, EOS]
// warmup (files *.prompt.bin — NOT prompt data, do not validate against);
// the SECOND eval (*.decode1.bin) is the actual prompt. Validate decode1.
// Row gather is host-side FP64 T2 decode from artifact payload bytes
// (test staging, exact); the device-side novel coverage is the production
// t2_fwht_sign_inverse op (H-then-signs). No transform on the vocab axis.
#include "ops/bq2_model_common.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace bq2full;

float f16_to_f32(std::uint16_t h) {
    std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    std::uint32_t exp  = (static_cast<std::uint32_t>(h) >> 10) & 0x1fu;
    std::uint32_t mant = static_cast<std::uint32_t>(h) & 0x03ffu;
    std::uint32_t w;
    if (exp == 0) {
        if (mant == 0) { w = sign; } else {
            int e = -14;
            while ((mant & 0x0400u) == 0) { mant <<= 1; --e; }
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

// Decode T2 rows [row_begin, row_end) of an [N,K] NInfer row-split payload
// to FP64. Layout: code plane (N*G*32B row-major) + pad to 256B + scale
// plane (N*G*2B F16LE). No high plane for T2.
std::vector<double> t2_decode_rows(const std::uint8_t* payload, int N, int row_begin, int row_end,
                                   int K) {
    const int G                 = K / 128;
    const std::size_t code_size = static_cast<std::size_t>(N) * G * 32;
    const std::size_t scale_off = (code_size + 255) / 256 * 256;
    std::vector<double> out(static_cast<std::size_t>(row_end - row_begin) * K);
    for (int n = row_begin; n < row_end; ++n) {
        for (int g = 0; g < G; ++g) {
            const std::size_t gi = static_cast<std::size_t>(n) * G + g;
            const std::uint8_t* codes =
                payload + gi * 32;
            std::uint16_t sh = static_cast<std::uint16_t>(
                payload[scale_off + gi * 2] | (payload[scale_off + gi * 2 + 1] << 8));
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

} // namespace

int main() {
    const char* art_path = std::getenv("NINFER_BQ2_FULL_ART");
    const char* p0_dir   = std::getenv("NINFER_BQ2_VECTORS_P0");
    const char* p1_dir   = std::getenv("NINFER_BQ2_VECTORS_P1");
    if (art_path == nullptr || p0_dir == nullptr || p1_dir == nullptr) {
        std::cout << "SKIP: set NINFER_BQ2_FULL_ART and NINFER_BQ2_VECTORS_P0/P1\n";
        return 77;
    }
    try {
        bool ok = true;
        Bq2Model model(art_path);
        DeviceContext& device = model.device;
        cudaStream_t stream   = model.stream;
        // Host payload bytes of the embedding table for exact row decode.
        const artifact::ObjectDescriptor* desc = nullptr;
        {
            const auto& objs = model.reader.objects();
            for (const auto& o : objs) {
                if (artifact::object_name(o) == "text/token_embedding") { desc = &o; break; }
            }
            if (desc == nullptr) { throw std::runtime_error("token_embedding missing"); }
        }
        const artifact::PayloadSpan span = model.reader.payload(*desc);
        const std::uint8_t* epayload     = reinterpret_cast<const std::uint8_t*>(span.data.data());

        const Gate kEmbed = {8e-3, 0.999, 5e-2};
        const struct {
            const char* vdir;
            std::vector<std::int32_t> ids;
        } cases[] = {{p0_dir, {9419}}, {p1_dir, {9419, 1814}}};
        for (const auto& c : cases) {
            const int T = static_cast<int>(c.ids.size());
            std::printf("== T=%d ==\n", T);
            // Host gather+dequant (exact FP64), then BF16 upload.
            std::vector<float> latent(5120 * T);
            for (int t = 0; t < T; ++t) {
                const std::vector<double> row =
                    t2_decode_rows(epayload, 248320, c.ids[t], c.ids[t] + 1, 5120);
                for (int d = 0; d < 5120; ++d) {
                    latent[static_cast<std::size_t>(t) * 5120 + d] =
                        static_cast<float>(row[d]);
                }
            }
            DeviceBuffer d_lat(5120 * T * 2), d_emb(5120 * T * 2);
            upload_bf16(latent.data(), d_lat.p, 5120 * T);
            {
                const Tensor x(d_lat.p, DType::BF16, {5120, T});
                Tensor out(d_emb.p, DType::BF16, {5120, T});
                ops::t2_fwht_sign_inverse(
                    x, static_cast<const float*>(model.signs5120.p), out, stream);
            }
            device.synchronize();
            const std::vector<float> got = d2h_bf16(device, d_emb.p, 5120 * T);
            char refname[64];
            std::snprintf(refname, sizeof(refname), "model.input_embed.decode1");
            const std::vector<float> ref0 = load_ref(c.vdir, refname, 5120 * T);
            // Prism dump is numpy (5120,T) row-major; transpose for T>1.
            const std::vector<float> ref = as_column_major(ref0, 5120, T);
            ok &= check_vec(got.data(), ref.data(), 5120 * T, "embed", kEmbed);
        }
        std::printf("%s BQ2_EMBED\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    }
}
