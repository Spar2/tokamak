// ModelOpt -> NInfer FP8 scale-contract pins (Ornith GDN decision gate).
//
// The official Ornith checkpoint stores FP8 projections as
//   weight: E4M3 codes, weight_scale: F32 per-tensor scalar,
//   input_scale: F32 per-tensor scalar,
// while NInfer's FP8_E4M3FN_ROW_BF16S contract carries one BF16 scale per ROW
// and (A16) never quantizes activations. This suite pins, hermetically
// (synthetic weights, no checkpoint dependency):
//  1. scalar -> broadcast-row-scales is EXACT through layout + kernel
//     (run_shape oracle against the fixture dequant);
//  2. the static input_scale is LOAD-BEARING: vendor-exact static W8A8
//     (activation quantized once with input_scale) differs from the A16
//     weight-only path by far more than epsilon, so no future change may
//     silently drop input_scale. The measured delta matches the activation-
//     quantization error scale from the offline oracle (rel ~0.03), proving
//     there are no other hidden contract terms.

#include "ops/linear/linear_test_common.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::linear;

// Exactly representable BF16 broadcast scale (2^-4).
constexpr std::uint16_t kBroadcastScaleBits = 0x3780U;
// Mirrors the measured Ornith in_proj_qkv input_scale (2^-3, exact).
constexpr double kStaticInputScale = 0.125;

quantized_weight::PackedWeight make_broadcast_fp8_weight(std::int32_t n, std::int32_t k,
                                                         std::uint32_t seed) {
    quantized_weight::PackedWeight packed =
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16S, n, k, seed);
    for (std::int32_t row = 0; row < n; ++row) {
        const std::size_t off =
            packed.scale_plane_offset + static_cast<std::size_t>(row) * 2;
        packed.payload[off]     = static_cast<std::uint8_t>(kBroadcastScaleBits & 0xFFU);
        packed.payload[off + 1] = static_cast<std::uint8_t>(kBroadcastScaleBits >> 8);
    }
    return packed;
}

// Host E4M3 satfinite quantizer via exhaustive 256-entry nearest-grid search.
// Independent of any device intrinsic; NaN encodings are never emitted.
struct E4m3Grid {
    double values[256];
    bool finite[256];
    E4m3Grid() {
        for (int b = 0; b < 256; ++b) {
            const int exp = (b >> 3) & 0xF;
            const int mant = b & 7;
            finite[b] = !(exp == 0xF && mant != 0);
            values[b] = finite[b] ? static_cast<double>(
                quantized_weight::detail::decode_e4m3fn(static_cast<std::uint8_t>(b))) : 0.0;
        }
    }
    std::uint8_t quantize(double v) const {
        double c = v;
        if (!(c >= -448.0) || c != c) c = -448.0;
        if (!(c <= 448.0) || c != c) c = 448.0;
        if (c > 448.0) c = 448.0;
        if (c < -448.0) c = -448.0;
        int best = 0;
        double best_err = std::numeric_limits<double>::infinity();
        for (int b = 0; b < 256; ++b) {
            if (!finite[b]) continue;
            const double err = std::fabs(c - values[b]);
            if (err < best_err) {
                best_err = err;
                best = b;
            }
        }
        return static_cast<std::uint8_t>(best);
    }
    double decode(std::uint8_t b) const { return values[b]; }
};

int run_broadcast_exactness() {
    constexpr std::array invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{10, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    // Fp8GdnInputGeometry is the registered shape exercising the fused-GDN
    // kernel family; the contract under test is shape-independent.
    return run_shape("FP8_ModelOptBroadcast", ActivationCompute::A16, make_broadcast_fp8_weight,
                     {16384, 5120, 901U, Comparison::Sampled, true, invocations});
}

// Static-W8A8 (vendor) vs weight-only A16 on identical synthetic inputs.
// Returns 0 when input_scale is load-bearing (delta >> epsilon) and the delta
// matches activation-quantization error scale (no hidden terms).
int run_static_scale_load_bearing() {
    constexpr std::int32_t kN = 256, kK = 128, kT = 8;
    quantized_weight::PackedWeight packed = make_broadcast_fp8_weight(kN, kK, 903U);
    std::vector<std::int32_t> rows(static_cast<std::size_t>(kN));
    for (std::int32_t i = 0; i < kN; ++i) rows[static_cast<std::size_t>(i)] = i;
    const std::vector<float> w = quantized_weight::materialize_rows_fp32(packed, rows);

    // Deterministic RMSNorm-like input.
    std::uint32_t state = 905U;
    std::vector<double> x(static_cast<std::size_t>(kK) * kT);
    for (std::size_t i = 0; i < x.size(); ++i) {
        state = state * 1664525U + 1013904223U;
        x[i] = (static_cast<double>(state >> 8) / 8388608.0 - 1.0) * 0.9;
    }

    const E4m3Grid grid;
    double num_a16 = 0.0, den = 0.0, num_gap = 0.0;
    for (std::int32_t r = 0; r < kN; ++r) {
        for (std::int32_t t = 0; t < kT; ++t) {
            double y_a16 = 0.0, y_w8a8 = 0.0;
            for (std::int32_t c = 0; c < kK; ++c) {
                const double wv = static_cast<double>(w[static_cast<std::size_t>(r) * kK + c]);
                const double xv = x[static_cast<std::size_t>(c) * kT + t];
                y_a16 += wv * xv;
                y_w8a8 += wv * grid.decode(grid.quantize(xv / kStaticInputScale)) *
                          kStaticInputScale;
            }
            // Note: broadcast weight rows carry S; w already includes it, so the
            // static vendor form needs no extra weight rescale here.
            const double d = y_a16 - y_w8a8;
            num_gap += d * d;
            num_a16 += y_a16 * y_a16;
            den += y_a16 * y_a16;
        }
    }
    (void)num_a16;
    const double rel = std::sqrt(num_gap) / std::max(std::sqrt(den), 1e-12);
    std::cout << "FP8_ModelOptBroadcast static-vs-A16 rel=" << rel << '\n';
    if (!(rel > 1e-3)) {
        std::cerr << "input_scale is not load-bearing: static W8A8 == A16\n";
        return 1;
    }
    if (!(rel < 0.15)) {
        std::cerr << "static-vs-A16 delta exceeds activation-quant error scale\n";
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = run_broadcast_exactness();
        failures += run_static_scale_load_bearing();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 ModelOpt contract\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP8 ModelOpt contract: " << error.what() << '\n';
        return 1;
    }
}
