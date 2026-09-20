// Production FWHT/sign conformance: randomized BF16 inputs and +-1 signs vs
// FP64 CPU reference (sign multiply, then orthonormal FWHT per 1024-block).
#include "ops/linear/t2/t2_rotation.h"

#include "core/device.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

using namespace ninfer;

std::uint16_t f32_to_bf16(float v) {
    std::uint32_t w = 0;
    std::memcpy(&w, &v, 4);
    const std::uint32_t bias = 0x7fffu + ((w >> 16) & 1u);
    return static_cast<std::uint16_t>((w + bias) >> 16);
}

float bf16_to_f32(std::uint16_t h) {
    std::uint32_t w = static_cast<std::uint32_t>(h) << 16;
    float f = 0.0F;
    std::memcpy(&f, &w, 4);
    return f;
}

void fwht_block_fp64(double* y, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                const double u = y[i + j];
                const double v = y[i + len + j];
                y[i + j]       = u + v;
                y[i + len + j] = u - v;
            }
        }
    }
}

int run_case(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
    std::vector<std::uint16_t> x_bits(static_cast<std::size_t>(k) * t);
    std::vector<float> signs(k);
    for (auto& s : signs) { s = (rng() & 1u) ? 1.0F : -1.0F; }
    std::vector<double> ref(static_cast<std::size_t>(k) * t);
    for (std::int32_t tok = 0; tok < t; ++tok) {
        for (std::int32_t i = 0; i < k; ++i) {
            const std::size_t idx = static_cast<std::size_t>(tok) * k + i;
            x_bits[idx]           = f32_to_bf16(dist(rng));
            ref[idx]              = static_cast<double>(bf16_to_f32(x_bits[idx])) * signs[i];
        }
        for (std::int32_t b = 0; b < k; b += 1024) {
            fwht_block_fp64(ref.data() + static_cast<std::size_t>(tok) * k + b, 1024);
        }
        for (std::int32_t i = 0; i < k; ++i) {
            ref[static_cast<std::size_t>(tok) * k + i] *= 1.0 / 32.0;
        }
    }

    void* d_x     = nullptr;
    void* d_out   = nullptr;
    void* d_signs = nullptr;
    if (cudaMalloc(&d_x, x_bits.size() * 2) != cudaSuccess) { return 1; }
    if (cudaMalloc(&d_out, x_bits.size() * 2) != cudaSuccess) { return 1; }
    if (cudaMalloc(&d_signs, signs.size() * 4) != cudaSuccess) { return 1; }
    if (cudaMemcpy(d_x, x_bits.data(), x_bits.size() * 2, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        return 1;
    }
    if (cudaMemcpy(d_signs, signs.data(), signs.size() * 4, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        return 1;
    }
    Tensor x(d_x, DType::BF16, {k, t});
    Tensor out(d_out, DType::BF16, {k, t});
    try {
        ops::t2_fwht_sign(x, static_cast<const float*>(d_signs), out, nullptr);
    } catch (const std::exception& error) {
        std::cerr << "FWHT threw: " << error.what() << '\n';
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) { return 1; }
    std::vector<std::uint16_t> got(x_bits.size());
    if (cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 1;
    }
    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_signs);

    // BF16-output tolerance: 2-ULP absolute + relative-L2 + cosine.
    double se = 0.0, ref2 = 0.0, dot = 0.0, out2 = 0.0, maxabs = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double a = bf16_to_f32(got[i]);
        const double b = ref[i];
        const double e = std::fabs(a - b);
        if (e > maxabs) { maxabs = e; }
        se += e * e;
        ref2 += b * b;
        dot += a * b;
        out2 += a * a;
    }
    const double rmse = std::sqrt(se / got.size());
    const double rel  = std::sqrt(se / ref2);
    const double cos  = dot / std::sqrt(out2 * ref2);
    const bool ok     = maxabs <= 0.02 && rel <= 5e-3 && cos >= 0.99999;
    std::printf("FWHT k=%d t=%d: maxabs=%.6f rmse=%.6f rel=%.6f cos=%.8f %s\n", k, t, maxabs,
                rmse, rel, cos, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Inverse rotation: out = signs .* (FWHT(x)/32) per 1024-block. Reference is
// FP64 FWHT then sign multiply; roundtrip forward(inverse(z)) == z pins the
// order (H-then-signs, not signs-then-H).
int run_inverse_case(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
    std::vector<std::uint16_t> x_bits(static_cast<std::size_t>(k) * t);
    std::vector<float> signs(k);
    for (auto& s : signs) { s = (rng() & 1u) ? 1.0F : -1.0F; }
    std::vector<double> ref(static_cast<std::size_t>(k) * t);
    for (std::int32_t tok = 0; tok < t; ++tok) {
        for (std::int32_t i = 0; i < k; ++i) {
            const std::size_t idx = static_cast<std::size_t>(tok) * k + i;
            x_bits[idx]           = f32_to_bf16(dist(rng));
            ref[idx]              = static_cast<double>(bf16_to_f32(x_bits[idx]));
        }
        for (std::int32_t b = 0; b < k; b += 1024) {
            fwht_block_fp64(ref.data() + static_cast<std::size_t>(tok) * k + b, 1024);
        }
        for (std::int32_t i = 0; i < k; ++i) {
            ref[static_cast<std::size_t>(tok) * k + i] *= signs[i] / 32.0;
        }
    }

    void* d_x     = nullptr;
    void* d_out   = nullptr;
    void* d_signs = nullptr;
    if (cudaMalloc(&d_x, x_bits.size() * 2) != cudaSuccess) { return 1; }
    if (cudaMalloc(&d_out, x_bits.size() * 2) != cudaSuccess) { return 1; }
    if (cudaMalloc(&d_signs, signs.size() * 4) != cudaSuccess) { return 1; }
    if (cudaMemcpy(d_x, x_bits.data(), x_bits.size() * 2, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        return 1;
    }
    if (cudaMemcpy(d_signs, signs.data(), signs.size() * 4, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        return 1;
    }
    Tensor x(d_x, DType::BF16, {k, t});
    Tensor out(d_out, DType::BF16, {k, t});
    try {
        ops::t2_fwht_sign_inverse(x, static_cast<const float*>(d_signs), out, nullptr);
    } catch (const std::exception& error) {
        std::cerr << "FWHT-inverse threw: " << error.what() << '\n';
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) { return 1; }
    std::vector<std::uint16_t> got(x_bits.size());
    if (cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 1;
    }

    // Roundtrip through the forward op must recover the input.
    Tensor back(d_x, DType::BF16, {k, t});
    try {
        ops::t2_fwht_sign(out, static_cast<const float*>(d_signs), back, nullptr);
    } catch (const std::exception& error) {
        std::cerr << "FWHT-forward roundtrip threw: " << error.what() << '\n';
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) { return 1; }
    std::vector<std::uint16_t> roundtrip(x_bits.size());
    if (cudaMemcpy(roundtrip.data(), d_x, roundtrip.size() * 2, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        return 1;
    }
    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_signs);

    double se = 0.0, ref2 = 0.0, dot = 0.0, out2 = 0.0, maxabs = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double a = bf16_to_f32(got[i]);
        const double b = ref[i];
        const double e = std::fabs(a - b);
        if (e > maxabs) { maxabs = e; }
        se += e * e;
        ref2 += b * b;
        dot += a * b;
        out2 += a * a;
    }
    const double rmse = std::sqrt(se / got.size());
    const double rel  = std::sqrt(se / ref2);
    const double cos  = dot / std::sqrt(out2 * ref2);
    bool ok           = maxabs <= 0.02 && rel <= 5e-3 && cos >= 0.99999;
    double rt_max     = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double e =
            std::fabs((double)bf16_to_f32(roundtrip[i]) - (double)bf16_to_f32(x_bits[i]));
        if (e > rt_max) { rt_max = e; }
    }
    ok = ok && rt_max <= 0.02;
    std::printf("FWHT-INV k=%d t=%d: maxabs=%.6f rmse=%.6f rel=%.6f cos=%.8f roundtrip=%.6f %s\n",
                k, t, maxabs, rmse, rel, cos, rt_max, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

} // namespace

int main() {
    int failures = 0;
    failures += run_case(1024, 1, 11U);
    failures += run_case(5120, 1, 12U);
    failures += run_case(5120, 4, 13U);
    failures += run_case(17408, 1, 14U);
    failures += run_inverse_case(5120, 1, 21U);
    failures += run_inverse_case(5120, 4, 22U);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " T2_FWHT\n";
    return failures == 0 ? 0 : 1;
}
