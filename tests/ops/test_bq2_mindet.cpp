// M7 minimal determinism harness (independent of the full-model harness).
// Synthetic fixed-seed PQ2 weights + input; NO artifact, NO model graph,
// NO caches, NO multiple streams.
//
// Buffers (allocated ONCE, reused for all launches):
//   codes | scales | input | output, each with 256B front/back canaries.
// Per launch: poison output, launch, GetLastError, DeviceSynchronize,
// verify canaries + full overwrite, D2H output, FNV-hash, compare to run0.
// Inputs re-hashed every 100 launches and on any failure.
//
// Usage:
//   mindet <kernel:0=sync,1=legacy> <N> <K> <T> <launches> <stream:0=created,1=default> <inpat:0=allpos,1=mixed>
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/t2/t2_launch.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;

// Deterministic PRNG (splitmix64).
std::uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
std::uint64_t rng_next() {
    std::uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

std::uint64_t fnv1a(const void* data, std::size_t n) {
    std::uint64_t h = 1469598103934665603ULL;
    const auto* p   = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

float b2f(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// float -> F16 bits (round-to-nearest-even-ish; deterministic).
std::uint16_t f32_to_f16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    const std::uint32_t sign = (u >> 16) & 0x8000u;
    const int exp            = static_cast<int>((u >> 23) & 0xFFu) - 112;
    const std::uint32_t mant = u & 0x7FFFFFu;
    if (exp <= 0) { return static_cast<std::uint16_t>(sign); }
    if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7BFFu); }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) |
                                      (mant >> 13));
}

// Guarded device buffer: [256B front canary][payload][256B back canary].
struct Guarded {
    void* base       = nullptr; // allocation base
    void* payload    = nullptr; // interior pointer handed to kernels
    std::size_t n    = 0;       // payload bytes
    static constexpr std::size_t G = 256;
    static constexpr unsigned char CAN = 0xC4;

    explicit Guarded(std::size_t payload_bytes) : n(payload_bytes) {
        CUDA_CHECK(cudaMalloc(&base, n + 2 * G));
        payload = static_cast<std::uint8_t*>(base) + G;
    }
    ~Guarded() {
        if (base != nullptr) { cudaFree(base); }
    }
    Guarded(const Guarded&)            = delete;
    Guarded& operator=(const Guarded&) = delete;
    void poison_canaries(cudaStream_t s) {
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::uint8_t*>(base), CAN, G, s));
        CUDA_CHECK(
            cudaMemsetAsync(static_cast<std::uint8_t*>(payload) + n, CAN, G, s));
    }
    // Returns false + prints on any canary corruption.
    bool check_canaries(const char* tag) {
        std::uint8_t f[G], b[G];
        CUDA_CHECK(cudaMemcpy(f, base, G, cudaMemcpyDeviceToHost));
        CUDA_CHECK(
            cudaMemcpy(b, static_cast<std::uint8_t*>(payload) + n, G, cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < G; ++i) {
            if (f[i] != CAN || b[i] != CAN) {
                std::printf("CANARY-CORRUPT %s front[%zu]=%02x back[%zu]=%02x\n", tag, i, f[i],
                            i, b[i]);
                return false;
            }
        }
        return true;
    }
};

} // namespace

int main(int argc, char** argv) {
    const int kernel   = argc > 1 ? std::atoi(argv[1]) : 0;
    const int N        = argc > 2 ? std::atoi(argv[2]) : 5120;
    const int K        = argc > 3 ? std::atoi(argv[3]) : 17408;
    const int T        = argc > 4 ? std::atoi(argv[4]) : 32;
    const int launches = argc > 5 ? std::atoi(argv[5]) : 1000;
    const int stream_m = argc > 6 ? std::atoi(argv[6]) : 0;
    const int inpat    = argc > 7 ? std::atoi(argv[7]) : 0;
    const int idle_ms  = argc > 8 ? std::atoi(argv[8]) : 0;
    if (N <= 0 || (N % 64) != 0 || K <= 0 || (K % 128) != 0 || T != 32 || launches <= 0) {
        std::printf("usage: mindet <k:0=sync,1=legacy> <N%%64> <K%%128> <T=32> <launches> "
                    "<stream:0=created,1=default> <inpat:0=allpos,1=mixed> [idle_ms_per_100]\n");
        return 2;
    }
    const int groups = K / 128;
    std::printf("mindet kernel=%s N=%d K=%d T=%d launches=%d stream=%s inpat=%s\n",
                kernel == 0 ? "sync" : "legacy", N, K, T, launches,
                stream_m == 0 ? "created" : "default", inpat == 0 ? "allpos" : "mixed");

    cudaStream_t stream = nullptr;
    if (stream_m == 0) { CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }

    // ---- synthetic fixed data (host) ----
    const std::size_t codes_n  = static_cast<std::size_t>(N) * groups * 32;
    const std::size_t scales_n = static_cast<std::size_t>(N) * groups;
    const std::size_t in_n     = static_cast<std::size_t>(K) * T;
    const std::size_t out_n    = static_cast<std::size_t>(N) * T;
    std::vector<std::uint8_t> h_codes(codes_n);
    std::vector<std::uint16_t> h_scales(scales_n), h_in(in_n);
    for (std::size_t i = 0; i < codes_n; ++i) {
        h_codes[i] = static_cast<std::uint8_t>(rng_next() & 0xFFu);
    }
    // Constrain to ternary codes 0..2 (code->val = code-1), like real PQ2.
    // Codes are 2-bit packed little-endian; regenerate per 32-bit word.
    for (std::size_t w = 0; w < codes_n / 4; ++w) {
        std::uint32_t u = 0;
        for (int c = 0; c < 16; ++c) {
            u |= (static_cast<std::uint32_t>(rng_next() % 3) << (c * 2));
        }
        std::memcpy(&h_codes[w * 4], &u, 4);
    }
    for (std::size_t i = 0; i < scales_n; ++i) {
        const float s = 0.01f + static_cast<float>(rng_next() % 1000) * 0.00009f; // ~0.01..0.1
        h_scales[i]   = f32_to_f16(s);
    }
    for (int c = 0; c < T; ++c) {
        for (int k = 0; k < K; ++k) {
            std::uint16_t v;
            if (inpat == 0) {
                v = static_cast<std::uint16_t>(0x3C00 + ((k * 37 + c * 12345) % 512));
            } else {
                // Mixed-sign pseudo-normal-ish in [-2,2).
                const float f =
                    (static_cast<float>(rng_next() % 2000) - 1000.0f) * 0.002f;
                std::uint32_t u;
                // Round via FP32->BF16 truncation of low 16 bits with RNE on bit 16.
                std::memcpy(&u, &f, 4);
                v = static_cast<std::uint16_t>((u >> 16) +
                                               ((u >> 15) & 1u)); // approx RNE
            }
            h_in[static_cast<std::size_t>(k) + static_cast<std::size_t>(K) * c] = v;
        }
    }
    const std::uint64_t h_codes_hash  = fnv1a(h_codes.data(), h_codes.size());
    const std::uint64_t h_scales_hash = fnv1a(h_scales.data(), h_scales.size() * 2);
    const std::uint64_t h_in_hash     = fnv1a(h_in.data(), h_in.size() * 2);
    std::printf("host hashes codes=%016llx scales=%016llx in=%016llx\n",
                (unsigned long long)h_codes_hash, (unsigned long long)h_scales_hash,
                (unsigned long long)h_in_hash);

    // ---- device buffers (allocated ONCE) ----
    Guarded d_codes(codes_n), d_scales(scales_n * 2), d_in(in_n * 2), d_out(out_n * 2);
    CUDA_CHECK(cudaMemcpy(d_codes.payload, h_codes.data(), codes_n, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_scales.payload, h_scales.data(), scales_n * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in.payload, h_in.data(), in_n * 2, cudaMemcpyHostToDevice));
    d_codes.poison_canaries(stream);
    d_scales.poison_canaries(stream);
    d_in.poison_canaries(stream);
    d_out.poison_canaries(stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight w{};
    w.qtype      = QType::T2G128_F16S;
    w.group_size = 128;
    w.qdata      = d_codes.payload;
    w.scales     = d_scales.payload;
    w.n          = N;
    w.k          = K;
    w.group      = 128;
    w.layout     = QuantLayout::RowSplit;
    w.scale_dtype = DType::FP16;
    w.ndim       = 2;
    w.shape[0] = w.padded_shape[0] = N;
    w.shape[1] = w.padded_shape[1] = K;
    const Tensor x(d_in.payload, DType::BF16, {K, T});
    Tensor y(d_out.payload, DType::BF16, {N, T});

    // Device-side input hash baseline (codes/scales/input), checked periodically.
    auto dev_hash3 = [&]() {
        std::vector<std::uint8_t> c(codes_n);
        std::vector<std::uint16_t> s(scales_n), xi(in_n);
        CUDA_CHECK(cudaMemcpy(c.data(), d_codes.payload, codes_n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(
            cudaMemcpy(s.data(), d_scales.payload, scales_n * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(xi.data(), d_in.payload, in_n * 2, cudaMemcpyDeviceToHost));
        return std::make_tuple(fnv1a(c.data(), c.size()), fnv1a(s.data(), s.size() * 2),
                               fnv1a(xi.data(), xi.size() * 2));
    };
    {
        auto [hc, hs, hi] = dev_hash3();
        std::printf("dev hashes codes=%016llx scales=%016llx in=%016llx %s\n",
                    (unsigned long long)hc, (unsigned long long)hs, (unsigned long long)hi,
                    (hc == h_codes_hash && hs == h_scales_hash && hi == h_in_hash) ? "MATCH"
                                                                                   : "MISMATCH");
        if (!(hc == h_codes_hash && hs == h_scales_hash && hi == h_in_hash)) { return 1; }
    }

    std::vector<std::uint16_t> ref(out_n), got(out_n);
    std::uint64_t ref_hash = 0;
    int fails              = 0;
    int fail_first_launch  = -1;
    const auto t_start     = std::chrono::steady_clock::now();
    for (int li = 0; li < launches; ++li) {
        // Poison output (known pattern) on the test stream.
        CUDA_CHECK(cudaMemsetAsync(d_out.payload, 0xAB, out_n * 2, stream));
        // Launch.
        if (kernel == 0) {
            ops::detail::launch_t2_prefill_mma_sync(x, w, y, stream);
        } else {
            ops::linear(x, w, y, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // Canaries.
        bool ok = true;
        ok &= d_out.check_canaries("out");
        ok &= d_codes.check_canaries("codes");
        ok &= d_scales.check_canaries("scales");
        ok &= d_in.check_canaries("in");
        if (!ok) {
            std::printf("STOP launch %d: canary corruption\n", li);
            return 3;
        }
        CUDA_CHECK(cudaMemcpy(got.data(), d_out.payload, out_n * 2, cudaMemcpyDeviceToHost));
        // Full-overwrite check: no poison words may remain.
        std::size_t poison_left = 0;
        for (auto v : got) {
            if (v == 0xABAB) { ++poison_left; }
        }
        if (poison_left != 0) {
            std::printf("STOP launch %d: %zu poison words left (partial overwrite)\n", li,
                        poison_left);
            return 3;
        }
        const std::uint64_t h = fnv1a(got.data(), got.size() * 2);
        if (li == 0) {
            ref      = got;
            ref_hash = h;
        } else if (h != ref_hash) {
            ++fails;
            if (fail_first_launch < 0) { fail_first_launch = li; }
            // Localize: count, first few (idx,row,tok,cta,ref,got,delta).
            std::size_t mm = 0;
            int shown      = 0;
            // CTA histogram over blockIdx.x (rows/64).
            std::vector<std::size_t> cta_hist((N + 63) / 64, 0);
            for (std::size_t i = 0; i < out_n; ++i) {
                if (ref[i] != got[i]) {
                    ++mm;
                    const int r = static_cast<int>(i % out_n / T); // placeholder, fixed below
                    (void)r;
                    cta_hist[(i % static_cast<std::size_t>(N)) / 64]++;
                    if (shown < 5) {
                        const int row = static_cast<int>(i % static_cast<std::size_t>(N));
                        const int tok = static_cast<int>(i / static_cast<std::size_t>(N));
                        const float a = b2f(ref[i]), b = b2f(got[i]);
                        const std::uint16_t d = ref[i] ^ got[i];
                        std::printf("  fail li=%d idx=%zu r=%d c=%d bx=%d by=%d ref=%04x got=%04x "
                                    "xor=%04x fa=%.6f fb=%.6f d=%.6f\n",
                                    li, i, row, tok, row / 64, tok / 32, ref[i], got[i], d, a,
                                    b, std::fabs(a - b));
                        ++shown;
                    }
                }
            }
            std::size_t worst = 0;
            int wbx = -1;
            for (std::size_t bxi = 0; bxi < cta_hist.size(); ++bxi) {
                if (cta_hist[bxi] > worst) {
                    worst = cta_hist[bxi];
                    wbx   = static_cast<int>(bxi);
                }
            }
            std::printf("  fail li=%d mm=%zu/%zu frac=%.4f worst-bx=%d(%zu) hash=%016llx\n",
                        li, mm, out_n, (double)mm / out_n, wbx, worst,
                        (unsigned long long)h);
        }
        if ((li + 1) % 100 == 0) {
            auto [hc, hs, hi] = dev_hash3();
            const bool im      = (hc == h_codes_hash && hs == h_scales_hash && hi == h_in_hash);
            const auto now     = std::chrono::steady_clock::now();
            const double secs =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count() /
                1000.0;
            std::printf("  block %d launches t=%.1fs fails=%d inputhash=%s\n", (li + 1) / 100,
                        secs, fails, im ? "OK" : "CORRUPT-STOP");
            if (!im) { return 3; }
            if (idle_ms > 0) {
                // Let the GPU go idle (clocks collapse) to test
                // idle->burst transition sensitivity.
                CUDA_CHECK(cudaDeviceSynchronize());
                std::this_thread::sleep_for(std::chrono::milliseconds(idle_ms));
            }
        }
    }
    const auto t_end    = std::chrono::steady_clock::now();
    const double secs   = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() / 1000.0;
    std::printf("DONE fails=%d/%d first_fail=%d time=%.1fs ref_hash=%016llx\n", fails, launches,
                fail_first_launch, secs, (unsigned long long)ref_hash);
    if (stream != nullptr) { CUDA_CHECK(cudaStreamDestroy(stream)); }
    return fails == 0 ? 0 : 1;
}
