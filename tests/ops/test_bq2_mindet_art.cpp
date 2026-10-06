// M7 minimal harness with REAL artifact weight VALUES (isolates weight-content).
// Reads text/layers/8/ffn_down host payload, splits code/scale planes via
// row-split geometry, uploads into OWN guarded device buffers, then runs the
// same guarded single-stream determinism loop as mindet with synthetic input.
#include "artifact/reader.h"
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

struct Guarded {
    void* base    = nullptr;
    void* payload = nullptr;
    std::size_t n = 0;
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
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::uint8_t*>(payload) + n, CAN, G, s));
    }
    bool check_canaries(const char* tag) {
        std::uint8_t f[G], b[G];
        CUDA_CHECK(cudaMemcpy(f, base, G, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b, static_cast<std::uint8_t*>(payload) + n, G,
                              cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < G; ++i) {
            if (f[i] != CAN || b[i] != CAN) {
                std::printf("CANARY-CORRUPT %s front[%zu]=%02x back[%zu]=%02x\n", tag, i,
                            f[i], i, b[i]);
                return false;
            }
        }
        return true;
    }
};

} // namespace

int main(int argc, char** argv) {
    const int kernel   = argc > 1 ? std::atoi(argv[1]) : 0;
    const int launches = argc > 2 ? std::atoi(argv[2]) : 1000;
    const int stream_m = argc > 3 ? std::atoi(argv[3]) : 0;
    const char* art_path = std::getenv("NINFER_BQ2_FULL_ART");
    if (art_path == nullptr) {
        std::cout << "SKIP: set NINFER_BQ2_FULL_ART\n";
        return 77;
    }
    // fd: N=5120 K=17408 T=32.
    const int N = 5120, K = 17408, T = 32, groups = K / 128;
    std::printf("mindet_art kernel=%s N=%d K=%d T=%d launches=%d stream=%s\n",
                kernel == 0 ? "sync" : "legacy", N, K, T, launches,
                stream_m == 0 ? "created" : "default");

    DeviceContext device(0);
    (void)device;
    artifact::Reader reader(art_path);
    const artifact::PayloadSpan span = reader.payload("text/layers/8/ffn_down");
    const std::vector<std::uint64_t> shv = {5120, 17408};
    const std::span<const std::uint64_t> sh_span(shv.data(), shv.size());
    const artifact::RowSplitGeometry geo = artifact::row_split_geometry(
        artifact::NumericFormat::T2G128_F16S, sh_span);
    std::printf("row-split geo: low_plane=%llu scale_off=%llu scale_bytes=%llu enc=%llu\n",
                (unsigned long long)geo.low_plane_bytes,
                (unsigned long long)geo.scale_plane_offset,
                (unsigned long long)geo.scale_plane_bytes,
                (unsigned long long)geo.encoded_bytes);
    if (geo.groups_per_row != 136 || geo.group_size != 128) {
        std::printf("unexpected geometry\n");
        return 2;
    }
    const std::size_t codes_n  = static_cast<std::size_t>(N) * groups * 32;
    const std::size_t scales_n = static_cast<std::size_t>(N) * groups;
    if (geo.low_plane_bytes != codes_n || geo.scale_plane_bytes != scales_n * 2) {
        std::printf("geometry/size mismatch\n");
        return 2;
    }
    const std::byte* pbase  = span.data.data();
    const void* h_codes_ptr = pbase;
    const void* h_scales_ptr = pbase + geo.scale_plane_offset;

    cudaStream_t stream = nullptr;
    if (stream_m == 0) { CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }

    const std::size_t in_n  = static_cast<std::size_t>(K) * T;
    const std::size_t out_n = static_cast<std::size_t>(N) * T;
    // Real weight bytes into OWN guarded buffers (controlled allocations).
    Guarded d_codes(codes_n), d_scales(scales_n * 2);
    CUDA_CHECK(cudaMemcpy(d_codes.payload, h_codes_ptr, codes_n, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_scales.payload, h_scales_ptr, scales_n * 2, cudaMemcpyHostToDevice));
    Weight w{};
    w.qtype       = QType::T2G128_F16S;
    w.group_size  = 128;
    w.qdata       = d_codes.payload;
    w.scales      = d_scales.payload;
    w.n           = N;
    w.k           = K;
    w.group       = 128;
    w.layout      = QuantLayout::RowSplit;
    w.scale_dtype = DType::FP16;
    w.ndim        = 2;
    w.shape[0] = w.padded_shape[0] = N;
    w.shape[1] = w.padded_shape[1] = K;
    d_codes.poison_canaries(stream);
    d_scales.poison_canaries(stream);
    std::vector<std::uint16_t> h_in(in_n);
    for (int c = 0; c < T; ++c) {
        for (int k = 0; k < K; ++k) {
            h_in[static_cast<std::size_t>(k) + static_cast<std::size_t>(K) * c] =
                static_cast<std::uint16_t>(0x3C00 + ((k * 37 + c * 12345) % 512));
        }
    }
    Guarded d_in(in_n * 2), d_out(out_n * 2);
    CUDA_CHECK(cudaMemcpy(d_in.payload, h_in.data(), in_n * 2, cudaMemcpyHostToDevice));
    d_in.poison_canaries(stream);
    d_out.poison_canaries(stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    const Tensor x(d_in.payload, DType::BF16, {K, T});
    Tensor y(d_out.payload, DType::BF16, {N, T});

    auto dev_hash_w = [&]() {
        std::vector<std::uint8_t> c(codes_n);
        std::vector<std::uint16_t> s(scales_n), xi(in_n);
        CUDA_CHECK(cudaMemcpy(c.data(), d_codes.payload, codes_n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(
            cudaMemcpy(s.data(), d_scales.payload, scales_n * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(xi.data(), d_in.payload, in_n * 2, cudaMemcpyDeviceToHost));
        return std::make_tuple(fnv1a(c.data(), c.size()), fnv1a(s.data(), s.size() * 2),
                               fnv1a(xi.data(), xi.size() * 2));
    };
    auto [hc0, hs0, hi0] = dev_hash_w();
    std::printf("baseline hashes codes=%016llx scales=%016llx in=%016llx\n",
                (unsigned long long)hc0, (unsigned long long)hs0, (unsigned long long)hi0);

    std::vector<std::uint16_t> ref(out_n), got(out_n);
    std::uint64_t ref_hash = 0;
    int fails              = 0;
    int fail_first         = -1;
    const auto t_start     = std::chrono::steady_clock::now();
    for (int li = 0; li < launches; ++li) {
        CUDA_CHECK(cudaMemsetAsync(d_out.payload, 0xAB, out_n * 2, stream));
        if (kernel == 0) {
            ops::detail::launch_t2_prefill_mma_sync(x, w, y, stream);
        } else {
            ops::linear(x, w, y, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        bool ok = true;
        ok &= d_out.check_canaries("out");
        ok &= d_in.check_canaries("in");
        ok &= d_codes.check_canaries("codes");
        ok &= d_scales.check_canaries("scales");
        if (!ok) {
            std::printf("STOP launch %d: canary corruption\n", li);
            return 3;
        }
        CUDA_CHECK(cudaMemcpy(got.data(), d_out.payload, out_n * 2, cudaMemcpyDeviceToHost));
        std::size_t poison_left = 0;
        for (auto v : got) {
            if (v == 0xABAB) { ++poison_left; }
        }
        if (poison_left != 0) {
            std::printf("STOP launch %d: %zu poison words left\n", li, poison_left);
            return 3;
        }
        const std::uint64_t h = fnv1a(got.data(), got.size() * 2);
        if (li == 0) {
            ref      = got;
            ref_hash = h;
        } else if (h != ref_hash) {
            ++fails;
            if (fail_first < 0) { fail_first = li; }
            std::size_t mm = 0;
            int shown      = 0;
            for (std::size_t i = 0; i < out_n; ++i) {
                if (ref[i] != got[i]) {
                    ++mm;
                    if (shown < 5) {
                        const int row = static_cast<int>(i % static_cast<std::size_t>(N));
                        const int tok = static_cast<int>(i / static_cast<std::size_t>(N));
                        std::printf("  fail li=%d idx=%zu r=%d c=%d bx=%d by=%d ref=%04x "
                                    "got=%04x d=%.6f\n",
                                    li, i, row, tok, row / 64, tok / 32, ref[i], got[i],
                                    std::fabs(b2f(ref[i]) - b2f(got[i])));
                        ++shown;
                    }
                }
            }
            std::printf("  fail li=%d mm=%zu/%zu frac=%.4f\n", li, mm, out_n,
                        (double)mm / out_n);
        }
        if ((li + 1) % 100 == 0) {
            auto [hc, hs, hi] = dev_hash_w();
            const bool im      = (hc == hc0 && hs == hs0 && hi == hi0);
            const auto now     = std::chrono::steady_clock::now();
            const double secs =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count() /
                1000.0;
            std::printf("  block %d t=%.1fs fails=%d w-hash=%s\n", (li + 1) / 100, secs,
                        fails, im ? "OK" : "CORRUPT-STOP");
            if (!im) { return 3; }
        }
    }
    const auto t_end  = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() / 1000.0;
    std::printf("DONE fails=%d/%d first_fail=%d time=%.1fs ref_hash=%016llx\n", fails,
                launches, fail_first, secs, (unsigned long long)ref_hash);
    if (stream != nullptr) { CUDA_CHECK(cudaStreamDestroy(stream)); }
    return fails == 0 ? 0 : 1;
}
