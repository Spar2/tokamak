#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr Invocation convenience(std::int32_t t) { return {t, CallForm::A16Convenience}; }

int t2_a16_conformance() {
    int failures = 0;

    // Representative Bonsai/Qwen3.8 ffn_gate geometry (M=17408, K=5120).
    constexpr std::array kN17408K5120{
        convenience(1), a16(2), a16(4), a16(8), a16(16), a16(17), a16(32),
    };
    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2g128_f16s_weight,
                          {17408, 5120, 101U, Comparison::Sampled, false, kN17408K5120});

    // ffn_down geometry (M=5120, K=17408).
    constexpr std::array kN5120K17408{
        convenience(1), a16(2), a16(4), a16(8), a16(16), a16(32),
    };
    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2g128_f16s_weight,
                          {5120, 17408, 103U, Comparison::Sampled, false, kN5120K17408});

    // Small exact-comparison shape.
    constexpr std::array kN256K512{
        convenience(1), a16(2), a16(4), a16(8), a16(16), a16(32),
    };
    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2g128_f16s_weight,
                          {256, 512, 107U, Comparison::Full, true, kN256K512});

    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = t2_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " T2_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "T2_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
