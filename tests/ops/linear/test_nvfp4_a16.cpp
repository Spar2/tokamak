#include "ops/linear/linear_test_common.h"

#include <array>
#include <cstdlib>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_nvfp4_a16() {
    constexpr std::array attn_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{20, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{32, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{33, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    constexpr std::array new_problem_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    constexpr std::array ornith_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{18, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{31, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{32, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    // Vocab projections: decode + small-T production widths. T>32 is served by
    // 32-chunked small-T calls through the same launchers (shared chunking
    // path already covered by the Ornith MLP T=17/18/31/129 cases above).
    constexpr std::array head_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{32, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    int failures = 0;
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {14336, 5120, 701U, Comparison::Sampled, true, attn_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {16384, 5120, 703U, Comparison::Sampled, true, new_problem_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {34816, 5120, 704U, Comparison::Sampled, true, new_problem_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 6144, 705U, Comparison::Sampled, true, new_problem_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 17408, 707U, Comparison::Sampled, true, new_problem_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {24576, 4096, 709U, Comparison::Sampled, true, ornith_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {4096, 12288, 711U, Comparison::Sampled, true, ornith_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {248320, 4096, 713U, Comparison::Sampled, true, head_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {131072, 4096, 715U, Comparison::Sampled, true, head_invocations});
    return failures;
}

} // namespace

int main() {
    // Hermetic A16-oracle validation: the graduated default routes Ornith MLP
    // A16Only at T>=128 to dynamic W4A4, which is approximate and must fail a
    // tight exact-dequant oracle. Pin the W4A16 fallback so this suite tests
    // what its name claims regardless of ambient environment.
    ::setenv("NINFER_ORNITH_DYNAMIC_W4A4", "0", 1);
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_nvfp4_a16();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
