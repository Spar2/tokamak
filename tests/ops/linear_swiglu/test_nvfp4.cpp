#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <cstdlib>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    // See test_nvfp4_a16.cpp: pin the W4A16 fallback for hermetic oracle checks.
    ::setenv("NINFER_ORNITH_DYNAMIC_W4A4", "0", 1);
    try {
        constexpr std::array<std::int32_t, 4> kA16Cases{1, 4, 8, 16};
        constexpr std::array<std::int32_t, 14> kA4Cases{2, 4, 5, 16, 56, 64, 65, 96, 97, 112, 128, 129, 256, 1024};
        int failures = 0;
        failures += run_profile("LinearSwiGLU NVFP4_A16",
                                {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16},
                                kA16Cases);
        constexpr std::array<std::int32_t, 10> kOrnithA16Cases{1, 4, 8, 16, 17, 18, 31,
                                                               32, 64, 128};
        failures += run_profile("LinearSwiGLU NVFP4_A16_Ornith",
                                {QType::NVFP4, 24576, 4096, 12288, 1805U, ActivationCompute::A16},
                                kOrnithA16Cases);
        failures +=
            run_profile("LinearSwiGLU NVFP4_A4",
                        {QType::NVFP4, 34816, 5120, 17408, 1803U, ActivationCompute::A4}, kA4Cases, std::array<std::int32_t, 4>{65, 97, 128, 129});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU NVFP4 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU NVFP4 test failed: " << error.what() << '\n';
        return 1;
    }
}
