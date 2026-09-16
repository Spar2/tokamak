#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <std::int32_t QueryRows, std::int32_t KeyRows, std::int32_t ValueRows,
          std::int32_t ZRows>
struct Fp8GdnInputSplitOutput {
    static constexpr std::int32_t kQueryRows = QueryRows;
    static constexpr std::int32_t kKeyRows   = KeyRows;
    static constexpr std::int32_t kValueRows = ValueRows;
    static constexpr std::int32_t kQkvRows   = kQueryRows + kKeyRows + kValueRows;
    static constexpr std::int32_t kZRows     = ZRows;
    static constexpr std::int32_t kRows      = kQkvRows + kZRows;

    __nv_bfloat16* qkv;
    __nv_bfloat16* z;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kQkvRows) {
            return qkv + static_cast<std::int64_t>(token) * kQkvRows + parent_row;
        }
        return z + static_cast<std::int64_t>(token) * kZRows + parent_row - kQkvRows;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        *destination(parent_row, token) = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

static_assert(Fp8GdnInputSplitOutput<2048, 2048, 6144, 6144>::kRows == 16384);
static_assert((Fp8GdnInputSplitOutput<2048, 2048, 6144, 6144>::kQkvRows % 128) == 0);
static_assert((Fp8GdnInputSplitOutput<2048, 2048, 6144, 6144>::kZRows % 128) == 0);

// 5120-wide (27B) parent split; unchanged production behavior.
using Fp8GdnInputOutput = Fp8GdnInputSplitOutput<2048, 2048, 6144, 6144>;
// Ornith-1.5-9B parent split (qkv 8192 = q2048+k2048+v4096, z 4096).
using Fp8GdnInput4096Output = Fp8GdnInputSplitOutput<2048, 2048, 4096, 4096>;
static_assert(Fp8GdnInput4096Output::kRows == 12288);

} // namespace ninfer::ops::detail
