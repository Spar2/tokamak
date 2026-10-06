#pragma once

// ninfer::ops::detail - private launch prototype for gdn_gating.

#include "core/tensor.h"
#include "ninfer/ops/gdn_gating.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void gdn_gating_launch(const Tensor& a, const Tensor& b, const Tensor& A, const Tensor& dt_bias,
                       GdnGateFormula formula, Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace ninfer::ops::detail
