#pragma once

#include "ninfer/ops/linear.h"
#include "ops/linear/t2/t2_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

T2Launch select_t2_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

void t2_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream);

// M7-P1 scratch integration (reversible, env-gated):
//   NINFER_T2_PREFILL_P1=1 routes exact tiles (N%64==0, K%128==0) to P1-SYNC:
//   T=8 -> P1-SYNC-8; T=32/64/128 -> P1-SYNC (all fully synchronous).
//   Every other T keeps the existing row-persistent/chunked path.
//   Init contract (churn hazard): persistent no-churn T2 buffers + mode-E
//   (sync + sacrificial stateless T2 warmup + sync) at startup; serving
//   reuses buffers. Env is read on every call so routing tests can toggle
//   it in-process. t2_dispatch additionally guards on BF16 contiguity and
//   silently falls back when the prototype's requirements are not met
//   (never throws into the model). Stats report (p1, fallback) launch
//   counts since process start.
[[nodiscard]] bool t2_prefill_p1_enabled();
[[nodiscard]] bool t2_prefill_p1_applies(std::int32_t n, std::int32_t k, std::int32_t t);
void t2_prefill_p1_stats(std::uint64_t& p1, std::uint64_t& fallback);

} // namespace ninfer::ops::detail
