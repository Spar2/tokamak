# Dynamic W4A4 graduation note (Ornith MLP, T>=128)

## Why the artifact `input_scale_divisor = 1.0` is unrelated

The `ornith-1.5-9b/nvfp4` artifact stores `input_scale_divisor = 1.0` next to
every NVFP4 MLP parent because `validate_nvfp4_weight` requires a positive
field. The Ornith checkpoint (`W4A16_NVFP4`, MLP `input_quantizer` disabled)
ships **no** `input_global_scale`. The dynamic path **never reads** the
artifact field (unit-tested: `input_scale_divisor = 9.0` yields bit-identical
output). Runtime activation scale is `d_x = 1` by definition (see contract).

## Dynamic activation scale contract

Per token-column, per K16 group: `scale_e4m3 = satfinite(max_abs(K16) / 6)`,
`codes = e2m1(x / decode(scale))`, `d_x = 1` (no extra global scale).
GEMM `alpha = 1 / (d_x * d_w) = 1 / d_w`. Weights are native checkpoint words;
`d_w = 1 / weight_scale_2` (ModelOpt multiply-scale inversion, same as W4A16).

## Threshold selection

- T <= 16: fused/small-T W4A16 (proven decode wins: +15/+47% gate_up, +52/+135% down).
- 32 <= T < 128: W4A16 BF16-MMA (recovers the tiled-A16 collapse; T=32 down still
  -7% vs Q5-MMA but bounded and prefill-minor).
- T >= 128: dynamic W4A4 (`kNvfp4OrnithDynamicW4a4MinT`). Prefill chunks (1024)
  and C8 MTP-verify-adjacent widths live here; decode (T<=8) and MTP-verify
  (T<=32 at C8) never touch dynamic. Threshold stays 128: no evidence supports
  lowering (verify path untested for acceptance below 128), and prefill is
  where the prize is.

## Quality evidence

- Real-activation replay (6 layer/input combos): cosine 0.995-0.999,
  rel 0.05-0.10, within gates incl. 106-magnitude outlier layer.
- Full-corpus perplexity: GW 6.220 -> W4A16 6.327 (+1.7%) -> dyn 6.367 (+2.4%
  overall, <=2.1% per domain). English-ref delta is weights-driven (shared
  with W4A16), not activation-driven.
- Acceptance, 11 prompts total (3 C1 legs + 8 C8-distinct x2 policies):
  dynamic mean -3.5pp vs W4A16, worst lane parity (both ~58% on code-heavy
  content, byte-identical 162 accepted). The "suspicious" C8 lanes reproduced
  under W4A16 control: content-driven, not policy-driven.
- Reasoning coherence: 1024-token greedy runs diverge at ~40-180 chars, both
  coherent/on-topic (normal quantization shift, not collapse).

## Fallback behavior

`NINFER_ORNITH_DYNAMIC_W4A4`: unset/`1`/`full` -> full dynamic; `gate_only` ->
dynamic gate_up, W4A16 down (prefill +95% vs W4A16, acceptance at parity);
`0`/`off`/`no` -> pure W4A16. W4A16 small-T and BF16-MMA paths are intact and
always available. Scope: Ornith MLP only; 27B routes never consult the gate.

## Known limitations

- Acceptance sample: 11 prompts, greedy only; stochastic acceptance unmeasured.
- TMA-Ornith path (T>=1024 %256==0) numerically covered at T=1024 only.
- R64C128 W4A16 fallback still trails Q4/Q5 MMA at T>=128 (irrelevant when
  dynamic is on; matters only under `0`/fallback).
- SwiGLU-dynamic uses M64/M96/M128 schedules incl. M96N128 (kBlockM=96);
  T=64 schedule hole referenced in earlier notes is covered by tests
  (T=64/96/97 ladder points pass).
