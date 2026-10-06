# Large-T options and W4A4 d_x (Ornith W4A16 checkpoint)

## 27B production W4A4

Qwen3.8-27B NVFP4 artifacts store a calibrated `input_global_scale` as `d_x`.
The binder must not default it to 1. Quantize kernel:

    scale_unencoded = d_x * max_abs_K16 / 6
    alpha = 1 / (d_x * d_w)

`d_x` is a **site-wide calibrated divisor**, not a dummy.

## Ornith checkpoint

`hf_quant_config.json` marks MLP as `W4A16_NVFP4`. There is no `input_global_scale`.
Artifact placeholder `input_scale_divisor=1.0` exists only because `validate_nvfp4_weight`
requires a positive field for A16. It is **not** a ModelOpt activation scale.

`nvfp4_w4a4.cu` currently throws for 24576x4096 and 4096x12288.

## Options considered

| Option | Correctness | Scope | Decision |
|---|---|---|---|
| A/B Dynamic W4A4 with per-K16 max_abs and explicit `d_x=1` meaning "no extra global scale" | Well-defined, **different** from 27B calibrated d_x; needs A4-tolerance tests | Reuse 27B W4A4 templates + new K=4096/12288 activation geometry | Not enabled. Must not silently reuse the A16 placeholder. |
| C Calibrated d_x | No source field | Would need a calibration pass | Rejected for this checkpoint. |
| D NVFP4-weight dequant + BF16 MMA | Matches W4A16 semantics | New kernel, similar to W8 row-split GEMM | Highest-correctness large-T candidate; not implemented this session. |
| E Extra groupwise MLP copy as fallback | Correct but ~GB of 16GB | Rejected unless measured. | Rejected. |

## Rule

Do not dispatch AllowA4 on Ornith MLP until a numeric test of the chosen `d_x` policy
passes against the A16 oracle. Prefill still uses tiled A16 until that lands.
