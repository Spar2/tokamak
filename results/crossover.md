# Ornith MLP T-crossover (RTX 5060 Ti, ops::linear, synthetic activations)

Known-good base: `ornith-nvfp4-mlp-working` (`c9e4d4bc`).

NVFP4 path is current production A16 (decode / small-T / tiled at 16). Groupwise is Q4 gate_up / Q5 down.

Source `weight_scale_2` is printed by the bench as the ModelOpt multiply scale; kernel latency does not depend on it.

## gate_up 24576 x 4096

| T | Q4 us | NVFP4 A16 us | NV vs Q4 |
|---:|---:|---:|---:|
| 1 | 157 | 135 | **+16%** |
| 2 | 167 | 140 | +16% |
| 4 | 192 | 161 | +16% |
| 8 | 298 | 247 | +17% |
| 16 | 585 | 417 | +29% |
| 32 | 589 | 812 | **-38%** |
| 64 | 599 | 1597 | -167% |
| 128 | 588 | 3162 | -437% |
| 256 | 1068 | 6379 | -497% |
| 512 | 2110 | 12918 | -512% |

Crossover: **between T=16 and T=32**. Q4 MMA saturates ~0.59 ms from T=32–128. NVFP4 A16 tiles and scales linearly (~8 TFLOP/s cap).

## down 4096 x 12288

| T | Q5 us | NVFP4 A16 us | NV vs Q4/Q5 |
|---:|---:|---:|---:|
| 1 | 108 | 71 | **+34%** |
| 2 | 114 | 81 | +29% |
| 4 | 122 | 87 | +29% |
| 8 | 271 | 136 | +50% |
| 16 | 503 | 216 | +57% |
| 32 | 339 | 411 | **-21%** |
| 64 | 343 | 798 | -132% |
| 128 | 339 | 1560 | -360% |
| 256 | 630 | 3087 | -390% |
| 512 | 1218 | 6219 | -411% |

Same crossover band.

## Implication

T=1/4 wins explain MTP C1. Tiled T=16 cannot beat Q4/Q5 MMA at prefill T~300+ or decode batch 8. Large-T needs a Tensor Core route, not more A16 tiles.
