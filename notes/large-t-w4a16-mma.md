# Ornith NVFP4 W4A16 large-T MMA (RTX 5060 Ti)

Dequant packed E2M1+E4M3 tiles into BF16 shared memory, then `m16n8k16` BF16 MMA.
Activations stay BF16. Artifact and `d_w = 1/weight_scale_2` are unchanged. No W4A4.

Dispatch (measured, both parents):

- T <= 16: existing fused/small-T A16
- T >= 32: W4A16 MMA (`kNvfp4OrnithMmaMinT`)

T=16 A16 remains faster than pushing MMA down. T=17-31 still tiles A16 SwiGLU at 16.

Kernel vs Q4/Q5 `ops::linear` median us (NVFP4 column is MMA for T>=32):

| T | gate_up Q4 | gate_up NV | down Q5 | down NV |
|---:|---:|---:|---:|---:|
| 16 | 586 | 415 (A16) | 501 | 216 (A16) |
| 32 | 589 | 453 | 333 | 360 |
| 64 | 597 | 462 | 340 | 292 |
| 128 | 589 | 722 | 337 | 422 |

Numeric: `ninfer_linear_nvfp4_a16_test`, `ninfer_linear_add_nvfp4_test`, `ninfer_linear_swiglu_nvfp4_test` including Ornith T=32/64/128.
