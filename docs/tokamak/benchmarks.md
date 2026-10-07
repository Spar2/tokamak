# Benchmark methodology

## Rig

- GPU: NVIDIA GeForce RTX 5060 Ti 16 GB (36 SMs, `cc 12.0`), driver + CUDA Toolkit 13.1
- Build: Release, Ninja, `CMAKE_CUDA_ARCHITECTURES=120a`, `BUILD_TESTING=ON`
- Timing: CUDA-event device time **and** wall time are recorded together; tables use
  device-time medians unless noted. Our rig is Linux containers (Ubuntu 24.04) on a
  Windows 11 WDDM host, where wall time varies ±2–3× run to run —
  device time is the stable comparator, wall is reported alongside for honesty.
- Activations for op microbenchmarks: deterministic synthetic ramp (`SYNTHETIC-RAMP`).
  Two committed methods exist: the crossover series (warmup 5, repeat 20,
  `results/crossover_*.csv` on `experiment/ornith-dynamic-w4a4`) and the MMA/validation
  reruns (warmup 20, repeat 50). Full-model runs use the locally converted artifact plus
  recorded oracle vectors; self-consistency hashes (FNV) are checked rep-to-rep.

## Environment gates

| Gate | Default | Effect when set |
|---|---|---|
| `NINFER_T2_PREFILL_P1` | unset (legacy path) | `=1` routes T≥8 prefill through P1-SYNC kernels |
| `NINFER_ORNITH_DYNAMIC_W4A4` | full dynamic at T≥128 | `gate_only` keeps down-projection on W4A16; `0/off/no` is pure W4A16 |

## Bonsai-27B prefill (full model, chain63, T=33 unless noted)

| Tokens | Legacy (ms) | P1-SYNC (ms) | Speedup | Note |
|---|---:|---:|---:|---|
| T=8 | 548 | 250 | 2.2× | sync8 path |
| T=32 | 1673 | 488 | 3.4× | sync32 path |
| T=33 | ~1900 | 540 | 3.5× | masked tail; layer-52 maxabs 9.60 legacy (FAIL) → 1.25 P1 (PASS) |
| T=64 | 2917 | 426 | 6.8× | |

Top-1 exact at T=32 (`506 == 506`) and T=33 (`271 == 271`) — archived
`bq2/chain63_AUTH_ON2.log` (local run logs; rerun via `ninfer_bq2_chain63_test`
with `NINFER_T2_PREFILL_P1=1`).
Determinism: `mindet` 0/1000 fails (synthetic), `mindet_art` 0/1000 (artifact weights),
reproduced on the showcase build.

## Bonsai-27B decode (T1-A, chain_gen, n=128)

min 70.07, p10 71.69, **median 75.07**, p90 80.96, max 92.29 ms/token.
Generation bit-exact over 32 reset steps; near-ties excused: 0.

## Ornith-9B MLP crossover (gate_up 24576×4096, medians, µs)

| T | Q4-prod | NVFP4 tiled W4A16 | NVFP4 BF16-MMA |
|---|---:|---:|---:|
| 1 | 156.7 | 134.8 | — |
| 2 | 167.0 | 140.4 | — |
| 4 | 191.6 | 160.8 | — |
| 8 | 298.0 | 246.9 | — |
| 16 | 584.8 | 416.8 | — |
| 32 | 588.9 | 812.1 | **453** |
| 64 | 599.1 | 1597.4 | **462** |
| 128 | 588.4 | 3161.5 | **722** |
| 256 | 1068.1 | 6378.8 | — (MMA not swept; tiled path stays off past T=32) |
| 512 | 2109.6 | 12918.1 | — |

MMA column: `notes/large-t-w4a16-mma.md` qualification runs (same rig, warmup 5 /
repeat 20 series extended). `down` 4096×12288 from the same note: T=32 Q5 333 vs
MMA 360, T=64 Q5 340 vs MMA **292**, T=128 Q5 337 vs MMA 422.

`down` 4096×12288 from the same note: T=32 Q5 333 vs MMA 360 (still trailing),
T=64 Q5 340 vs MMA **292**, T=128 Q5 337 vs MMA 422.
Raw tiled series: `results/crossover_gate_up.csv`, `results/crossover_down.csv` on the
`experiment/ornith-dynamic-w4a4` branch (warmup 5, repeat 20); MMA column from the
qualification note `notes/large-t-w4a16-mma.md` (same campaign). An independent rerun
on 7 Oct (warmup 20, repeat 50) confirmed gate_up T=32 at 446 (≈1.5%).

## Ornith quality (dynamic W4A4)

- Real-activation replay (6 layer/input combos): cosine 0.995–0.999, rel 0.05–0.10;
- Full-corpus perplexity: GW 6.220 → W4A16 6.327 (+1.7%) → dynamic 6.367 (+2.4% overall);
- Acceptance, 11 prompts (greedy only; stochastic decoding unmeasured): dynamic mean
  −3.5pp vs W4A16, worst lane parity.

## Ornith FP8 GDN (contract proof, no throughput claim yet)

- `ninfer_linear_fp8_modelopt_contract_test`: static-vs-A16 broadcast rel 0.025 — OK;
- `ninfer_linear_nvfp4_a16_test` regression on the same build — OK
  (heads geometries 248320/131072 vs exact-dequant oracle, T=1..32).

## Concurrency-9

No throughput claim — this track raises the batch ceiling (8→9) across GDN replay,
attention cache, engine, and serve layers, with B=9 oracle cases green in
`gated_delta_net`, `gdn_replay_records`, `gdn_input_proj_conv_record`,
`gated_delta_net_replay_record`, and `softmax_attention` tests, plus C9TRACE
scheduler diagnostics for 9-wide debugging.
