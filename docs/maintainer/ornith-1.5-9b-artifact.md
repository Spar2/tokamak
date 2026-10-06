# Ornith-1.5-9B Artifact Reference

Registered Ornith identities are `ornith-1.5-9b/groupwise-int` and `ornith-1.5-9b/nvfp4`, with
target key `ornith_1_5_9b`. Runtime uses the Qwen3.5-9B-shaped family execution package: 32 layers,
24 Gated DeltaNet layers, 8 full-attention layers, and one MTP layer. Both public artifacts keep
the complete Text, Vision, MTP, and frontend resource inventory. The `nvfp4` identity stores MLP
`gate_up` `[24576,4096]` and `down` `[4096,12288]` as `blockscale-k16-m128x4-v1` NVFP4 and keeps
attention, GDN, MTP, embeddings, and vision on the groupwise-int formats.

## Source and conversion

The source is the official
[`ornith-ai/Ornith-1.5-9B-NVFP4`](https://huggingface.co/ornith-ai/Ornith-1.5-9B-NVFP4)
checkpoint. The converter is [`tools/convert/ornith_1_5_9b/`](../../tools/convert/ornith_1_5_9b/):

```bash
python3 -m tools.convert.ornith_1_5_9b.convert \
  --model /path/to/Ornith-1.5-9B-NVFP4 \
  --out models/ornith_1_5_9b.ninfer

python3 -m tools.convert.ornith_1_5_9b.convert_nvfp4 \
  --model /path/to/Ornith-1.5-9B-NVFP4 \
  --out models/ornith_1_5_9b_nvfp4.ninfer
```

The checkpoint contains one safetensors file with NVFP4 U8 MLP/output-head weights and FP8
linear-attention weights. `source.py` decodes those representations on demand to logical BF16;
the normal groupwise-int converter then emits NInfer Q4/Q5/Q6/W8 tensors. `convert_nvfp4` does not
BF16-roundtrip MLP packed weights: it concatenates native U8 codes, swizzles E4M3 block scales,
and writes a dummy `input_scale_divisor=1.0` required by the NVFP4 weight contract. Execution is
W4A16 (`LinearPolicy::A16Only`); W4A4 is not enabled. The MTP tensors are already BF16 in the
source checkpoint and use the registered W8 MTP profile.

The six frontend resources are pinned independently from Qwen3.5-9B. The Ornith chat template
digest is registered as `ThinkingToggle`, preserving `<think>` reasoning and final-answer
separation for CLI and HTTP serving. The source tokenizer config's stale embedded Qwen template is
validated by source hash and normalized to the adjacent Ornith template before artifact writing.

## Verification

```bash
python3 -m tools.convert.ornith_1_5_9b.verify \
  --artifact models/ornith_1_5_9b.ninfer \
  --model /path/to/Ornith-1.5-9B-NVFP4

python3 -m tools.convert.ornith_1_5_9b.verify_nvfp4 \
  --artifact models/ornith_1_5_9b_nvfp4.ninfer \
  --model /path/to/Ornith-1.5-9B-NVFP4
```

The verifier checks artifact identity, complete object order/counts, representative direct
payloads against the dequantizing source reader, and representative row-split quantized payloads.
