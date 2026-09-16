from __future__ import annotations

import struct

import torch

from tools.artifact.layouts import decode_nvfp4_words, encode_nvfp4
from tools.artifact.layouts import decode_fp8_row_scaled_words, encode_fp8_row_scaled
from tools.convert.ornith_1_5_9b import inventory as groupwise_inventory
from tools.convert.ornith_1_5_9b import inventory_nvfp4 as inventory
from tools.convert.ornith_1_5_9b import recipe_fp8gdn
from tools.convert.ornith_1_5_9b import recipe_nvfp4 as recipe
from tools.convert.ornith_1_5_9b.source import OrnithShardReader


def test_mixed_mlp_only_inventory():
    assert inventory.MODEL_ID == "ornith-1.5-9b"
    assert inventory.WEIGHTS_ID == "nvfp4"
    assert groupwise_inventory.WEIGHTS_ID == "groupwise-int"
    assert len(inventory.OBJECT_SPECS) == len(groupwise_inventory.OBJECT_SPECS) + 88
    assert len(inventory.NVFP4_TENSOR_SPECS) == 64
    assert len(inventory.FP8_GDN_TENSOR_SPECS) == 24
    assert len(inventory.INPUT_SCALE_DIVISOR_SPECS) == 64
    tensors = {spec.name: spec for spec in inventory.TENSOR_SPECS}
    assert tensors["text/layers/0/mlp/gate_up"].format == "NVFP4"
    assert tensors["text/layers/0/mlp/gate_up"].layout == "blockscale-k16-m128x4-v1"
    assert tensors["text/layers/0/mlp/gate_up"].shape == (24576, 4096)
    assert tensors["text/layers/0/mlp/down"].format == "NVFP4"
    assert tensors["text/layers/0/mlp/down"].shape == (4096, 12288)
    assert tensors["text/layers/3/attention/query_key"].format == "Q4G64_F16S"
    assert tensors["text/layers/0/gdn/value_z"].format == "Q5G64_F16S"
    assert tensors["text/layers/0/gdn/query_key_value_z"].format == "FP8_E4M3FN_ROW_BF16S"
    assert tensors["text/layers/0/gdn/query_key_value_z"].layout == "row-scale-v1"
    assert tensors["text/layers/0/gdn/query_key_value_z"].shape == (12288, 4096)
    assert tensors["text/layers/30/gdn/query_key_value_z"].format == "FP8_E4M3FN_ROW_BF16S"
    assert tensors["mtp/layer/mlp/gate_up"].format == "W8G32_F16S"
    assert tensors["text/token_embedding"].format == "Q6G64_F16S"
    assert (
        tensors["text/layers/0/mlp/gate_up_projection/input_scale_divisor"].format == "FP32"
    )


def test_recipe_covers_inventory_order():
    recipe.validate_recipe()
    assert tuple(recipe.NVFP4_WEIGHTS_BY_NAME) == tuple(
        spec.name for spec in inventory.NVFP4_TENSOR_SPECS
    )
    assert tuple(recipe.INPUT_DIVISORS_BY_NAME) == tuple(
        spec.name for spec in inventory.INPUT_SCALE_DIVISOR_SPECS
    )


def test_placeholder_input_divisor():
    scalar = recipe.materialize_input_divisor(recipe.INPUT_DIVISOR_RECIPES[0])
    assert scalar.dtype == torch.float32
    assert float(scalar.item()) == 1.0


def test_gate_up_concat_round_trip():
    gate = torch.arange(12288 * 2048, dtype=torch.uint8).reshape(12288, 2048)
    up = (torch.arange(12288 * 2048, dtype=torch.uint8) + 3).reshape(12288, 2048)
    packed = torch.cat((gate, up), dim=0)
    scales = torch.full((24576, 256), 0x3C, dtype=torch.uint8)
    divisor = struct.pack("<f", 0.000106085)
    payload = encode_nvfp4(packed, scales, divisor, (24576, 4096))
    decoded_packed, decoded_scales, decoded_divisor = decode_nvfp4_words(
        payload, (24576, 4096)
    )
    assert torch.equal(decoded_packed, packed)
    assert torch.equal(decoded_scales, scales)
    assert bytes(decoded_divisor.reshape(1).view(torch.uint8).numpy()) == divisor


def test_source_reader_exposes_raw_packed_getter():
    assert hasattr(OrnithShardReader, "get_raw")


def test_fp8_gdn_recipe_covers_inventory_order():
    recipe_fp8gdn.validate_recipe()
    assert tuple(recipe_fp8gdn.FP8_GDN_WEIGHTS_BY_NAME) == tuple(
        spec.name for spec in inventory.FP8_GDN_TENSOR_SPECS
    )


def test_fp8_gdn_fused_round_trip():
    # Finite E4M3 codes only (0x7F/0xFF are NaN and correctly rejected).
    qkv = (torch.arange(8192 * 64) % 126).to(torch.uint8).reshape(8192, 64)
    z = ((torch.arange(4096 * 64) + 5) % 126).to(torch.uint8).reshape(4096, 64)
    codes = torch.cat((qkv, z), dim=0)
    scales = torch.cat(
        (
            torch.full((8192,), 0.0008, dtype=torch.float32).to(torch.bfloat16),
            torch.full((4096,), 0.0006, dtype=torch.float32).to(torch.bfloat16),
        ),
        dim=0,
    )
    payload = encode_fp8_row_scaled(codes, scales, (12288, 64))
    decoded_codes, decoded_scales = decode_fp8_row_scaled_words(payload, (12288, 64))
    assert torch.equal(decoded_codes, codes)
    assert torch.equal(decoded_scales, scales)
