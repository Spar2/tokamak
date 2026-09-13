from __future__ import annotations

import struct

import torch

from tools.artifact.layouts import decode_nvfp4_words, encode_nvfp4
from tools.convert.ornith_1_5_9b import inventory as groupwise_inventory
from tools.convert.ornith_1_5_9b import inventory_nvfp4 as inventory
from tools.convert.ornith_1_5_9b import recipe_nvfp4 as recipe
from tools.convert.ornith_1_5_9b.source import OrnithShardReader


def test_mixed_mlp_only_inventory():
    assert inventory.MODEL_ID == "ornith-1.5-9b"
    assert inventory.WEIGHTS_ID == "nvfp4"
    assert groupwise_inventory.WEIGHTS_ID == "groupwise-int"
    assert len(inventory.OBJECT_SPECS) == len(groupwise_inventory.OBJECT_SPECS) + 64
    assert len(inventory.NVFP4_TENSOR_SPECS) == 64
    assert len(inventory.INPUT_SCALE_DIVISOR_SPECS) == 64
    tensors = {spec.name: spec for spec in inventory.TENSOR_SPECS}
    assert tensors["text/layers/0/mlp/gate_up"].format == "NVFP4"
    assert tensors["text/layers/0/mlp/gate_up"].layout == "blockscale-k16-m128x4-v1"
    assert tensors["text/layers/0/mlp/gate_up"].shape == (24576, 4096)
    assert tensors["text/layers/0/mlp/down"].format == "NVFP4"
    assert tensors["text/layers/0/mlp/down"].shape == (4096, 12288)
    assert tensors["text/layers/3/attention/query_key"].format == "Q4G64_F16S"
    assert tensors["text/layers/0/gdn/value_z"].format == "Q5G64_F16S"
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
