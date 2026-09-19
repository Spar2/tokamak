from __future__ import annotations

import pytest
import torch

from tools.artifact.layouts import (
    decode_row_split_codes,
    dequantize_row_split,
    encode_row_split,
    row_split_geometry,
)


def _f16(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float16)


def test_t2g128_geometry_known_values():
    geometry = row_split_geometry("T2G128_F16S", (5120, 17408))
    assert geometry.groups_per_row == 136
    assert geometry.base_bytes_per_group == 32
    assert geometry.high_bytes_per_group == 0
    assert geometry.base_row_bytes == 136 * 32
    assert geometry.scale_row_bytes == 136 * 2
    assert geometry.payload_bytes == 5120 * 136 * 32 + 5120 * 136 * 2


def test_t2g128_pack_order_matches_prism():
    # codes 0,1,2,3,0,1,2,3,... over one group must pack LSB-first:
    # byte j//4 holds codes 4k..4k+3 at shifts 0,2,4,6.
    codes = (torch.tensor([[0, 1, 2, 3] * 32], dtype=torch.int8) - 1).reshape(1, 1, 128)  # signed logical
    scales = _f16(1.0).reshape(1, 1)
    payload = encode_row_split(codes, scales, "T2G128_F16S", (1, 128))
    base = payload[:32]
    assert bytes(base[:2]) == bytes((0b11100100, 0b11100100))
    out_scales, out_codes = decode_row_split_codes(payload, "T2G128_F16S", (1, 128))
    assert (out_codes.reshape(-1) == codes.reshape(-1)).all()


def test_t2g128_signed_round_trip_and_dequant_equation():
    torch.manual_seed(0)
    codes = torch.randint(-1, 2, (4, 2, 128), dtype=torch.int8)  # logical -1/0/+1
    scales = torch.rand(4, 2, dtype=torch.float16)
    payload = encode_row_split(codes, scales, "T2G128_F16S", (4, 256))
    _, back = decode_row_split_codes(payload, "T2G128_F16S", (4, 256))
    assert (back == codes).all()
    dq = dequantize_row_split(payload, "T2G128_F16S", (4, 256), dtype=torch.float32)
    expect = (codes.float() * scales.float().unsqueeze(-1).expand(4, 2, 128)).reshape(4, 256)
    assert torch.equal(dq, expect)


def test_t2g128_reserved_code_detectable():
    # code 3 (+2) is encodable but must be detectable by census, never silent.
    codes = torch.full((1, 1, 128), 2, dtype=torch.int8)  # logical +2
    scales = _f16(0.5).reshape(1, 1)
    payload = encode_row_split(codes, scales, "T2G128_F16S", (1, 128))
    _, back = decode_row_split_codes(payload, "T2G128_F16S", (1, 128))
    assert (back == 2).all()
    dq = dequantize_row_split(payload, "T2G128_F16S", (1, 128), dtype=torch.float32)
    assert (dq == 1.0).all()  # 2 * 0.5: mapping is mechanical, policy rejects it
