"""Closed single-source recipe for Ornith-1.5-9B native FP8 GDN objects.

Each GDN layer contributes one fused row-scaled FP8 parent
``text/layers/{L}/gdn/query_key_value_z`` (12288x4096) built losslessly from
the checkpoint ``in_proj_qkv`` (8192x4096) and ``in_proj_z`` (4096x4096)
tensors: raw E4M3 words are concatenated and each row scale broadcasts its
tensor's scalar ``weight_scale``. No BF16 round-trip, no requantization.

The checkpoint ``input_scale`` scalars are intentionally not preserved: the
A16 production path consumes BF16 activations directly, and synthetic
outlier analysis rejected vendor-static activation quantization for this
operator. The checkpoint remains the source of truth if a future static
route needs them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tools.convert.common.safetensors import ShardReader

from . import inventory_nvfp4 as inventory


QKV_ROWS = 8192
Z_ROWS = 4096
PARENT_ROWS = QKV_ROWS + Z_ROWS
PARENT_K = 4096


@dataclass(frozen=True, slots=True)
class Fp8GdnSource:
    name: str
    shape: tuple[int, int]

    def field(self, suffix: str) -> str:
        return f"{self.name}.{suffix}"


@dataclass(frozen=True, slots=True)
class Fp8GdnWeightRecipe:
    object_name: str
    shape: tuple[int, int]
    qkv: Fp8GdnSource
    z: Fp8GdnSource


def _build_recipes() -> tuple[Fp8GdnWeightRecipe, ...]:
    recipes: list[Fp8GdnWeightRecipe] = []
    for layer in inventory.GDN_LAYERS:
        source_prefix = f"model.language_model.layers.{layer}.linear_attn."
        recipes.append(
            Fp8GdnWeightRecipe(
                f"text/layers/{layer}/gdn/query_key_value_z",
                (PARENT_ROWS, PARENT_K),
                Fp8GdnSource(source_prefix + "in_proj_qkv", (QKV_ROWS, PARENT_K)),
                Fp8GdnSource(source_prefix + "in_proj_z", (Z_ROWS, PARENT_K)),
            )
        )
    return tuple(recipes)


FP8_GDN_WEIGHT_RECIPES = _build_recipes()
FP8_GDN_WEIGHTS_BY_NAME = {item.object_name: item for item in FP8_GDN_WEIGHT_RECIPES}


def _raw_fp8_words(reader: ShardReader, source: Fp8GdnSource) -> torch.Tensor:
    getter = getattr(reader, "get_raw", reader.get)
    words = getter(source.field("weight"))
    if words.dtype != torch.float8_e4m3fn or tuple(words.shape) != source.shape:
        raise TypeError(
            f"{source.field('weight')}: expected F8_E4M3 "
            f"{source.shape}, got {words.dtype} {tuple(words.shape)}"
        )
    return words.view(torch.uint8).contiguous()


def _row_scale(reader: ShardReader, source: Fp8GdnSource, rows: int) -> torch.Tensor:
    scale = reader.get(source.field("weight_scale"))
    if scale.dtype != torch.float32 or scale.numel() != 1:
        raise ValueError(f"{source.field('weight_scale')}: expected one F32 value")
    value = float(scale.reshape(()).item())
    if not (value > 0.0 and value != float("inf")):
        raise ValueError(f"{source.field('weight_scale')}: scale must be finite positive")
    return torch.full((rows,), value, dtype=torch.float32).to(torch.bfloat16)


def materialize_fp8_gdn_weight(
    selected: Fp8GdnWeightRecipe,
    reader: ShardReader,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (raw E4M3 code words, BF16 row scales) for the fused parent."""
    qkv_words = _raw_fp8_words(reader, selected.qkv)
    z_words = _raw_fp8_words(reader, selected.z)
    codes = torch.cat((qkv_words, z_words), dim=0)
    scales = torch.cat(
        (
            _row_scale(reader, selected.qkv, QKV_ROWS),
            _row_scale(reader, selected.z, Z_ROWS),
        ),
        dim=0,
    )
    return codes, scales


def validate_recipe() -> None:
    if len(FP8_GDN_WEIGHT_RECIPES) != len(inventory.GDN_LAYERS):
        raise ValueError("Ornith FP8 GDN source recipe is incomplete")
    if tuple(FP8_GDN_WEIGHTS_BY_NAME) != tuple(
        spec.name for spec in inventory.FP8_GDN_TENSOR_SPECS
    ):
        raise ValueError("FP8 GDN weight recipe order does not match inventory")
    for selected in FP8_GDN_WEIGHT_RECIPES:
        if selected.shape != (PARENT_ROWS, PARENT_K):
            raise ValueError(f"{selected.object_name}: invalid fused row geometry")
        if selected.qkv.shape != (QKV_ROWS, PARENT_K):
            raise ValueError(f"{selected.object_name}: invalid qkv source geometry")
        if selected.z.shape != (Z_ROWS, PARENT_K):
            raise ValueError(f"{selected.object_name}: invalid z source geometry")


validate_recipe()


__all__ = [
    "FP8_GDN_WEIGHTS_BY_NAME",
    "FP8_GDN_WEIGHT_RECIPES",
    "PARENT_K",
    "PARENT_ROWS",
    "QKV_ROWS",
    "Z_ROWS",
    "materialize_fp8_gdn_weight",
    "validate_recipe",
]
