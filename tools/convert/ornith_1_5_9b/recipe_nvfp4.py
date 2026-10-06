"""Closed single-source recipe for Ornith-1.5-9B native MLP NVFP4 objects."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable

import torch

from tools.artifact.numeric import valid_positive_fp32_word
from tools.convert.common.safetensors import ShardReader

from . import inventory_nvfp4 as inventory


PLACEHOLDER_INPUT_SCALE_DIVISOR = 1.0


@dataclass(frozen=True, slots=True)
class RowRange:
    begin: int
    end: int

    @property
    def rows(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True, slots=True)
class Nvfp4Source:
    name: str
    shape: tuple[int, int]

    def field(self, suffix: str) -> str:
        return f"{self.name}.{suffix}"


@dataclass(frozen=True, slots=True)
class Nvfp4Part:
    source: Nvfp4Source
    rows: tuple[RowRange, ...]

    @property
    def output_rows(self) -> int:
        return sum(item.rows for item in self.rows)


@dataclass(frozen=True, slots=True)
class Nvfp4WeightRecipe:
    object_name: str
    shape: tuple[int, int]
    parts: tuple[Nvfp4Part, ...]
    divisor_sources: tuple[Nvfp4Source, ...]


@dataclass(frozen=True, slots=True)
class InputDivisorRecipe:
    object_name: str
    sources: tuple[Nvfp4Source, ...]
    weight_names: tuple[str, ...]


def _source(name: str, n: int, k: int) -> Nvfp4Source:
    return Nvfp4Source(name, (n, k))


def _all(source: Nvfp4Source) -> Nvfp4Part:
    return Nvfp4Part(source, (RowRange(0, source.shape[0]),))


def _build_recipes() -> tuple[
    tuple[Nvfp4WeightRecipe, ...],
    tuple[InputDivisorRecipe, ...],
    tuple[tuple[Nvfp4Source, ...], ...],
]:
    weights: list[Nvfp4WeightRecipe] = []
    inputs: list[InputDivisorRecipe] = []
    weight_groups: list[tuple[Nvfp4Source, ...]] = []
    for layer in range(32):
        source_prefix = f"model.language_model.layers.{layer}.mlp."
        object_prefix = f"text/layers/{layer}/mlp/"
        gate = _source(source_prefix + "gate_proj", 12288, 4096)
        up = _source(source_prefix + "up_proj", 12288, 4096)
        down = _source(source_prefix + "down_proj", 4096, 12288)
        gate_up_group = (gate, up)
        weight_groups.append(gate_up_group)
        weights.extend(
            (
                Nvfp4WeightRecipe(
                    object_prefix + "gate_up",
                    (24576, 4096),
                    (_all(gate), _all(up)),
                    gate_up_group,
                ),
                Nvfp4WeightRecipe(
                    object_prefix + "down",
                    down.shape,
                    (_all(down),),
                    (down,),
                ),
            )
        )
        inputs.extend(
            (
                InputDivisorRecipe(
                    object_prefix + "gate_up_projection/input_scale_divisor",
                    gate_up_group,
                    (object_prefix + "gate_up",),
                ),
                InputDivisorRecipe(
                    object_prefix + "down_projection/input_scale_divisor",
                    (down,),
                    (object_prefix + "down",),
                ),
            )
        )
    return tuple(weights), tuple(inputs), tuple(weight_groups)


NVFP4_WEIGHT_RECIPES, INPUT_DIVISOR_RECIPES, WEIGHT_DIVISOR_GROUPS = _build_recipes()
NVFP4_WEIGHTS_BY_NAME = {item.object_name: item for item in NVFP4_WEIGHT_RECIPES}
INPUT_DIVISORS_BY_NAME = {item.object_name: item for item in INPUT_DIVISOR_RECIPES}
NVFP4_SOURCES = tuple(
    part.source for selected in NVFP4_WEIGHT_RECIPES for part in selected.parts
)


def _word(tensor: torch.Tensor, name: str) -> int:
    if tensor.dtype != torch.float32 or tensor.numel() != 1:
        raise ValueError(f"{name}: divisor must be FP32[1]")
    raw = tensor.detach().contiguous().cpu().view(torch.int32)
    word = int(raw.item()) & 0xFFFFFFFF
    if not valid_positive_fp32_word(word):
        raise ValueError(f"{name}: divisor must be finite and positive")
    return word


def _same_divisor(
    reader: ShardReader,
    sources: Iterable[Nvfp4Source],
    suffix: str,
) -> int:
    items = tuple(sources)
    words = tuple(
        _word(reader.get(source.field(suffix)), source.field(suffix)) for source in items
    )
    if len(set(words)) != 1:
        raise ValueError(f"{items[0].name}: fused {suffix} words do not match")
    return words[0]


def validate_nvfp4_words(reader: ShardReader) -> None:
    for source in NVFP4_SOURCES:
        scales = reader.get(source.field("weight_scale")).view(torch.uint8)
        invalid = ((scales & 0x80) != 0) | (scales == 0x7F)
        if bool(invalid.any()):
            raise ValueError(f"{source.field('weight_scale')}: invalid E4M3FN scale word")
        _word(reader.get(source.field("weight_scale_2")), source.field("weight_scale_2"))
    for group in WEIGHT_DIVISOR_GROUPS:
        _same_divisor(reader, group, "weight_scale_2")


def _select_rows(tensor: torch.Tensor, part: Nvfp4Part) -> torch.Tensor:
    pieces = [
        tensor.narrow(0, row_range.begin, row_range.rows) for row_range in part.rows
    ]
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=0)


def _packed(reader: ShardReader, name: str) -> torch.Tensor:
    getter = getattr(reader, "get_raw", reader.get)
    return getter(name)


def materialize_nvfp4_weight(
    selected: Nvfp4WeightRecipe,
    reader: ShardReader,
) -> tuple[torch.Tensor, torch.Tensor, bytes]:
    packed_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    for part in selected.parts:
        packed = _packed(reader, part.source.field("weight"))
        if packed.dtype != torch.uint8 or packed.dim() != 2:
            raise TypeError(f"{part.source.field('weight')}: packed NVFP4 must be rank-two U8")
        n, k = part.source.shape
        if tuple(packed.shape) != (n, k // 2):
            raise ValueError(
                f"{part.source.field('weight')}: packed shape {tuple(packed.shape)} "
                f"!= {(n, k // 2)}"
            )
        packed_parts.append(_select_rows(packed, part))
        scale_parts.append(
            _select_rows(
                reader.get(part.source.field("weight_scale")).view(torch.uint8),
                part,
            )
        )
    packed = (
        packed_parts[0].contiguous()
        if len(packed_parts) == 1
        else torch.cat(packed_parts, dim=0)
    )
    scales = (
        scale_parts[0].contiguous()
        if len(scale_parts) == 1
        else torch.cat(scale_parts, dim=0)
    )
    word = _same_divisor(reader, selected.divisor_sources, "weight_scale_2")
    # ModelOpt `weight_scale_2` is the multiply global scale. NInfer stores the
    # positive divisor d_w used as e2m1 * e4m3 / d_w.
    source_scale = struct.unpack("<f", struct.pack("<I", word))[0]
    divisor = 1.0 / source_scale
    if not math.isfinite(divisor) or divisor <= 0.0:
        raise ValueError(f"{selected.object_name}: inverted weight divisor is invalid")
    return packed, scales, struct.pack("<f", divisor)


def materialize_input_divisor(_recipe: InputDivisorRecipe) -> torch.Tensor:
    return torch.tensor(PLACEHOLDER_INPUT_SCALE_DIVISOR, dtype=torch.float32)


def validate_recipe() -> None:
    if (
        len(NVFP4_WEIGHT_RECIPES),
        len(INPUT_DIVISOR_RECIPES),
        len(WEIGHT_DIVISOR_GROUPS),
        len(NVFP4_SOURCES),
    ) != (64, 64, 32, 96):  # 32 gate_up + 32 down; 32 fused pairs; 96 source tensors
        raise ValueError("Ornith NVFP4 source recipe is incomplete")
    if tuple(NVFP4_WEIGHTS_BY_NAME) != tuple(spec.name for spec in inventory.NVFP4_TENSOR_SPECS):
        raise ValueError("NVFP4 weight recipe order does not match inventory")
    if tuple(INPUT_DIVISORS_BY_NAME) != tuple(
        spec.name for spec in inventory.INPUT_SCALE_DIVISOR_SPECS
    ):
        raise ValueError("input-divisor recipe order does not match inventory")
    for selected in NVFP4_WEIGHT_RECIPES:
        rows = sum(part.output_rows for part in selected.parts)
        if (rows, selected.parts[0].source.shape[1]) != selected.shape:
            raise ValueError(f"{selected.object_name}: invalid fused row geometry")
        if any(part.source.shape[1] != selected.shape[1] for part in selected.parts):
            raise ValueError(f"{selected.object_name}: incompatible source K")
    bound_weights = tuple(name for site in INPUT_DIVISOR_RECIPES for name in site.weight_names)
    if (
        len(bound_weights) != 64
        or len(set(bound_weights)) != 64
        or set(bound_weights) != set(NVFP4_WEIGHTS_BY_NAME)
    ):
        raise ValueError("input-divisor sites do not cover NVFP4 parents exactly once")


validate_recipe()


__all__ = [
    "INPUT_DIVISOR_RECIPES",
    "INPUT_DIVISORS_BY_NAME",
    "NVFP4_SOURCES",
    "NVFP4_WEIGHT_RECIPES",
    "NVFP4_WEIGHTS_BY_NAME",
    "PLACEHOLDER_INPUT_SCALE_DIVISOR",
    "WEIGHT_DIVISOR_GROUPS",
    "materialize_input_divisor",
    "materialize_nvfp4_weight",
    "validate_nvfp4_words",
    "validate_recipe",
]
