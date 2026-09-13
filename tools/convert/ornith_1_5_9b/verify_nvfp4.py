"""Verify the Ornith-1.5-9B mixed NVFP4-MLP artifact."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from tools.artifact.container import Artifact, ArtifactIdentity, ResourceObject, TensorObject
from tools.artifact.layouts import (
    align_up,
    decode_direct,
    decode_nvfp4_words,
    decode_row_split_codes,
    encoded_size,
)
from tools.artifact.container import object_alignment

from . import inventory_nvfp4 as inventory
from . import recipe_nvfp4 as recipe
from . import verify as base_verify
from .source import OrnithShardReader


class VerificationError(ValueError):
    """The artifact does not satisfy the registered Ornith NVFP4 contract."""


@dataclass(frozen=True, slots=True)
class StructureSummary:
    objects: int
    tensors: int
    resources: int
    nvfp4_weights: int
    input_divisors: int
    payload_bytes: int


def _error(message: str) -> None:
    raise VerificationError(message)


def validate_structure(artifact: Artifact) -> tuple[int, StructureSummary]:
    expected_identity = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID)
    if artifact.identity != expected_identity:
        _error(f"artifact identity is {artifact.identity!r}, expected {expected_identity!r}")
    if len(artifact.objects) != len(inventory.OBJECT_SPECS):
        _error(
            f"artifact has {len(artifact.objects)} objects, expected {len(inventory.OBJECT_SPECS)}"
        )
    cursor = 0
    formats: Counter[str] = Counter()
    layouts: Counter[str] = Counter()
    for position, (actual, expected) in enumerate(zip(artifact.objects, inventory.OBJECT_SPECS)):
        if actual.name != expected.name:
            _error(f"object {position} is {actual.name!r}, expected {expected.name!r}")
        expected_offset = align_up(cursor, object_alignment(actual))
        if actual.offset != expected_offset:
            _error(f"{actual.name}: offset {actual.offset}, expected {expected_offset}")
        if isinstance(expected, inventory.TensorSpec):
            if not isinstance(actual, TensorObject):
                _error(f"{actual.name}: expected tensor descriptor")
            signature = (actual.shape, actual.format, actual.layout)
            registered = (expected.shape, expected.format, expected.layout)
            if signature != registered:
                _error(f"{actual.name}: signature {signature} != {registered}")
            if actual.bytes != encoded_size(actual.layout, actual.format, actual.shape):
                _error(f"{actual.name}: encoded byte count is invalid")
            formats[actual.format] += 1
            layouts[actual.layout] += 1
        else:
            if not isinstance(actual, ResourceObject):
                _error(f"{actual.name}: expected resource descriptor")
            if actual.encoding != expected.encoding:
                _error(f"{actual.name}: resource encoding is invalid")
        cursor = actual.offset + actual.bytes
    if dict(formats) != inventory.FORMAT_COUNTS:
        _error(f"numeric-format counts are {dict(formats)}")
    if dict(layouts) != inventory.LAYOUT_COUNTS:
        _error(f"layout counts are {dict(layouts)}")
    payload_bytes = artifact.file_bytes - artifact.payload_offset
    if cursor != payload_bytes:
        _error(f"payload ends at {cursor}, file contains {payload_bytes} bytes")
    summary = StructureSummary(
        objects=len(artifact.objects),
        tensors=sum(isinstance(obj, TensorObject) for obj in artifact.objects),
        resources=sum(isinstance(obj, ResourceObject) for obj in artifact.objects),
        nvfp4_weights=len(inventory.NVFP4_TENSOR_SPECS),
        input_divisors=len(inventory.INPUT_SCALE_DIVISOR_SPECS),
        payload_bytes=payload_bytes,
    )
    return payload_bytes, summary


def verify_nvfp4_mlp(artifact: Artifact, model_dir: str | Path) -> int:
    count = 0
    with OrnithShardReader(model_dir) as reader:
        for spec in inventory.NVFP4_TENSOR_SPECS[:2]:
            obj = artifact.find(spec.name)
            if not isinstance(obj, TensorObject):
                _error(f"{spec.name} is not a tensor")
            packed, scales, divisor = recipe.materialize_nvfp4_weight(
                recipe.NVFP4_WEIGHTS_BY_NAME[spec.name], reader
            )
            stored_packed, stored_scales, stored_divisor = decode_nvfp4_words(
                artifact.payload(obj), obj.shape
            )
            if (
                not torch.equal(stored_packed, packed)
                or not torch.equal(stored_scales, scales)
                or bytes(stored_divisor.reshape(1).view(torch.uint8).numpy()) != divisor
            ):
                _error(f"{spec.name}: NVFP4 words differ from the source packing")
            count += 1
        for spec in inventory.INPUT_SCALE_DIVISOR_SPECS[:2]:
            obj = artifact.find(spec.name)
            stored = decode_direct(artifact.payload(obj), obj.format, obj.shape)
            if float(stored.reshape(()).item()) != recipe.PLACEHOLDER_INPUT_SCALE_DIVISOR:
                _error(f"{spec.name}: input_scale_divisor is not the W4A16 placeholder")
            count += 1
    return count


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args(argv)

    artifact = Artifact(args.artifact)
    _, structure = validate_structure(artifact)
    nvfp4_count = verify_nvfp4_mlp(artifact, args.model)
    direct_count = base_verify.verify_direct_tensors(artifact, args.model)
    quant_probe_count, quant_rows = base_verify.verify_quantized_tensors(artifact)
    print(
        f"structure: {structure.objects} objects ({structure.tensors} tensors, "
        f"{structure.resources} resources), nvfp4={structure.nvfp4_weights}, "
        f"divisors={structure.input_divisors}, {structure.payload_bytes} payload bytes",
        flush=True,
    )
    print(
        f"payloads: {nvfp4_count} NVFP4 probes, {direct_count} direct probes, "
        f"{quant_probe_count} quantized probes, {quant_rows} quantized rows",
        flush=True,
    )
    print("verification complete", flush=True)


if __name__ == "__main__":
    main()
