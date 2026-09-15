"""Persistent-object contract for the Ornith-1.5-9B mixed NVFP4-MLP artifact."""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import TensorSpec
from tools.convert.qwen3_5_9b import inventory as base


MODEL_ID = "ornith-1.5-9b"
WEIGHTS_ID = "nvfp4"
TARGET_KEY = "ornith_1_5_9b"

BF16 = base.BF16
FP32 = base.FP32
I32 = base.I32
Q4 = base.Q4
Q5 = base.Q5
Q6 = base.Q6
W8 = base.W8
NVFP4 = "NVFP4"
FP8 = "FP8_E4M3FN_ROW_BF16S"

CONTIGUOUS_LAYOUT = base.CONTIGUOUS_LAYOUT
ROW_SPLIT_LAYOUT = base.ROW_SPLIT_LAYOUT
BLOCK_SCALE_LAYOUT = "blockscale-k16-m128x4-v1"
ROW_SCALE_LAYOUT = "row-scale-v1"

RESOURCE_SPECS = base.RESOURCE_SPECS
ResourceSpec = base.ResourceSpec
StoredObjectSpec = base.StoredObjectSpec

FULL_ATTENTION_LAYERS = base.FULL_ATTENTION_LAYERS
GDN_LAYERS = base.GDN_LAYERS


def _tensor(name: str, shape: tuple[int, ...], numeric_format: str) -> TensorSpec:
    if numeric_format in (BF16, FP32, I32):
        layout = CONTIGUOUS_LAYOUT
    elif numeric_format == NVFP4:
        layout = BLOCK_SCALE_LAYOUT
    elif numeric_format == FP8:
        layout = ROW_SCALE_LAYOUT
    else:
        layout = ROW_SPLIT_LAYOUT
    return TensorSpec(name, shape, numeric_format, layout)


def _input_scale_after(spec: TensorSpec) -> str | None:
    if spec.format != NVFP4:
        return None
    prefix, suffix = spec.name.rsplit("/", 1)
    if suffix == "gate_up" and prefix.endswith("/mlp"):
        return prefix + "/gate_up_projection/input_scale_divisor"
    if suffix == "down" and prefix.endswith("/mlp"):
        return prefix + "/down_projection/input_scale_divisor"
    if suffix == "output_head" and prefix == "text":
        return prefix + "/output_head_projection/input_scale_divisor"
    return None


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = []
    for original in base.TEXT_CORE_TENSOR_SPECS:
        if original.name.endswith("/mlp/gate_up"):
            spec = _tensor(original.name, original.shape, NVFP4)
            specs.append(spec)
            specs.append(_tensor(_input_scale_after(spec), (), FP32))
            continue
        if original.name.endswith("/mlp/down"):
            spec = _tensor(original.name, original.shape, NVFP4)
            specs.append(spec)
            specs.append(_tensor(_input_scale_after(spec), (), FP32))
            continue
        if original.name == "text/output_head":
            spec = _tensor(original.name, original.shape, NVFP4)
            specs.append(spec)
            specs.append(_tensor(_input_scale_after(spec), (), FP32))
            continue
        specs.append(original)
        # Native FP8 large-T parent alongside the groupwise small-T parents.
        # Unlike the 27B NVFP4 contract (which replaces the split parents),
        # the hybrid T-policy needs both representations resident.
        if original.name.endswith("/gdn/value_z"):
            prefix = original.name.removesuffix("/value_z")
            specs.append(_tensor(prefix + "/query_key_value_z", (12288, 4096), FP8))
    return tuple(specs)


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
DRAFT_HEAD_TENSOR_SPECS = base.DRAFT_HEAD_TENSOR_SPECS
MTP_TENSOR_SPECS = base.MTP_TENSOR_SPECS
VISION_TENSOR_SPECS = base.VISION_TENSOR_SPECS

TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_NAMES = (BF16, FP32, I32, Q4, Q5, Q6, W8, NVFP4, FP8)
LAYOUT_NAMES = (CONTIGUOUS_LAYOUT, ROW_SPLIT_LAYOUT, BLOCK_SCALE_LAYOUT, ROW_SCALE_LAYOUT)
FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

LOGICAL_ROW_VIEW_SPECS = base.LOGICAL_ROW_VIEW_SPECS
ALIAS_SPECS = base.ALIAS_SPECS

NVFP4_TENSOR_SPECS = tuple(spec for spec in TENSOR_SPECS if spec.format == NVFP4)
FP8_GDN_TENSOR_SPECS = tuple(spec for spec in TENSOR_SPECS if spec.format == FP8)
INPUT_SCALE_DIVISOR_SPECS = tuple(
    spec
    for spec in TENSOR_SPECS
    if spec.format == FP32 and spec.name.endswith("/input_scale_divisor")
)

# Native FP8 GDN parent count. Kept symbolic so the head-branch delta (+1
# NVFP4 object) cherry-picks against these lines with a mechanical offset.
N_FP8_GDN_PARENTS = 24


def validate_inventory() -> None:
    names = tuple(spec.name for spec in OBJECT_SPECS)
    if len(names) != len(set(names)):
        raise ValueError("NVFP4 inventory contains duplicate object names")
    if (
        len(TEXT_CORE_TENSOR_SPECS),
        len(DRAFT_HEAD_TENSOR_SPECS),
        len(MTP_TENSOR_SPECS),
        len(VISION_TENSOR_SPECS),
        len(TENSOR_SPECS),
        len(OBJECT_SPECS),
        len(NVFP4_TENSOR_SPECS),
        len(FP8_GDN_TENSOR_SPECS),
        len(INPUT_SCALE_DIVISOR_SPECS),
    ) != (
        len(base.TEXT_CORE_TENSOR_SPECS) + 65 + N_FP8_GDN_PARENTS,
        2,
        12,
        333,
        len(base.TENSOR_SPECS) + 65 + N_FP8_GDN_PARENTS,
        len(base.OBJECT_SPECS) + 65 + N_FP8_GDN_PARENTS,
        65,
        N_FP8_GDN_PARENTS,
        65,
    ):
        raise ValueError("registered Ornith NVFP4 inventory is incomplete")
    expected_formats = dict(base.FORMAT_COUNTS)
    expected_formats[Q4] = expected_formats[Q4] - 32
    expected_formats[Q5] = expected_formats[Q5] - 32
    expected_formats[Q6] = expected_formats[Q6] - 1
    expected_formats[FP32] = expected_formats[FP32] + 65
    expected_formats[NVFP4] = 65
    expected_formats[FP8] = N_FP8_GDN_PARENTS
    if FORMAT_COUNTS != expected_formats:
        raise ValueError(f"unexpected NVFP4 format allocation: {FORMAT_COUNTS}")
    expected_layouts = dict(base.LAYOUT_COUNTS)
    expected_layouts[ROW_SPLIT_LAYOUT] = expected_layouts[ROW_SPLIT_LAYOUT] - 65
    expected_layouts[CONTIGUOUS_LAYOUT] = expected_layouts[CONTIGUOUS_LAYOUT] + 65
    expected_layouts[BLOCK_SCALE_LAYOUT] = 65
    expected_layouts[ROW_SCALE_LAYOUT] = N_FP8_GDN_PARENTS
    if LAYOUT_COUNTS != expected_layouts:
        raise ValueError(f"unexpected NVFP4 layout allocation: {LAYOUT_COUNTS}")
    if any(spec.shape != (24576, 4096) for spec in NVFP4_TENSOR_SPECS if spec.name.endswith("/gate_up")):
        raise ValueError("MLP gate_up NVFP4 shape is invalid")
    if any(spec.shape != (4096, 12288) for spec in NVFP4_TENSOR_SPECS if spec.name.endswith("/down")):
        raise ValueError("MLP down NVFP4 shape is invalid")
    if any(
        spec.shape != (12288, 4096)
        for spec in FP8_GDN_TENSOR_SPECS
        if spec.name.endswith("/query_key_value_z")
    ):
        raise ValueError("GDN query_key_value_z FP8 shape is invalid")
    if len(FP8_GDN_TENSOR_SPECS) != len(GDN_LAYERS):
        raise ValueError("FP8 GDN parents must cover every GDN layer exactly once")
    if any(spec.shape != (248320, 4096) for spec in NVFP4_TENSOR_SPECS if spec.name == "text/output_head"):
        raise ValueError("output_head NVFP4 shape is invalid")


validate_inventory()


__all__ = [
    "ALIAS_SPECS",
    "BF16",
    "BLOCK_SCALE_LAYOUT",
    "CONTIGUOUS_LAYOUT",
    "DRAFT_HEAD_TENSOR_SPECS",
    "FORMAT_COUNTS",
    "FORMAT_NAMES",
    "FP8",
    "FP8_GDN_TENSOR_SPECS",
    "FP32",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "I32",
    "INPUT_SCALE_DIVISOR_SPECS",
    "LAYOUT_COUNTS",
    "LAYOUT_NAMES",
    "LOGICAL_ROW_VIEW_SPECS",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "N_FP8_GDN_PARENTS",
    "NVFP4",
    "NVFP4_TENSOR_SPECS",
    "OBJECT_SPECS",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_SPECS",
    "ROW_SCALE_LAYOUT",
    "ResourceSpec",
    "StoredObjectSpec",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "TensorSpec",
    "VISION_TENSOR_SPECS",
    "W8",
    "WEIGHTS_ID",
    "validate_inventory",
]
