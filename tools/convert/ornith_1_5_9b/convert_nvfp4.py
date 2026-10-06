"""Build the Ornith-1.5-9B mixed NVFP4-MLP artifact from the official checkpoint.

Canonical invocation::

    python -m tools.convert.ornith_1_5_9b.convert_nvfp4 \
      --model /path/to/Ornith-1.5-9B-NVFP4 \
      --out models/ornith_1_5_9b_nvfp4.ninfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactObject, ArtifactWriter
from tools.artifact.layouts import decode_nvfp4_words, encode_direct, encode_nvfp4
from tools.convert.common.quantize import pick_device
from tools.convert.qwen3_6.common import conversion as family_conversion

from . import convert as base_convert
from . import draft_head
from . import inventory_nvfp4 as inventory
from . import recipe as groupwise_recipe
from . import recipe_nvfp4 as recipe
from .source import OrnithShardReader


RECIPE_ID = "ornith_1_5_9b_nvfp4-v1"
OUTPUT_BASENAME = "ornith_1_5_9b_nvfp4.ninfer"

ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: object
    resources: tuple[ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def preflight_inventory() -> None:
    inventory.validate_inventory()
    recipe.validate_recipe()


def build_object_plan(resources: Mapping[str, bytes]) -> ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def preflight_conversion(model_dir: str | Path) -> ConversionPreflight:
    model = Path(model_dir)
    config_summary = base_convert.validate_config(base_convert._load_config(model))
    preflight_inventory()
    source = groupwise_recipe.preflight_sources(model)
    with OrnithShardReader(model) as reader:
        recipe.validate_nvfp4_words(reader)
    resources = base_convert.load_resources(model)
    object_plan = build_object_plan({resource.name: resource.data for resource in resources})
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model)
    return ConversionPreflight(model, config_summary, source, resources, draft, object_plan)


def _encode_nvfp4_weight(spec: inventory.TensorSpec, reader: OrnithShardReader) -> bytes:
    selected = recipe.NVFP4_WEIGHTS_BY_NAME[spec.name]
    packed, scales, divisor = recipe.materialize_nvfp4_weight(selected, reader)
    payload = encode_nvfp4(packed, scales, divisor, spec.shape)
    decoded_packed, decoded_scales, decoded_divisor = decode_nvfp4_words(payload, spec.shape)
    if (
        not torch.equal(decoded_packed, packed)
        or not torch.equal(decoded_scales, scales)
        or bytes(decoded_divisor.reshape(1).view(torch.uint8).numpy()) != divisor
    ):
        raise RuntimeError(f"{spec.name}: NVFP4 layout word verification failed")
    return payload


def build_conversion_report(
    *,
    model_dir: str | Path,
    out_path: str | Path,
    arguments: Mapping[str, object],
    config_summary: Mapping[str, object],
    source_preflight: object,
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    ranking_path: str | Path,
    revision: str | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=model_dir,
        out_path=out_path,
        arguments=arguments,
        config_summary=config_summary,
        source_preflight=source_preflight,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking_path,
        revision=revision,
        environment_summary=environment,
    )


def convert(
    model_dir: str | Path, out_path: str | Path, *, device: str | torch.device = "cuda"
) -> Path:
    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model)
    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"nvfp4_mlp={len(inventory.NVFP4_TENSOR_SPECS)}, device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    with OrnithShardReader(model) as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("writer object plan differs from completed preflight")
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                elif spec.format == inventory.NVFP4:
                    payload = _encode_nvfp4_weight(spec, reader)
                elif spec.name in recipe.INPUT_DIVISORS_BY_NAME:
                    payload = encode_direct(
                        recipe.materialize_input_divisor(recipe.INPUT_DIVISORS_BY_NAME[spec.name]),
                        inventory.FP32,
                    )
                else:
                    tensor = base_convert.materialize_tensor(spec, reader, preflight.draft)
                    payload = base_convert.encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}", flush=True)
    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    report = build_conversion_report(
        model_dir=model,
        out_path=output,
        arguments={"model": str(model_dir), "out": str(out_path), "device": requested_device},
        config_summary=preflight.config_summary,
        source_preflight=preflight.source,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        ranking_path=ranking,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}", flush=True)
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    convert(arguments.model, arguments.out, device=arguments.device)


if __name__ == "__main__":
    main()
