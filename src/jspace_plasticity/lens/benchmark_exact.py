"""Bounded memory/throughput benchmark for Anthropic's exact Jacobian fit.

This deliberately computes only a few coordinate blocks.  It uses the same
one-hot cotangents, retained graph, source-position averaging, and CPU row copy
as ``jlens.fitting.jacobian_for_prompt`` and extrapolates the cost of covering
all ``d_model`` output coordinates.  No lens produced here is scientifically
usable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import jlens
import torch
import torch.distributed as dist
from jlens.fitting import valid_position_mask
from jlens.hooks import ActivationRecorder
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig

try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Replicate
except ImportError:  # pragma: no cover - pinned torch provides DTensor
    DTensor = None  # type: ignore[assignment,misc]
    Replicate = None  # type: ignore[assignment,misc]

GIB = 1024**3
SCHEMA_VERSION = 1


def evenly_spaced_source_layers(target_layer: int, count: int) -> list[int]:
    """Return ``count`` unique layers spanning ``0..target_layer-1``."""
    if target_layer < 1:
        raise ValueError("target_layer must leave at least one source layer")
    if not 1 <= count <= target_layer:
        raise ValueError("source-layer count must be in [1, target_layer]")
    if count == 1:
        return [0]
    layers = [round(i * (target_layer - 1) / (count - 1)) for i in range(count)]
    if len(set(layers)) != count:
        raise RuntimeError("evenly spaced layer selection produced duplicates")
    return layers


def parse_layers(value: str) -> list[int]:
    layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not layers:
        raise argparse.ArgumentTypeError("--layers must contain an integer")
    return layers


def load_prompt(path: Path, index: int) -> tuple[str, str]:
    if index < 0:
        raise ValueError("prompt index must be non-negative")
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index != index:
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"row {index} has no non-empty text field")
            return text, hashlib.sha256(text.encode()).hexdigest()
    raise ValueError(f"prompt index {index} is outside {path}")


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _distributed_context(
    expected_world_size: int,
) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != expected_world_size:
        raise RuntimeError(
            f"WORLD_SIZE={world_size}, expected {expected_world_size}; "
            "launch with the rendered torchrun command"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("the exact-lens benchmark requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, local_rank, world_size, device


def _is_dtensor(value: torch.Tensor) -> bool:
    return DTensor is not None and isinstance(value, DTensor)


def _replicated_local(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if not _is_dtensor(value):
        return value
    assert DTensor is not None and Replicate is not None
    if not all(isinstance(placement, Replicate) for placement in value.placements):
        raise RuntimeError(
            f"{label} must be replicated under TP, found {value.placements}"
        )
    return value.to_local()


def _replicated_like(local: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not _is_dtensor(reference):
        return local
    assert DTensor is not None
    return DTensor.from_local(
        local,
        device_mesh=reference.device_mesh,
        placements=reference.placements,
        run_check=False,
        shape=reference.shape,
        stride=reference.stride(),
    )


def _synchronize() -> None:
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        torch.cuda.synchronize()


def _gather_rank_metrics(
    local: dict[str, Any], world_size: int
) -> list[dict[str, Any]]:
    if world_size == 1:
        return [local]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return [item for item in gathered if item is not None]


def _local_parameter_bytes(model: torch.nn.Module) -> int:
    total = 0
    for parameter in model.parameters():
        local = parameter.to_local() if _is_dtensor(parameter) else parameter
        total += local.numel() * local.element_size()
    return total


def compare_sample_artifacts(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    for key in ("model", "revision", "prompt_sha256", "source_layers", "target_layer"):
        if candidate["metadata"][key] != reference["metadata"][key]:
            raise ValueError(f"sample artifacts disagree on {key}")
    layer_metrics: dict[str, dict[str, float]] = {}
    candidate_rows = candidate["rows"]
    reference_rows = reference["rows"]
    common_directions = min(
        next(iter(candidate_rows.values())).shape[0],
        next(iter(reference_rows.values())).shape[0],
    )
    flattened_candidate: list[torch.Tensor] = []
    flattened_reference: list[torch.Tensor] = []
    for layer in candidate["metadata"]["source_layers"]:
        left = candidate_rows[layer][:common_directions].float().flatten()
        right = reference_rows[layer][:common_directions].float().flatten()
        difference = left - right
        relative_l2 = float(difference.norm() / right.norm().clamp_min(1e-12))
        cosine = float(torch.nn.functional.cosine_similarity(left, right, dim=0))
        layer_metrics[str(layer)] = {
            "cosine": cosine,
            "relative_l2": relative_l2,
            "mean_abs_delta": float(difference.abs().mean()),
            "max_abs_delta": float(difference.abs().max()),
        }
        flattened_candidate.append(left)
        flattened_reference.append(right)
    left_all = torch.cat(flattened_candidate)
    right_all = torch.cat(flattened_reference)
    delta_all = left_all - right_all

    candidate_logits = candidate["final_logits"].float()
    reference_logits = reference["final_logits"].float()
    logit_delta = candidate_logits - reference_logits
    reference_log_probs = reference_logits.log_softmax(dim=-1)
    candidate_log_probs = candidate_logits.log_softmax(dim=-1)
    reference_probs = reference_log_probs.exp()
    kl = float((reference_probs * (reference_log_probs - candidate_log_probs)).sum())
    return {
        "common_directions": common_directions,
        "global_cosine": float(
            torch.nn.functional.cosine_similarity(left_all, right_all, dim=0)
        ),
        "global_relative_l2": float(
            delta_all.norm() / right_all.norm().clamp_min(1e-12)
        ),
        "layer_metrics": layer_metrics,
        "forward": {
            "argmax_agreement": bool(
                candidate_logits.argmax() == reference_logits.argmax()
            ),
            "mean_abs_logit_delta": float(logit_delta.abs().mean()),
            "max_abs_logit_delta": float(logit_delta.abs().max()),
            "kl_reference_to_candidate": kl,
        },
    }


def _load_model(
    model_id: str,
    revision: str,
    *,
    tensor_parallel: bool,
    world_size: int,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    if getattr(config, "vision_config", None) is not None:
        raise ValueError("the exact-lens benchmark expects a text-only causal LM")
    kwargs: dict[str, Any] = {
        "config": config,
        "revision": revision,
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if tensor_parallel:
        if world_size <= 1:
            raise ValueError("tensor parallel loading requires world_size > 1")
        if getattr(config, "base_model_tp_plan", None) is None:
            raise ValueError(f"{model_id} has no native Transformers TP plan")
        # Transformers 5.15 routes native TP through DistributedConfig. Passing
        # tp_plan directly is not a supported from_pretrained keyword in this
        # pinned runtime and leaks into Qwen3ForCausalLM.__init__.
        kwargs["distributed_config"] = DistributedConfig(tp_size=world_size)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not tensor_parallel:
        model.to(device)
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    return model, tokenizer, config


def _benchmark_coordinate_blocks(
    lens_model: Any,
    prompt: str,
    *,
    source_layers: list[int],
    target_layer: int,
    dim_batch: int,
    coordinate_blocks: int,
    max_seq_len: int,
    skip_first: int,
) -> tuple[dict[str, Any], dict[int, torch.Tensor], torch.Tensor]:
    input_ids = lens_model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    valid_positions_cpu = position_mask.nonzero(as_tuple=True)[0]
    if not len(valid_positions_cpu):
        raise ValueError("prompt has no valid positions after skip_first")
    if dim_batch * coordinate_blocks > lens_model.d_model:
        raise ValueError("sampled coordinate count exceeds d_model")

    rows_by_layer: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in source_layers
    }
    backward_seconds: list[float] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()

    with (
        ActivationRecorder(
            lens_model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        replicated_ids = input_ids.expand(dim_batch, -1)
        _synchronize()
        forward_start = time.perf_counter()
        lens_model.forward(replicated_ids)
        _synchronize()
        forward_seconds = time.perf_counter() - forward_start

        target_activation = recorder.activations[target_layer]
        target_local = _replicated_local(target_activation, label="target activation")
        source_activations = [recorder.activations[layer] for layer in source_layers]
        valid_positions = valid_positions_cpu.to(target_local.device)
        batch_indices = torch.arange(dim_batch, device=target_local.device)

        with torch.no_grad():
            final_logits = lens_model.unembed(target_activation.detach()[:, -1, :])
            final_logits_local = _replicated_local(
                final_logits, label="unembedding output"
            )[0].float().cpu()

        for block_index in range(coordinate_blocks):
            dimension_start = block_index * dim_batch
            cotangent_local = torch.zeros_like(target_local)
            cotangent_local[
                batch_indices[:, None],
                valid_positions[None, :],
                dimension_start + batch_indices[:, None],
            ] = 1.0
            cotangent = _replicated_like(cotangent_local, target_activation)
            _synchronize()
            backward_start = time.perf_counter()
            gradients = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=block_index < coordinate_blocks - 1,
            )
            for layer, gradient in zip(source_layers, gradients, strict=True):
                gradient_local = _replicated_local(
                    gradient, label=f"source gradient L{layer}"
                )
                positions = valid_positions_cpu.to(gradient_local.device)
                rows = gradient_local[:, positions, :].float().mean(dim=1).cpu()
                rows_by_layer[layer].append(rows)
            _synchronize()
            backward_seconds.append(time.perf_counter() - backward_start)
            del gradients, cotangent, cotangent_local

    sampled_rows = {
        layer: torch.cat(blocks, dim=0) for layer, blocks in rows_by_layer.items()
    }
    median_backward = statistics.median(backward_seconds)
    batches_per_prompt = math.ceil(lens_model.d_model / dim_batch)
    projected_prompt_seconds = forward_seconds + batches_per_prompt * median_backward
    metrics = {
        "seq_len": seq_len,
        "valid_positions": len(valid_positions_cpu),
        "forward_seconds": forward_seconds,
        "backward_block_seconds": backward_seconds,
        "median_backward_block_seconds": median_backward,
        "coordinate_blocks_measured": coordinate_blocks,
        "coordinate_directions_measured": dim_batch * coordinate_blocks,
        "coordinate_batches_per_exact_prompt": batches_per_prompt,
        "projected_exact_prompt_seconds": projected_prompt_seconds,
        "sampled_directions_per_second": dim_batch / median_backward,
        "baseline_allocated_gib": baseline_allocated / GIB,
        "baseline_reserved_gib": baseline_reserved / GIB,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
    }
    return metrics, sampled_rows, final_logits_local


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompts-jsonl", required=True, type=Path)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument(
        "--tensor-parallel", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--layers", type=parse_layers)
    parser.add_argument("--source-layer-count", type=int, default=25)
    parser.add_argument("--target-layer", type=int)
    parser.add_argument("--dim-batch", type=int, required=True)
    parser.add_argument("--coordinate-blocks", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--reference-samples", type=Path)
    parser.add_argument("--min-row-cosine", type=float, default=0.999)
    parser.add_argument("--max-row-relative-l2", type=float, default=0.05)
    args = parser.parse_args()
    if args.expected_world_size < 1:
        parser.error("--expected-world-size must be positive")
    if args.tensor_parallel != (args.expected_world_size > 1):
        parser.error("TP must be enabled exactly when expected world size exceeds one")
    if args.dim_batch < 1 or args.coordinate_blocks < 1:
        parser.error("dim batch and coordinate blocks must be positive")

    rank, local_rank, world_size, device = _distributed_context(
        args.expected_world_size
    )
    started = time.perf_counter()
    try:
        prompt, prompt_sha256 = load_prompt(args.prompts_jsonl, args.prompt_index)
        torch.cuda.reset_peak_memory_stats()
        model, tokenizer, config = _load_model(
            args.model,
            args.revision,
            tensor_parallel=args.tensor_parallel,
            world_size=world_size,
            device=device,
        )
        _synchronize()
        load_peak_allocated = torch.cuda.max_memory_allocated(device) / GIB
        load_peak_reserved = torch.cuda.max_memory_reserved(device) / GIB
        lens_model = jlens.from_hf(model, tokenizer, compile=False)
        target_layer = (
            lens_model.n_layers - 1
            if args.target_layer is None
            else args.target_layer
        )
        source_layers = (
            evenly_spaced_source_layers(target_layer, args.source_layer_count)
            if args.layers is None
            else args.layers
        )
        if any(layer < 0 or layer >= target_layer for layer in source_layers):
            raise ValueError("every source layer must be in [0, target_layer)")

        local_metrics, sampled_rows, final_logits = _benchmark_coordinate_blocks(
            lens_model,
            prompt,
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=args.dim_batch,
            coordinate_blocks=args.coordinate_blocks,
            max_seq_len=args.max_seq_len,
            skip_first=args.skip_first,
        )
        local_metrics.update(
            {
                "rank": rank,
                "local_rank": local_rank,
                "load_peak_allocated_gib": load_peak_allocated,
                "load_peak_reserved_gib": load_peak_reserved,
                "local_parameter_storage_gib": _local_parameter_bytes(model) / GIB,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_gib": torch.cuda.get_device_properties(
                    device
                ).total_memory
                / GIB,
            }
        )
        rank_metrics = _gather_rank_metrics(local_metrics, world_size)

        if rank == 0:
            metadata = {
                "model": args.model,
                "revision": args.revision,
                "prompt_sha256": prompt_sha256,
                "prompt_index": args.prompt_index,
                "source_layers": source_layers,
                "target_layer": target_layer,
                "d_model": lens_model.d_model,
                "n_layers": lens_model.n_layers,
                "dim_batch": args.dim_batch,
                "coordinate_blocks": args.coordinate_blocks,
                "world_size": world_size,
                "tensor_parallel": args.tensor_parallel,
            }
            samples = {
                "metadata": metadata,
                "rows": sampled_rows,
                "final_logits": final_logits,
            }
            samples_path = args.output_dir / "sample_rows.pt"
            _atomic_torch_save(samples, samples_path)

            slowest_prompt_projection = max(
                item["projected_exact_prompt_seconds"] for item in rank_metrics
            )
            peak_reserved = max(item["peak_reserved_gib"] for item in rank_metrics)
            total_memory = min(
                item["gpu_total_memory_gib"] for item in rank_metrics
            )
            result: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "scientifically_usable_lens": False,
                "method": "official-exact-coordinate-jacobian-bounded-benchmark",
                "metadata": metadata,
                "rank_metrics": rank_metrics,
                "summary": {
                    "projected_exact_prompt_seconds": slowest_prompt_projection,
                    "projected_gpu_seconds_per_prompt": (
                        slowest_prompt_projection * world_size
                    ),
                    "peak_reserved_gib": peak_reserved,
                    "minimum_memory_headroom_gib": total_memory - peak_reserved,
                    "full_25_layer_fp32_lens_storage_gib": (
                        len(source_layers) * lens_model.d_model**2 * 4 / GIB
                    ),
                },
                "samples_path": str(samples_path),
                "elapsed_seconds": time.perf_counter() - started,
            }
            if args.reference_samples is not None:
                reference = torch.load(
                    args.reference_samples, map_location="cpu", weights_only=True
                )
                equivalence = compare_sample_artifacts(samples, reference)
                equivalence["passed"] = bool(
                    equivalence["global_cosine"] >= args.min_row_cosine
                    and equivalence["global_relative_l2"]
                    <= args.max_row_relative_l2
                    and equivalence["forward"]["argmax_agreement"]
                )
                equivalence["thresholds"] = {
                    "min_row_cosine": args.min_row_cosine,
                    "max_row_relative_l2": args.max_row_relative_l2,
                    "require_forward_argmax_agreement": True,
                }
                result["reference_equivalence"] = equivalence
            _atomic_json(result, args.output_dir / "result.json")
            print(json.dumps(result, indent=2, sort_keys=True))
            if (
                args.reference_samples is not None
                and not result["reference_equivalence"]["passed"]
            ):
                raise RuntimeError("TP exact-Jacobian equivalence gate failed")
        if dist.is_initialized():
            dist.barrier()
    finally:
        gc.collect()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
