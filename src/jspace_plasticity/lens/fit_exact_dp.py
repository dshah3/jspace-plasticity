"""Resumable exact Jacobian-lens fitting with independent GPU workers.

Every torchrun rank owns a complete model on one GPU and processes a disjoint,
deterministic prompt shard.  The workers deliberately never initialize a torch
distributed process group: exact Jacobians are prompt-separable, and avoiding
collectives prevents a failed scientific gate from stranding GPUs in a barrier.

Rank-local fp32 running sums are checkpointed by Anthropic's ``jlens.fit``.
After every worker has written a stage receipt, rank zero merges those fp32 sums
in rank order and emits a stage-specific lens plus provenance.  Increasing
``--num-prompts`` with the same output directory resumes the same fit.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jlens
import torch
from jlens.hooks import ActivationRecorder
from transformers import AutoConfig, AutoTokenizer

from jspace_plasticity.lens.benchmark_exact import (
    compare_sample_artifacts,
    evenly_spaced_source_layers,
    parse_layers,
)
from jspace_plasticity.modeling import auto_model_class_for_config

SCHEMA_VERSION = 1
GIB = 1024**3
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndependentRank:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_lens_save(lens: jlens.JacobianLens, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    lens.save(str(temporary), dtype=torch.float16)
    temporary.replace(path)


def independent_rank(expected_world_size: int) -> IndependentRank:
    """Resolve torchrun rank variables without creating a process group."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != expected_world_size:
        raise RuntimeError(
            f"WORLD_SIZE={world_size}, expected {expected_world_size}; "
            "launch with the rendered torchrun command"
        )
    if not 0 <= rank < world_size:
        raise RuntimeError(f"RANK={rank} is outside WORLD_SIZE={world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("exact lens fitting requires CUDA")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} GPUs exist"
        )
    torch.cuda.set_device(local_rank)
    return IndependentRank(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def assigned_prompt_indices(
    *, offset: int, total: int, rank: int, world_size: int
) -> list[int]:
    if offset < 0 or total < 1:
        raise ValueError("prompt offset must be non-negative and total positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be inside world size")
    return list(range(offset + rank, offset + total, world_size))


def load_assigned_prompts(
    path: Path, *, offset: int, total: int, rank: int, world_size: int
) -> tuple[list[str], list[int]]:
    indices = assigned_prompt_indices(
        offset=offset, total=total, rank=rank, world_size=world_size
    )
    wanted = set(indices)
    prompts: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index not in wanted:
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"row {index} in {path} has no non-empty text")
            prompts[index] = text
            if len(prompts) == len(indices):
                break
    missing = sorted(wanted - prompts.keys())
    if missing:
        raise ValueError(f"frozen corpus is missing assigned rows: {missing[:8]}")
    return [prompts[index] for index in indices], indices


def model_load_kwargs(
    config: Any, *, revision: str, device: torch.device
) -> dict[str, Any]:
    return {
        "config": config,
        "revision": revision,
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
        # Loading directly into the rank-local GPU avoids eight simultaneous
        # full 32B CPU copies before model.to(device).
        "device_map": str(device),
    }


def _load_model(
    model_id: str, revision: str, device: torch.device
) -> tuple[torch.nn.Module, Any]:
    local_checkpoint = Path(model_id).is_dir()
    revision_kwargs = {} if local_checkpoint else {"revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **revision_kwargs)
    config = AutoConfig.from_pretrained(model_id, **revision_kwargs)
    kwargs = model_load_kwargs(config, revision=revision, device=device)
    if local_checkpoint:
        kwargs.pop("revision")
    auto_model = auto_model_class_for_config(config)
    model = auto_model.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    devices = {
        parameter.device
        for parameter in model.parameters()
        if parameter.device.type != "meta"
    }
    if devices != {device}:
        found = sorted(map(str, devices))
        raise RuntimeError(
            f"rank-local model must be entirely on {device}, found {found}"
        )
    return model, tokenizer


def validate_model_manifest(
    path: Path,
    *,
    expected_sha256: str,
    model_dir: Path,
) -> dict[str, Any]:
    """Bind a local recovered checkpoint without rereading every 5 GB shard.

    The manifest was hash-bound into the completed SFT result.  Every rank
    verifies that receipt plus each listed relative path and byte size; loading
    the safetensors then provides the structural integrity check.
    """

    observed = _sha256(path)
    if observed != expected_sha256:
        raise ValueError("recovered checkpoint manifest SHA-256 mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if Path(manifest.get("path", "")) != model_dir:
        raise ValueError("recovered checkpoint manifest is bound to another path")
    files = manifest.get("files", [])
    if not files:
        raise ValueError("recovered checkpoint manifest contains no files")
    total = 0
    for receipt in files:
        relative = Path(receipt["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checkpoint manifest contains an unsafe relative path")
        artifact = model_dir / relative
        if not artifact.is_file() or artifact.stat().st_size != int(receipt["bytes"]):
            raise ValueError(f"checkpoint artifact size mismatch: {relative}")
        total += artifact.stat().st_size
    if total != int(manifest.get("total_bytes", -1)):
        raise ValueError("checkpoint manifest total byte count mismatch")
    return {
        "path": str(path),
        "sha256": observed,
        "model_dir": str(model_dir),
        "files": len(files),
        "total_bytes": total,
    }


def _wait_for_path(path: Path, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(1)


def ensure_immutable_config(
    path: Path,
    payload: dict[str, Any],
    *,
    rank: int,
    timeout_seconds: float = 300,
) -> str:
    """Create once on rank zero and require exact compatibility on resume."""
    if rank == 0:
        if not path.exists():
            _atomic_json(payload, path)
    else:
        _wait_for_path(path, timeout_seconds=timeout_seconds)
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != payload:
        raise ValueError(
            f"fit configuration at {path} is incompatible with this invocation"
        )
    return _canonical_sha256(payload)


def validate_checkpoint_progress(path: Path, *, assigned_prompts: int) -> None:
    if not path.exists():
        return
    state = torch.load(path, map_location="cpu", weights_only=True)
    next_idx = int(state["next_idx"])
    n_done = int(state["n_done"])
    if not 0 <= n_done <= next_idx <= assigned_prompts:
        raise ValueError(
            f"checkpoint {path} has n_done={n_done}, next_idx={next_idx}, "
            f"but this stage assigns only {assigned_prompts} prompts"
        )


def prepare_rank_checkpoint(output_dir: Path, *, rank: int) -> Path:
    """Create this independent worker's shard directory and return its checkpoint.

    ``jlens.fit`` writes its checkpoint directly with ``torch.save`` and therefore
    requires the parent directory to exist before the first completed prompt.
    Every rank creates only its own directory; ``parents=True`` safely handles
    concurrent creation of the shared ``shards`` parent.
    """
    rank_dir = output_dir / "shards" / f"rank-{rank:03d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    return rank_dir / "checkpoint.pt"


def merge_fp32_checkpoints(
    checkpoints: list[Path],
    *,
    expected_counts: list[int],
    source_layers: list[int],
    target_layer: int,
    skip_first: int,
) -> jlens.JacobianLens:
    """Merge rank sums in deterministic order without fp16 shard roundtrips."""
    if len(checkpoints) != len(expected_counts) or not checkpoints:
        raise ValueError(
            "checkpoints and expected counts must be non-empty and aligned"
        )
    merged_sum: dict[int, torch.Tensor] | None = None
    n_total = 0
    d_model: int | None = None
    for path, expected in zip(checkpoints, expected_counts, strict=True):
        state = torch.load(path, map_location="cpu", weights_only=True)
        for key, value in (
            ("source_layers", source_layers),
            ("target_layer", target_layer),
            ("skip_first", skip_first),
        ):
            if state.get(key) != value:
                raise ValueError(f"checkpoint {path} disagrees on {key}")
        n_done = int(state["n_done"])
        next_idx = int(state["next_idx"])
        if n_done != expected or next_idx != expected:
            raise ValueError(
                f"checkpoint {path} completed {n_done}/{next_idx}, expected {expected}"
            )
        rank_sum = state["jacobian_sum"]
        if sorted(rank_sum) != source_layers:
            raise ValueError(f"checkpoint {path} has unexpected layer keys")
        if merged_sum is None:
            merged_sum = {
                layer: rank_sum[layer].float().clone() for layer in source_layers
            }
            d_model = int(merged_sum[source_layers[0]].shape[0])
        else:
            for layer in source_layers:
                if rank_sum[layer].shape != merged_sum[layer].shape:
                    raise ValueError(f"checkpoint {path} has an incompatible shape")
                merged_sum[layer].add_(rank_sum[layer].float())
        n_total += n_done
        del state, rank_sum
        gc.collect()
    assert merged_sum is not None and d_model is not None
    means = {layer: value.div_(n_total) for layer, value in merged_sum.items()}
    return jlens.JacobianLens(jacobians=means, n_prompts=n_total, d_model=d_model)


def _final_logits(lens_model: Any, prompt: str, target_layer: int) -> torch.Tensor:
    input_ids = lens_model.encode(prompt, max_length=128)
    with (
        ActivationRecorder(lens_model.layers, at=[target_layer]) as recorder,
        torch.no_grad(),
    ):
        lens_model.forward(input_ids)
        residual = recorder.activations[target_layer][:, -1, :]
        return lens_model.unembed(residual)[0].float().cpu()


def compare_rank_zero_reference(
    local_lens: jlens.JacobianLens,
    lens_model: Any,
    prompt: str,
    *,
    model_id: str,
    revision: str,
    target_layer: int,
    reference_path: Path,
) -> dict[str, Any]:
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    candidate = {
        "metadata": {
            "model": model_id,
            "revision": revision,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "source_layers": local_lens.source_layers,
            "target_layer": target_layer,
        },
        "rows": local_lens.jacobians,
        "final_logits": _final_logits(lens_model, prompt, target_layer),
    }
    metrics = compare_sample_artifacts(candidate, reference)
    metrics["passed"] = bool(
        metrics["global_cosine"] >= 0.999
        and metrics["global_relative_l2"] <= 0.05
        and metrics["forward"]["argmax_agreement"]
    )
    metrics["thresholds"] = {
        "min_row_cosine": 0.999,
        "max_row_relative_l2": 0.05,
        "require_forward_argmax_agreement": True,
    }
    return metrics


def _stage_name(num_prompts: int) -> str:
    return f"{num_prompts:04d}"


def _export_stage(
    lens: jlens.JacobianLens,
    stage_dir: Path,
    *,
    export_layer_files: bool,
) -> dict[str, Any]:
    lens_path = stage_dir / "lens.pt"
    _atomic_lens_save(lens, lens_path)
    artifacts: dict[str, Any] = {
        "lens": {
            "path": str(lens_path),
            "sha256": _sha256(lens_path),
            "bytes": lens_path.stat().st_size,
            "dtype": "float16",
        },
        "layer_files": [],
    }
    if export_layer_files:
        for layer in lens.source_layers:
            path = stage_dir / "layers" / f"J_{layer:03d}.pt"
            _atomic_torch_save(lens.jacobians[layer].float(), path)
            artifacts["layer_files"].append(
                {
                    "layer": layer,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                    "dtype": "float32",
                }
            )
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompts-jsonl", required=True, type=Path)
    parser.add_argument("--expected-corpus-sha256", required=True)
    parser.add_argument("--partition-offset", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--layers", type=parse_layers)
    parser.add_argument("--source-layer-count", type=int, default=25)
    parser.add_argument("--target-layer", type=int)
    parser.add_argument("--dim-batch", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--reference-samples", type=Path)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--expected-model-manifest-sha256")
    parser.add_argument(
        "--export-layer-files", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--merge-wait-seconds", type=float, default=7200)
    args = parser.parse_args()
    if args.num_prompts < args.expected_world_size:
        parser.error("--num-prompts must assign at least one prompt to every rank")
    if args.partition_offset < 0:
        parser.error("--partition-offset must be non-negative")
    if args.dim_batch < 1 or args.checkpoint_every < 1:
        parser.error("dim batch and checkpoint interval must be positive")
    if args.source_layer_count < 1:
        parser.error("source layer count must be positive")
    if (args.model_manifest is None) != (args.expected_model_manifest_sha256 is None):
        parser.error(
            "model manifest path and expected SHA-256 must be supplied together"
        )

    context = independent_rank(args.expected_world_size)
    logging.basicConfig(
        level=logging.INFO,
        format=(
            f"%(asctime)s | rank={context.rank} | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )
    stage_dir = args.output_dir / "stages" / _stage_name(args.num_prompts)
    completed_result = stage_dir / "result.json"
    if completed_result.exists():
        raise FileExistsError(f"refusing to repeat completed stage: {completed_result}")

    corpus_sha256 = _sha256(args.prompts_jsonl)
    if corpus_sha256 != args.expected_corpus_sha256:
        raise ValueError(
            f"corpus SHA-256 is {corpus_sha256}, expected {args.expected_corpus_sha256}"
        )
    model_manifest = None
    if args.model_manifest is not None:
        model_manifest = validate_model_manifest(
            args.model_manifest,
            expected_sha256=args.expected_model_manifest_sha256,
            model_dir=Path(args.model),
        )
    prompts, prompt_indices = load_assigned_prompts(
        args.prompts_jsonl,
        offset=args.partition_offset,
        total=args.num_prompts,
        rank=context.rank,
        world_size=context.world_size,
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(context.device)
    model, tokenizer = _load_model(args.model, args.revision, context.device)
    lens_model = jlens.from_hf(model, tokenizer, compile=False)
    target_layer = (
        lens_model.n_layers - 1 if args.target_layer is None else args.target_layer
    )
    source_layers = (
        evenly_spaced_source_layers(target_layer, args.source_layer_count)
        if args.layers is None
        else args.layers
    )
    if any(layer < 0 or layer >= target_layer for layer in source_layers):
        raise ValueError("every source layer must be in [0, target_layer)")

    immutable_config = {
        "schema_version": SCHEMA_VERSION,
        "method": "official-exact-coordinate-jacobian-independent-dp",
        "model": args.model,
        "revision": args.revision,
        "model_manifest": model_manifest,
        "corpus": str(args.prompts_jsonl),
        "corpus_sha256": corpus_sha256,
        "partition_offset": args.partition_offset,
        "world_size": context.world_size,
        "source_layers": source_layers,
        "target_layer": target_layer,
        "d_model": lens_model.d_model,
        "dim_batch": args.dim_batch,
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
        "activation_dtype": "bfloat16",
        "accumulator_dtype": "float32",
    }
    config_sha256 = ensure_immutable_config(
        args.output_dir / "fit_config.json",
        immutable_config,
        rank=context.rank,
        timeout_seconds=args.merge_wait_seconds,
    )

    checkpoint_path = prepare_rank_checkpoint(args.output_dir, rank=context.rank)
    rank_dir = checkpoint_path.parent
    validate_checkpoint_progress(checkpoint_path, assigned_prompts=len(prompts))
    logger.info(
        "fitting absolute prompt indices %s..%s (%d prompts)",
        prompt_indices[0],
        prompt_indices[-1],
        len(prompts),
    )
    local_lens = jlens.fit(
        lens_model,
        prompts,
        source_layers=source_layers,
        target_layer=target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=str(checkpoint_path),
        checkpoint_every=args.checkpoint_every,
        resume=True,
    )
    if local_lens.n_prompts != len(prompts):
        raise RuntimeError(
            f"rank {context.rank} fitted {local_lens.n_prompts}, "
            f"expected {len(prompts)} prompts"
        )
    shard_path = rank_dir / "lens.pt"
    _atomic_lens_save(local_lens, shard_path)

    reference_equivalence = None
    if args.reference_samples is not None and context.is_primary:
        if len(prompts) != 1:
            raise ValueError(
                "the DP1 reference gate requires exactly one prompt on rank zero"
            )
        reference_equivalence = compare_rank_zero_reference(
            local_lens,
            lens_model,
            prompts[0],
            model_id=args.model,
            revision=args.revision,
            target_layer=target_layer,
            reference_path=args.reference_samples,
        )
        _atomic_json(reference_equivalence, stage_dir / "reference_equivalence.json")
        if not reference_equivalence["passed"]:
            raise RuntimeError("rank-zero full-computation DP1 equivalence gate failed")

    rank_result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "rank": context.rank,
        "local_rank": context.local_rank,
        "world_size": context.world_size,
        "num_prompts_global": args.num_prompts,
        "num_prompts_local": len(prompts),
        "prompt_indices": prompt_indices,
        "checkpoint": str(checkpoint_path),
        "shard_lens": str(shard_path),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(context.device) / GIB,
        "elapsed_seconds": time.perf_counter() - started,
        "experiment_image": os.environ.get("EXPERIMENT_IMAGE"),
        "model_manifest": model_manifest,
    }
    if reference_equivalence is not None:
        rank_result["reference_equivalence"] = reference_equivalence
    rank_result_path = stage_dir / "rank-results" / f"rank-{context.rank:03d}.json"
    _atomic_json(rank_result, rank_result_path)

    del local_lens, lens_model, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if not context.is_primary:
        return

    rank_result_paths = [
        stage_dir / "rank-results" / f"rank-{rank:03d}.json"
        for rank in range(context.world_size)
    ]
    for path in rank_result_paths:
        _wait_for_path(path, timeout_seconds=args.merge_wait_seconds)
    rank_results = [
        json.loads(path.read_text(encoding="utf-8")) for path in rank_result_paths
    ]
    for rank, result in enumerate(rank_results):
        if (
            result.get("status") != "completed"
            or result.get("rank") != rank
            or result.get("num_prompts_global") != args.num_prompts
        ):
            raise RuntimeError(f"rank receipt {rank_result_paths[rank]} is stale")

    expected_counts = [
        len(
            assigned_prompt_indices(
                offset=args.partition_offset,
                total=args.num_prompts,
                rank=rank,
                world_size=context.world_size,
            )
        )
        for rank in range(context.world_size)
    ]
    checkpoints = [
        args.output_dir / "shards" / f"rank-{rank:03d}" / "checkpoint.pt"
        for rank in range(context.world_size)
    ]
    merged = merge_fp32_checkpoints(
        checkpoints,
        expected_counts=expected_counts,
        source_layers=source_layers,
        target_layer=target_layer,
        skip_first=args.skip_first,
    )
    if merged.n_prompts != args.num_prompts:
        raise RuntimeError(
            f"merged {merged.n_prompts} prompts, expected {args.num_prompts}"
        )
    artifacts = _export_stage(
        merged, stage_dir, export_layer_files=args.export_layer_files
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scientifically_usable_lens": False,
        "requires_section_2_2_validation": True,
        "num_prompts": merged.n_prompts,
        "fit_config": str(args.output_dir / "fit_config.json"),
        "fit_config_sha256": config_sha256,
        "rank_results": rank_results,
        "reference_equivalence": reference_equivalence,
        "artifacts": artifacts,
        "elapsed_seconds_rank_zero": time.perf_counter() - started,
        "experiment_image": os.environ.get("EXPERIMENT_IMAGE"),
    }
    _atomic_json(result, completed_result)
    _atomic_json(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": _stage_name(args.num_prompts),
            "result": str(completed_result),
            "lens": artifacts["lens"]["path"],
            "lens_sha256": artifacts["lens"]["sha256"],
            "num_prompts": args.num_prompts,
        },
        args.output_dir / "latest.json",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
