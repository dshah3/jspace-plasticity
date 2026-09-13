from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from jspace_plasticity.lens.fit_exact_dp import (
    assigned_prompt_indices,
    ensure_immutable_config,
    load_assigned_prompts,
    merge_fp32_checkpoints,
    model_load_kwargs,
    prepare_rank_checkpoint,
    validate_checkpoint_progress,
    validate_model_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _checkpoint(
    path: Path,
    *,
    values: dict[int, float],
    n_done: int,
    next_idx: int | None = None,
) -> None:
    torch.save(
        {
            "jacobian_sum": {
                layer: torch.full((2, 2), value, dtype=torch.float32)
                for layer, value in values.items()
            },
            "n_done": n_done,
            "next_idx": n_done if next_idx is None else next_idx,
            "source_layers": sorted(values),
            "target_layer": 3,
            "skip_first": 16,
        },
        path,
    )


def test_prompt_assignment_is_disjoint_and_prefix_stable() -> None:
    gate = [
        assigned_prompt_indices(offset=0, total=8, rank=rank, world_size=8)
        for rank in range(8)
    ]
    pilot = [
        assigned_prompt_indices(offset=0, total=64, rank=rank, world_size=8)
        for rank in range(8)
    ]
    assert sorted(index for shard in gate for index in shard) == list(range(8))
    assert sorted(index for shard in pilot for index in shard) == list(range(64))
    assert all(
        long[: len(short)] == short for short, long in zip(gate, pilot, strict=True)
    )


def test_load_assigned_prompts_uses_frozen_row_indices(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(json.dumps({"text": f"row-{index}"}) + "\n" for index in range(12))
    )
    prompts, indices = load_assigned_prompts(
        corpus, offset=2, total=8, rank=1, world_size=4
    )
    assert indices == [3, 7]
    assert prompts == ["row-3", "row-7"]


def test_model_load_kwargs_places_whole_model_on_local_gpu() -> None:
    class Config:
        vision_config = None

    kwargs = model_load_kwargs(
        Config(), revision="revision", device=torch.device("cuda", 3)
    )
    assert kwargs["device_map"] == "cuda:3"
    assert kwargs["dtype"] is torch.bfloat16
    assert "tp_plan" not in kwargs
    assert "distributed_config" not in kwargs


def test_model_load_kwargs_accepts_multimodal_wrapper_config() -> None:
    class Config:
        vision_config = object()

    kwargs = model_load_kwargs(
        Config(), revision="revision", device=torch.device("cuda", 1)
    )
    assert kwargs["device_map"] == "cuda:1"
    assert kwargs["dtype"] is torch.bfloat16


def test_recovered_checkpoint_manifest_binds_path_and_sizes(tmp_path: Path) -> None:
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "weights.safetensors").write_bytes(b"weights")
    manifest = {
        "path": str(model_dir),
        "total_bytes": 9,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": "unused"},
            {"path": "weights.safetensors", "bytes": 7, "sha256": "unused"},
        ],
    }
    manifest_path = tmp_path / "checkpoint-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt = validate_model_manifest(
        manifest_path,
        expected_sha256=manifest_sha,
        model_dir=model_dir,
    )
    assert receipt["files"] == 2
    assert receipt["total_bytes"] == 9
    (model_dir / "weights.safetensors").write_bytes(b"short")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_model_manifest(
            manifest_path,
            expected_sha256=manifest_sha,
            model_dir=model_dir,
        )


def test_immutable_config_allows_exact_resume_and_rejects_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fit_config.json"
    payload = {"model": "model", "world_size": 8}
    first_hash = ensure_immutable_config(path, payload, rank=0)
    assert ensure_immutable_config(path, payload, rank=0) == first_hash
    with pytest.raises(ValueError, match="incompatible"):
        ensure_immutable_config(path, {**payload, "world_size": 4}, rank=0)


def test_checkpoint_progress_rejects_smaller_resume_target(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    _checkpoint(path, values={0: 1.0}, n_done=3)
    validate_checkpoint_progress(path, assigned_prompts=3)
    with pytest.raises(ValueError, match="assigns only 2"):
        validate_checkpoint_progress(path, assigned_prompts=2)


def test_all_rank_checkpoint_directories_exist_before_fit(tmp_path: Path) -> None:
    checkpoints = [prepare_rank_checkpoint(tmp_path, rank=rank) for rank in range(8)]
    assert checkpoints == [
        tmp_path / "shards" / f"rank-{rank:03d}" / "checkpoint.pt" for rank in range(8)
    ]
    assert all(path.parent.is_dir() for path in checkpoints)
    assert not any(path.exists() for path in checkpoints)


def test_fp32_merge_uses_rank_sums_not_fp16_shards(tmp_path: Path) -> None:
    rank_zero = tmp_path / "rank-zero.pt"
    rank_one = tmp_path / "rank-one.pt"
    _checkpoint(rank_zero, values={0: 2.0, 2: 4.0}, n_done=2)
    _checkpoint(rank_one, values={0: 3.0, 2: 6.0}, n_done=1)
    lens = merge_fp32_checkpoints(
        [rank_zero, rank_one],
        expected_counts=[2, 1],
        source_layers=[0, 2],
        target_layer=3,
        skip_first=16,
    )
    assert lens.n_prompts == 3
    assert lens.source_layers == [0, 2]
    assert torch.allclose(lens.jacobians[0], torch.full((2, 2), 5 / 3))
    assert torch.allclose(lens.jacobians[2], torch.full((2, 2), 10 / 3))
