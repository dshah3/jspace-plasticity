"""Entity-disjoint synthetic multihop SFT recovery sweep.

Each rank trains one frozen arm on one GPU.  Training is raw-prompt,
single-token teacher-forced cross-entropy (SFT), never RL and never CoT.  The
accepted online-current J-space lesion remains active for every gradient
forward in J-space arms.  Validation, screen, and Anthropic transfer entities
never contribute gradients.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from jspace_plasticity.config import InterventionConfig, ModelConfig
from jspace_plasticity.evals.two_hop_probe import expected_token_id, load_probe_swap
from jspace_plasticity.intervention import JSpaceAblator
from jspace_plasticity.lens import LensMatrices
from jspace_plasticity.modeling import load_model_and_tokenizer
from jspace_plasticity.rehab_sft import _memory, _parameter_probe
from jspace_plasticity.tasks.closedbook_geo import load_split

SCHEMA_VERSION = 1
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LESION_LAYERS = [16, 18, 19, 20, 21, 22]
EXPECTED_WORLD_SIZE = 8
TRIAGE_RANKS = {
    "train": {"jspace": 0, "matched_random": 1},
    "val": {"jspace": 2, "matched_random": 3},
    "screen": {"jspace": 4, "matched_random": 5},
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _require_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256_path(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return observed


def load_design(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require_sha(path, expected_sha256, "recovery design")
    design = json.loads(path.read_text(encoding="utf-8"))
    calibration = design.get("decision") == "full_decoder_lr_calibration_authorized"
    final_run = design.get("decision") == "final_capability_check_authorized"
    corrected = (
        final_run
        or calibration
        or design.get("decision") == "corrected_gain_sft_retraining_authorized"
    )
    if design.get("decision") not in {
        "synthetic_entity_disjoint_sft_recovery_sweep_authorized",
        "corrected_gain_sft_retraining_authorized",
        "full_decoder_lr_calibration_authorized",
        "final_capability_check_authorized",
    }:
        raise ValueError("design does not authorize the synthetic recovery sweep")
    if design.get("checkpoint") != {
        "name": MODEL_ID,
        "revision": MODEL_REVISION,
    }:
        raise ValueError("recovery design checkpoint mismatch")
    lesion = design.get("lesion", {})
    expected_lesion = {
        "selection_source": "online_current",
        "projection": "sequential",
        "layers": LESION_LAYERS,
        "k": 10,
        "exclude_output_top_k": 10,
        "strength": 1.0,
        "matched_random_resampling": "per_example",
    }
    if corrected:
        expected_lesion["direction_convention"] = "effective_gain"
        if design["training"].get("gradient_checkpointing") is not False:
            raise ValueError("corrected retraining pins checkpointing off")
        if design["training"].get("graph_version") != (
            "hooks_span_backward_clean_autocast_isolated_v2"
        ):
            raise ValueError("corrected retraining requires isolated autocast graph")
    if lesion != expected_lesion:
        raise ValueError("recovery design lesion mismatch")
    arms = design.get("arms", [])
    expected_arms = 2 if calibration else (4 if corrected else EXPECTED_WORLD_SIZE)
    if len(arms) != expected_arms:
        raise ValueError(f"design must contain {expected_arms} arms")
    expected_steps = 100 if final_run else 160
    if (
        corrected
        and not calibration
        and [arm.get("name") for arm in arms]
        != [
            f"j-full-s{expected_steps}-primary",
            f"j-full-s{expected_steps}-replicate",
            f"random-full-s{expected_steps}",
            f"sham-full-s{expected_steps}",
        ]
    ):
        raise ValueError("corrected retraining arm assignments drifted")
    if [arm.get("index") for arm in arms] != list(range(expected_arms)):
        raise ValueError("design arm indices must be ordered and contiguous")
    if len({arm.get("name") for arm in arms}) != len(arms):
        raise ValueError("design arm names must be unique")
    if (
        not calibration
        and sum(arm.get("name") == design["evaluation"]["primary_arm"] for arm in arms)
        != 1
    ):
        raise ValueError("design must contain exactly one primary arm")
    for arm in arms:
        if arm.get("training_condition") not in {
            "jspace",
            "matched_random",
            "sham",
        }:
            raise ValueError("unknown recovery training condition")
        if not 0 < float(arm.get("data_fraction", 0)) <= 1:
            raise ValueError("arm data fraction must be in (0, 1]")
        if int(arm.get("expected_train_rows", 0)) < 1:
            raise ValueError("arm expected_train_rows must be positive")
        if int(arm.get("expected_train_clusters", 0)) < 1:
            raise ValueError("arm expected_train_clusters must be positive")
        if int(arm.get("max_steps", 0)) < 1:
            raise ValueError("arm max_steps must be positive")
    if final_run:
        if design["training"]["learning_rate"] != 1e-6:
            raise ValueError("final capability check pins selected LR to 1e-6")
        if any(arm["max_steps"] != 100 for arm in arms):
            raise ValueError("final capability check pins duration to 100 steps")
        if [a["training_condition"] for a in arms] != [
            "jspace",
            "jspace",
            "matched_random",
            "sham",
        ]:
            raise ValueError("final capability check requires matched controls")
        if [a["seed"] for a in arms] != [20260825, 20260826, 20260825, 20260825]:
            raise ValueError("final capability check seed assignments drifted")
        if [a["save_checkpoint"] for a in arms] != [True, True, False, False]:
            raise ValueError("final capability check retains only two J checkpoints")
    if calibration:
        from jspace_plasticity.recovery_lr_calibration import validate_design

        validate_design(design)
    return design


def arm_for(design: dict[str, Any], index: int) -> dict[str, Any]:
    if not 0 <= index < len(design["arms"]):
        raise ValueError("arm index is outside the frozen sweep")
    arm = design["arms"][index]
    if arm["index"] != index:
        raise ValueError("arm ordering drift")
    return arm


def _find_expected_hash(summary: dict[str, Any], result_path: Path) -> str:
    matches = [
        value
        for path, value in summary["rank_result_sha256"].items()
        if Path(path).name == result_path.name
        and Path(path).parent.name == result_path.parent.name
    ]
    if len(matches) != 1:
        raise ValueError(f"triage summary does not uniquely bind {result_path}")
    return matches[0]


def load_triage(
    triage_dir: Path,
    *,
    summary_sha256: str,
    eligible_sha256: str,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    summary_path = triage_dir / "summary.json"
    _require_sha(summary_path, summary_sha256, "triage summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("triage_passed") is not True:
        raise ValueError("synthetic triage did not pass")
    eligibility_path = triage_dir / "eligible_train_rows.jsonl"
    _require_sha(eligibility_path, eligible_sha256, "eligible train rows")
    eligibility = read_jsonl(eligibility_path)
    if len(eligibility) != summary.get("eligible_train_rows"):
        raise ValueError("triage eligibility row count mismatch")

    cohorts: dict[str, dict[str, dict[str, Any]]] = {}
    for split, conditions in TRIAGE_RANKS.items():
        by_condition: dict[str, dict[str, dict[str, Any]]] = {}
        for condition, rank in conditions.items():
            rank_dir = triage_dir / "ranks" / f"rank-{rank:02d}"
            result_path = rank_dir / "result.json"
            _require_sha(
                result_path,
                _find_expected_hash(summary, result_path),
                f"triage rank {rank} result",
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("split") != split or result.get("condition") != condition:
                raise ValueError(f"triage rank {rank} assignment mismatch")
            predictions_path = rank_dir / "predictions.jsonl"
            _require_sha(
                predictions_path,
                result["predictions_sha256"],
                f"triage rank {rank} predictions",
            )
            by_condition[condition] = {
                row["source_id"]: row for row in read_jsonl(predictions_path)
            }
        jspace = by_condition["jspace"]
        random_control = by_condition["matched_random"]
        if set(jspace) != set(random_control):
            raise ValueError(f"triage condition rows differ for {split}")
        clean_correct: dict[str, dict[str, Any]] = {}
        for source_id, row in jspace.items():
            other = random_control[source_id]
            for field in (
                "token_compatible",
                "expected_token_id",
                "clean_predicted_token_id",
                "clean_exact",
            ):
                if row[field] != other[field]:
                    raise ValueError(f"triage clean-path drift for {source_id}")
            if row["token_compatible"] and row["clean_exact"] == 1.0:
                clean_correct[source_id] = {
                    "source_id": source_id,
                    "expected_token_id": int(row["expected_token_id"]),
                    "clean_predicted_token_id": int(row["clean_predicted_token_id"]),
                    "jspace_predicted_token_id": int(
                        row["intervention_predicted_token_id"]
                    ),
                    "random_predicted_token_id": int(
                        other["intervention_predicted_token_id"]
                    ),
                    "baseline_jspace_exact": int(row["intervention_exact"]),
                    "baseline_random_exact": int(other["intervention_exact"]),
                    "cluster_id": row["cluster_id"],
                }
        cohorts[split] = clean_correct
    if set(cohorts["train"]) != {row["source_id"] for row in eligibility}:
        raise ValueError("eligible receipt does not equal clean-correct train cohort")
    return cohorts, eligibility, summary


def intervention_config(
    condition: str, lens_path: Path, *, direction_convention: str = "legacy_weight"
) -> InterventionConfig:
    if condition not in {"jspace", "matched_random"}:
        raise ValueError(f"unknown intervention condition: {condition}")
    return InterventionConfig(
        enabled=True,
        lens_path=str(lens_path),
        layers=LESION_LAYERS,
        k=10,
        exclude_output_top_k=10,
        selection_source="online_current",
        projection="sequential",
        # Frozen 20260825 design: historical replay only. New checks replace this.
        direction_convention=direction_convention,
        strength=1.0,
        control="none" if condition == "jspace" else "matched_random",
        control_resample="per_example",
        random_seed=20260825,
    )


def select_training_rows(
    rows: list[dict[str, Any]],
    eligibility: list[dict[str, Any]],
    data_fraction: float,
    *,
    subset_seed: int = 20260825,
) -> tuple[list[dict[str, Any]], list[str]]:
    by_id = {row["source_id"]: row for row in rows}
    eligible_ids = {row["source_id"] for row in eligibility}
    if not eligible_ids <= set(by_id):
        raise ValueError("eligible receipt references rows outside the train split")
    clusters = sorted({row["cluster_id"] for row in eligibility})
    clusters.sort(
        key=lambda value: hashlib.sha256(f"{subset_seed}:{value}".encode()).digest()
    )
    count = math.ceil(data_fraction * len(clusters))
    selected_clusters = clusters[:count]
    selected_set = set(selected_clusters)
    selected = [
        by_id[row["source_id"]]
        for row in eligibility
        if row["cluster_id"] in selected_set
    ]
    if not selected:
        raise ValueError("data-scale arm selected no gradient rows")
    if {row["cluster_id"] for row in selected} != selected_set:
        raise ValueError("a selected cluster has no eligible rows")
    return selected, selected_clusters


def _encode(
    tokenizer: Any, prompt: str, device: torch.device
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def _normalized_synthetic_rows(
    tokenizer: Any,
    split_rows: dict[str, list[dict[str, Any]]],
    cohorts: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    normalized: dict[str, list[dict[str, Any]]] = {}
    for split, rows in split_rows.items():
        by_id = {row["source_id"]: row for row in rows}
        normalized[split] = []
        for source_id, baseline in sorted(cohorts[split].items()):
            row = by_id[source_id]
            target, _, reason = expected_token_id(
                tokenizer, row["prompt"], row["answer"]
            )
            if reason is not None or target != baseline["expected_token_id"]:
                raise ValueError(f"synthetic target drift for {source_id}: {reason}")
            normalized[split].append(
                {
                    "dataset": "synthetic_geo",
                    "split": split,
                    "source_id": source_id,
                    "cluster_id": row["cluster_id"],
                    "relation": row["relation"],
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                    "expected_token_id": int(target),
                    **baseline,
                }
            )
    return normalized


def _transfer_rows(
    tokenizer: Any,
    data_path: Path,
    data_sha256: str,
    eligibility_path: Path,
    eligibility_sha256: str,
) -> list[dict[str, Any]]:
    _require_sha(data_path, data_sha256, "Anthropic transfer data")
    _require_sha(eligibility_path, eligibility_sha256, "Anthropic eligibility")
    rows, _ = load_probe_swap(data_path)
    eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
    selected = []
    for index, receipt in enumerate(eligibility["rows"]):
        if receipt["example_index"] != index or receipt["name"] != rows[index]["name"]:
            raise ValueError(f"Anthropic eligibility/source drift at row {index}")
        if not receipt["baseline_clean_correct"]:
            continue
        target, _, reason = expected_token_id(
            tokenizer, rows[index]["prompt"], rows[index]["answer"]
        )
        if reason is not None or target != receipt["expected_token_id"]:
            raise ValueError(f"Anthropic transfer target drift at row {index}")
        selected.append(
            {
                "dataset": "anthropic_probe_swap",
                "split": "transfer",
                "source_id": f"anthropic-{index:03d}-{rows[index]['name']}",
                "cluster_id": rows[index]["name"],
                "relation": rows[index]["category"],
                "prompt": rows[index]["prompt"],
                "answer": rows[index]["answer"],
                "expected_token_id": int(target),
            }
        )
    if len(selected) != 49:
        raise ValueError(f"expected 49 Anthropic transfer rows, got {len(selected)}")
    return selected


@torch.inference_mode()
def evaluate_conditions(
    model: Any,
    tokenizer: Any,
    jspace: JSpaceAblator,
    random_control: JSpaceAblator,
    rows: list[dict[str, Any]],
    *,
    phase: str,
) -> list[dict[str, Any]]:
    model.eval()
    records = []
    for row in rows:
        encoded = _encode(tokenizer, row["prompt"], model.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            j_plan = jspace.build_plan(
                model, encoded["input_ids"], encoded["attention_mask"]
            )
            clean_logits = j_plan.clean_next_logits[0].float()
            with jspace.apply(j_plan):
                j_logits = model(**encoded, use_cache=False).logits[0, -1].float()
            random_plan = random_control.build_plan(
                model, encoded["input_ids"], encoded["attention_mask"]
            )
            random_clean_logits = random_plan.clean_next_logits[0].float()
            with random_control.apply(random_plan):
                random_logits = model(**encoded, use_cache=False).logits[0, -1].float()
        if set(j_plan.selected_token_ids) != set(LESION_LAYERS):
            raise RuntimeError("J-space intervention did not fire at every layer")
        if set(random_plan.selected_token_ids) != set(LESION_LAYERS):
            raise RuntimeError("random intervention did not fire at every layer")
        clean_predicted = int(clean_logits.argmax())
        if clean_predicted != int(random_clean_logits.argmax()):
            raise RuntimeError("clean argmax drifted across intervention plans")
        j_predicted = int(j_logits.argmax())
        random_predicted = int(random_logits.argmax())
        target = int(row["expected_token_id"])
        records.append(
            {
                "phase": phase,
                "dataset": row["dataset"],
                "split": row["split"],
                "source_id": row["source_id"],
                "cluster_id": row["cluster_id"],
                "relation": row["relation"],
                "expected_token_id": target,
                "clean_predicted_token_id": clean_predicted,
                "jspace_predicted_token_id": j_predicted,
                "random_predicted_token_id": random_predicted,
                "clean_predicted_token": tokenizer.decode([clean_predicted]),
                "jspace_predicted_token": tokenizer.decode([j_predicted]),
                "random_predicted_token": tokenizer.decode([random_predicted]),
                "clean_exact": int(clean_predicted == target),
                "jspace_exact": int(j_predicted == target),
                "random_exact": int(random_predicted == target),
                "clean_gold_margin": float(clean_logits[target] - clean_logits.max()),
                "jspace_gold_margin": float(j_logits[target] - j_logits.max()),
                "random_gold_margin": float(
                    random_logits[target] - random_logits.max()
                ),
            }
        )
    return records


def verify_initial_synthetic(
    records: list[dict[str, Any]],
    cohorts: dict[str, dict[str, dict[str, Any]]],
    *,
    direction_convention: str = "legacy_weight",
) -> None:
    if direction_convention not in {"legacy_weight", "effective_gain"}:
        raise ValueError("unknown initialization direction convention")
    for row in records:
        baseline = cohorts[row["split"]][row["source_id"]]
        expected = {
            "clean_predicted_token_id": baseline["clean_predicted_token_id"],
            "jspace_predicted_token_id": baseline["jspace_predicted_token_id"],
            "random_predicted_token_id": baseline["random_predicted_token_id"],
        }
        for field, value in expected.items():
            if (
                direction_convention == "effective_gain"
                and field != "clean_predicted_token_id"
            ):
                continue
            if row[field] != value:
                raise ValueError(
                    f"initial synthetic parity failed for {row['source_id']} {field}"
                )


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[f"{row['dataset']}/{row['phase']}/{row['split']}"].append(row)
        if row["dataset"] == "synthetic_geo" and row["split"] in {"val", "screen"}:
            groups[f"{row['dataset']}/{row['phase']}/heldout"].append(row)
    for name, group in sorted(groups.items()):
        summary[name] = {
            "rows": len(group),
            "clusters": len({row["cluster_id"] for row in group}),
            "clean_accuracy": sum(row["clean_exact"] for row in group) / len(group),
            "jspace_accuracy": sum(row["jspace_exact"] for row in group) / len(group),
            "random_accuracy": sum(row["random_exact"] for row in group) / len(group),
        }
    return summary


def _runtime_receipt(model: Any, resolved: Any) -> dict[str, Any]:
    versions = {}
    for package in ("torch", "transformers", "accelerate", "trl", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "python": sys.version,
        "packages": versions,
        "experiment_image": os.environ.get("EXPERIMENT_IMAGE"),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "model_parameters": trainable + frozen,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "final_norm_trainable": any(
            parameter.requires_grad for parameter in resolved.final_norm.parameters()
        ),
        "lm_head_trainable": any(
            parameter.requires_grad for parameter in resolved.lm_head.parameters()
        ),
        "model_storage_dtype": str(next(resolved.layers[0].parameters()).dtype),
        "compute_dtype": "torch.bfloat16 autocast",
        "gradient_checkpointing": bool(
            getattr(model, "is_gradient_checkpointing", False)
        ),
        "final_norm_class": (
            type(resolved.final_norm).__module__
            + "."
            + type(resolved.final_norm).__name__
        ),
        "training_graph_version": "hooks_span_backward_clean_autocast_isolated_v2",
        "trainable_parameter_inventory": [
            {
                "name": name,
                "shape": list(p.shape),
                "numel": p.numel(),
                "dtype": str(p.dtype),
            }
            for name, p in model.named_parameters()
            if p.requires_grad
        ],
        "optimizer": "torch.optim.AdamW; FP32 parameters and optimizer states",
    }


def _write_metrics(output_dir: Path, metrics: list[dict[str, Any]]) -> None:
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for field in ("loss", "grad_norm_pre_clip", "learning_rate", "step_seconds"):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot([row["step"] for row in metrics], [row[field] for row in metrics])
        axis.set_xlabel("optimizer step")
        axis.set_ylabel(field)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / f"{field}.png", dpi=150)
        plt.close(figure)


def _save_checkpoint(model: Any, tokenizer: Any, output_dir: Path) -> dict[str, Any]:
    checkpoint_dir = output_dir / "checkpoint-terminal"
    model.save_pretrained(
        checkpoint_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(checkpoint_dir)
    files = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(checkpoint_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                }
            )
    manifest = {
        "path": str(checkpoint_dir),
        "optimizer_saved": False,
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
    }
    write_json(output_dir / "checkpoint-manifest.json", manifest)
    manifest["manifest_sha256"] = sha256_path(output_dir / "checkpoint-manifest.json")
    return manifest


def _calibration_milestone(
    model: Any,
    tokenizer: Any,
    jspace: JSpaceAblator,
    random_control: JSpaceAblator,
    rows: list[dict[str, Any]],
    *,
    step: int,
    output_dir: Path,
    initial_jspace_correct: int,
    learning_rate: float,
    previous: list[dict[str, Any]],
) -> dict[str, Any]:
    from jspace_plasticity.recovery_lr_calibration import (
        select_candidate,
        validation_counts,
    )

    if any(row["split"] != "val" for row in rows):
        raise ValueError("milestone inference must use validation only")
    stage = output_dir / "milestones" / f"step-{step:04d}"
    stage.mkdir(parents=True, exist_ok=False)
    was_training = model.training
    print(
        json.dumps({"event": "validation_checkpoint_start", "step": step}), flush=True
    )
    try:
        devices = [] if model.device.type == "cpu" else [model.device]
        with torch.random.fork_rng(devices=devices):
            predictions = evaluate_conditions(
                model,
                tokenizer,
                jspace,
                random_control,
                rows,
                phase=f"step-{step:04d}",
            )
            counts = validation_counts(predictions)
            prediction_path = stage / "predictions.jsonl"
            prediction_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
            )
            candidate = {
                **counts,
                "step": step,
                "learning_rate": learning_rate,
                "initial_jspace_correct": initial_jspace_correct,
            }
            earlier = [
                {
                    **r["counts"],
                    "step": r["step"],
                    "learning_rate": learning_rate,
                    "initial_jspace_correct": initial_jspace_correct,
                }
                for r in previous
            ]
            best = select_candidate(earlier + [candidate])
            checkpoint = None
            if best is not None and best["step"] == step:
                # Serialize replacements across arms: at most three full copies
                # coexist, and old weights go only after the new save is hashed.
                with (output_dir.parent / ".checkpoint-retention.lock").open(
                    "a"
                ) as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    checkpoint = _save_checkpoint(model, tokenizer, stage)
                    for old in previous:
                        if old["checkpoint"] is None:
                            continue
                        old_stage = (
                            output_dir / "milestones" / f"step-{old['step']:04d}"
                        )
                        old_weights = old_stage / "checkpoint-terminal"
                        if old_weights.exists():
                            if (
                                old_weights.is_symlink()
                                or old_weights.resolve().parent != old_stage.resolve()
                            ):
                                raise ValueError("unsafe checkpoint retirement path")
                            shutil.rmtree(old_weights)
                            write_json(
                                old_stage / "checkpoint-retired.json",
                                {
                                    "superseded_by_step": step,
                                    "reason": (
                                        "better validation candidate; "
                                        "retention authorized by user"
                                    ),
                                },
                            )
    finally:
        model.train(was_training)
    receipt = {
        "step": step,
        "counts": counts,
        "checkpoint": checkpoint,
        "predictions_sha256": sha256_path(prediction_path),
    }
    write_json(stage / "result.json", receipt)
    print(
        json.dumps({"event": "validation_checkpoint_complete", **receipt}), flush=True
    )
    return receipt


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite arm output: {args.output_dir}")
    design = load_design(args.design, args.design_sha256)
    arm = arm_for(design, args.arm_index)
    calibration = design["decision"] == "full_decoder_lr_calibration_authorized"
    calibration_milestones = []
    learning_rate_setting = float(
        arm["learning_rate"] if calibration else design["training"]["learning_rate"]
    )
    expected_image = os.environ.get("EXPERIMENT_IMAGE")
    if expected_image != args.experiment_image:
        raise ValueError("runtime image does not match the immutable CLI receipt")
    evidence = design["evidence"]
    if evidence["triage_summary"]["sha256"] != args.triage_summary_sha256:
        raise ValueError("CLI triage summary is not the frozen design evidence")
    if evidence["eligible_train_rows"]["sha256"] != args.eligible_sha256:
        raise ValueError("CLI eligibility is not the frozen design evidence")
    if evidence["data"]["sha256"] != args.data_sha256:
        raise ValueError("CLI data is not the frozen design evidence")
    if evidence["published_lens"]["sha256"] != args.lens_sha256:
        raise ValueError("CLI lens is not the frozen design evidence")
    transfer_evidence = evidence["anthropic_transfer"]
    if transfer_evidence["data_sha256"] != args.transfer_data_sha256:
        raise ValueError("CLI transfer data is not frozen evidence")
    if transfer_evidence["eligibility_sha256"] != args.transfer_eligibility_sha256:
        raise ValueError("CLI transfer eligibility is not frozen evidence")

    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "status.json", {"status": "starting"})
    random.seed(int(arm["seed"]))
    torch.manual_seed(int(arm["seed"]))
    torch.cuda.manual_seed_all(int(arm["seed"]))
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")

    cohorts, eligibility, triage_summary = load_triage(
        args.triage_dir,
        summary_sha256=args.triage_summary_sha256,
        eligible_sha256=args.eligible_sha256,
    )
    split_rows = {
        split: load_split(args.data, expected_sha256=args.data_sha256, split=split)[0]
        for split in ("train", "val", "screen")
    }
    selected_train, selected_clusters = select_training_rows(
        split_rows["train"], eligibility, float(arm["data_fraction"])
    )
    if len(selected_train) != int(arm["expected_train_rows"]):
        raise ValueError("selected train row count differs from frozen arm")
    if len(selected_clusters) != int(arm["expected_train_clusters"]):
        raise ValueError("selected train cluster count differs from frozen arm")

    model_config = ModelConfig(
        name_or_path=MODEL_ID,
        revision=MODEL_REVISION,
        dtype="float32",
        attn_implementation="sdpa",
        gradient_checkpointing=False,
        freeze_output_head=True,
    )
    model, tokenizer, resolved = load_model_and_tokenizer(model_config, device)
    runtime = _runtime_receipt(model, resolved)
    if (calibration or design["decision"] == "final_capability_check_authorized") and (
        runtime["trainable_parameters"] != 3570049536
    ):
        raise ValueError(
            "Selected-LR experiments require all current decoder parameters trainable"
        )
    if runtime["final_norm_trainable"] or runtime["lm_head_trainable"]:
        raise ValueError("final norm and unembedding must remain frozen")
    _require_sha(args.lens, args.lens_sha256, "published lens")
    lens = LensMatrices.load(args.lens)
    if lens.n_prompts != 1000:
        raise ValueError(f"expected published n=1000 lens, got {lens.n_prompts}")
    convention = design["lesion"].get("direction_convention", "legacy_weight")
    jspace = JSpaceAblator(
        resolved,
        lens,
        intervention_config("jspace", args.lens, direction_convention=convention),
        device=device,
        dtype=torch.bfloat16,
    )
    random_control = JSpaceAblator(
        resolved,
        lens,
        intervention_config(
            "matched_random", args.lens, direction_convention=convention
        ),
        device=device,
        dtype=torch.bfloat16,
    )
    normalized = _normalized_synthetic_rows(tokenizer, split_rows, cohorts)
    transfer_rows = (
        []
        if calibration
        else _transfer_rows(
            tokenizer,
            args.transfer_data,
            args.transfer_data_sha256,
            args.transfer_eligibility,
            args.transfer_eligibility_sha256,
        )
    )
    heldout_rows = (
        normalized["val"] if calibration else normalized["val"] + normalized["screen"]
    )
    initial = evaluate_conditions(
        model,
        tokenizer,
        jspace,
        random_control,
        heldout_rows + transfer_rows,
        phase="initial",
    )
    verify_initial_synthetic(
        [row for row in initial if row["dataset"] == "synthetic_geo"],
        cohorts,
        direction_convention=convention,
    )

    targets = {}
    eligible_by_id = {row["source_id"]: row for row in eligibility}
    for row in selected_train:
        target, _, reason = expected_token_id(tokenizer, row["prompt"], row["answer"])
        receipt = eligible_by_id[row["source_id"]]
        if reason is not None or target != receipt["expected_token_id"]:
            raise ValueError(f"training target drift for {row['source_id']}: {reason}")
        targets[row["source_id"]] = int(target)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate_setting,
        betas=tuple(design["training"]["betas"]),
        eps=1e-8,
        weight_decay=float(design["training"]["weight_decay"]),
        foreach=False,
    )
    optimizer.zero_grad(set_to_none=True)
    before_names, before_values = _parameter_probe(model)
    metrics = []
    order: list[dict[str, Any]] = []
    cursor = 0
    epoch = 0
    accumulation = int(design["training"]["gradient_accumulation_steps"])
    model.train()
    for step in range(int(arm["max_steps"])):
        started = time.perf_counter()
        losses = []
        source_ids = []
        for _ in range(accumulation):
            if cursor >= len(order):
                order = selected_train.copy()
                random.Random(int(arm["seed"]) + epoch * 100_003).shuffle(order)
                cursor = 0
                epoch += 1
            row = order[cursor]
            cursor += 1
            source_ids.append(row["source_id"])
            encoded = _encode(tokenizer, row["prompt"], device)
            plan = None
            context = nullcontext()
            if arm["training_condition"] != "sham":
                ablator = (
                    jspace if arm["training_condition"] == "jspace" else random_control
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    plan = ablator.build_plan(
                        model, encoded["input_ids"], encoded["attention_mask"]
                    )
                context = ablator.apply(plan)
            with context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(**encoded, use_cache=False).logits[:, -1].float()
                    target = torch.tensor(
                        [targets[row["source_id"]]], dtype=torch.long, device=device
                    )
                    loss = F.cross_entropy(logits, target)
                (loss / accumulation).backward()
            if plan is not None and set(plan.selected_token_ids) != set(LESION_LAYERS):
                raise RuntimeError("training intervention did not fire at every layer")
            losses.append(float(loss.detach()))
        if step == 0:
            write_json(
                args.output_dir / "gradient-inventory.json",
                {
                    "parameters": [
                        {
                            "name": name,
                            "numel": p.numel(),
                            "has_grad": p.grad is not None,
                            "grad_norm": None
                            if p.grad is None
                            else float(p.grad.float().norm()),
                        }
                        for name, p in model.named_parameters()
                        if p.requires_grad
                    ],
                    "optimizer": type(optimizer).__module__
                    + "."
                    + type(optimizer).__name__,
                    "foreach": optimizer.defaults.get("foreach"),
                },
            )
        if step == 0 and convention == "effective_gain":
            missing_gradients = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            if missing_gradients:
                raise RuntimeError(
                    f"corrected retraining missing gradients: {missing_gradients}"
                )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, float(design["training"]["max_grad_norm"])
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step + 1}")
        warmup = int(design["training"]["warmup_steps"])
        base_lr = learning_rate_setting
        learning_rate = base_lr * min(1.0, (step + 1) / max(1, warmup))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metric = {
            "step": step + 1,
            "loss": sum(losses) / len(losses),
            "grad_norm_pre_clip": float(grad_norm),
            "learning_rate": learning_rate,
            "epoch_passes": (step + 1) * accumulation / len(selected_train),
            "step_seconds": time.perf_counter() - started,
            "source_ids": json.dumps(source_ids),
            **_memory(),
        }
        if not math.isfinite(metric["loss"]):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        metrics.append(metric)
        print(json.dumps(metric, sort_keys=True), flush=True)
        if calibration and step + 1 in design["evaluation"]["milestones"]:
            calibration_milestones.append(
                _calibration_milestone(
                    model,
                    tokenizer,
                    jspace,
                    random_control,
                    normalized["val"],
                    step=step + 1,
                    output_dir=args.output_dir,
                    initial_jspace_correct=sum(r["jspace_exact"] for r in initial),
                    learning_rate=learning_rate_setting,
                    previous=calibration_milestones,
                )
            )
            _write_metrics(args.output_dir, metrics)
            write_json(
                args.output_dir / "calibration-progress.json",
                {
                    "milestones": calibration_milestones,
                },
            )

    after_names, after_values = _parameter_probe(model)
    if before_names != after_names:
        raise RuntimeError("parameter probe names changed during training")
    delta = (after_values - before_values).abs()
    update_probe = {
        "parameter_names": before_names,
        "sampled_values": int(delta.numel()),
        "changed_values": int((delta > 0).sum()),
        "changed_fraction": float((delta > 0).float().mean()),
        "max_abs_delta": float(delta.max()),
    }
    if update_probe["changed_values"] == 0:
        raise RuntimeError("optimizer did not change any sampled FP32 parameter")

    terminal = (
        []
        if calibration
        else evaluate_conditions(
            model,
            tokenizer,
            jspace,
            random_control,
            normalized["train"] + heldout_rows + transfer_rows,
            phase="terminal",
        )
    )
    _write_metrics(args.output_dir, metrics)
    predictions_path = args.output_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in initial + terminal),
        encoding="utf-8",
    )
    checkpoint = (
        _save_checkpoint(model, tokenizer, args.output_dir)
        if arm["save_checkpoint"]
        else None
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "protocol": "full_decoder_lr_calibration"
        if calibration
        else "synthetic_entity_disjoint_sft_recovery_sweep_graph_v2",
        "arm": arm,
        "learning_rate": learning_rate_setting,
        "calibration_milestones": calibration_milestones,
        "checkpoint": MODEL_ID,
        "checkpoint_revision": MODEL_REVISION,
        "experiment_image": args.experiment_image,
        "design_sha256": args.design_sha256,
        "triage_summary_sha256": args.triage_summary_sha256,
        "eligible_train_rows_sha256": args.eligible_sha256,
        "triage_summary": triage_summary,
        "objective": "single_token_answer_only_teacher_forced_cross_entropy",
        "no_chat_template": True,
        "thinking_disabled": True,
        "rl": False,
        "training_intervention": arm["training_condition"],
        "intervention_config": asdict(jspace.config),
        "direction_convention": convention,
        "selected_train_rows": len(selected_train),
        "selected_train_clusters": selected_clusters,
        "examples_seen": int(arm["max_steps"]) * accumulation,
        "eligible_dataset_passes": int(arm["max_steps"])
        * accumulation
        / len(selected_train),
        "runtime": runtime,
        "memory": _memory(),
        "parameter_update_probe": update_probe,
        "evaluation": summarize(initial + terminal),
        "predictions_sha256": sha256_path(predictions_path),
        "saved_checkpoint": checkpoint,
        "claim_boundary": design["claim_boundary"],
    }
    write_json(args.output_dir / "result.json", result)
    write_json(args.output_dir / "status.json", {"status": "completed"})
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _binomial_two_sided(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    lower = (
        sum(
            math.comb(trials, value)
            for value in range(min(successes, trials - successes) + 1)
        )
        / 2**trials
    )
    return min(1.0, 2 * lower)


def paired_change(
    initial: list[dict[str, Any]],
    terminal: list[dict[str, Any]],
    *,
    field: str,
) -> dict[str, Any]:
    terminal_by_id = {row["source_id"]: row for row in terminal}
    if set(terminal_by_id) != {row["source_id"] for row in initial}:
        raise ValueError("paired change rows differ")
    pairs = [
        (int(row[field]), int(terminal_by_id[row["source_id"]][field]))
        for row in initial
    ]
    recovered = sum(before == 0 and after == 1 for before, after in pairs)
    regressed = sum(before == 1 and after == 0 for before, after in pairs)
    count = len(pairs)
    return {
        "rows": count,
        "initial_correct": sum(before for before, _ in pairs),
        "terminal_correct": sum(after for _, after in pairs),
        "recovered": recovered,
        "regressed": regressed,
        "net_correct": recovered - regressed,
        "net_accuracy_gain": (recovered - regressed) / count,
        "mcnemar_exact_two_sided_p": _binomial_two_sided(
            min(recovered, regressed), recovered + regressed
        ),
    }


def paired_difference(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    field: str,
) -> dict[str, Any]:
    right_by_id = {row["source_id"]: row for row in right}
    if set(right_by_id) != {row["source_id"] for row in left}:
        raise ValueError("paired arm rows differ")
    pairs = [
        (int(row[field]), int(right_by_id[row["source_id"]][field])) for row in left
    ]
    left_only = sum(a == 1 and b == 0 for a, b in pairs)
    right_only = sum(a == 0 and b == 1 for a, b in pairs)
    count = len(pairs)
    return {
        "rows": count,
        "left_correct": sum(a for a, _ in pairs),
        "right_correct": sum(b for _, b in pairs),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "left_minus_right": (left_only - right_only) / count,
        "mcnemar_exact_two_sided_p": _binomial_two_sided(
            min(left_only, right_only), left_only + right_only
        ),
    }


def _select_predictions(
    rows: list[dict[str, Any]],
    *,
    phase: str,
    dataset: str,
    splits: set[str],
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["phase"] == phase
        and row["dataset"] == dataset
        and row["split"] in splits
    ]


def _write_sweep_tracking(
    output_dir: Path,
    arm_metrics: dict[str, dict[str, Any]],
) -> dict[str, str]:
    rows = []
    for name, metrics in arm_metrics.items():
        heldout = metrics["synthetic_heldout_jspace_change"]
        transfer = metrics["anthropic_transfer_jspace_change"]
        rows.append(
            {
                "arm": name,
                "training_condition": metrics["arm"]["training_condition"],
                "data_fraction": metrics["arm"]["data_fraction"],
                "max_steps": metrics["arm"]["max_steps"],
                "seed": metrics["arm"]["seed"],
                "selected_train_rows": metrics["selected_train_rows"],
                "eligible_dataset_passes": metrics["eligible_dataset_passes"],
                "heldout_initial_jspace_accuracy": heldout["initial_correct"]
                / heldout["rows"],
                "heldout_terminal_jspace_accuracy": heldout["terminal_correct"]
                / heldout["rows"],
                "heldout_jspace_net_gain": heldout["net_accuracy_gain"],
                "heldout_jspace_mcnemar_p": heldout["mcnemar_exact_two_sided_p"],
                "heldout_terminal_clean_accuracy": metrics[
                    "synthetic_terminal_clean_accuracy"
                ],
                "heldout_terminal_random_accuracy": metrics[
                    "synthetic_terminal_random_accuracy"
                ],
                "screen_jspace_net_gain": metrics["synthetic_screen_jspace_change"][
                    "net_accuracy_gain"
                ],
                "transfer_initial_jspace_accuracy": transfer["initial_correct"]
                / transfer["rows"],
                "transfer_terminal_jspace_accuracy": transfer["terminal_correct"]
                / transfer["rows"],
                "transfer_jspace_net_gain": transfer["net_accuracy_gain"],
                "transfer_terminal_clean_accuracy": metrics[
                    "anthropic_terminal_clean_accuracy"
                ],
            }
        )
    csv_path = output_dir / "sweep_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["arm"]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths = []
    dose = sorted(
        (
            row
            for row in rows
            if row["training_condition"] == "jspace"
            and row["data_fraction"] == 1.0
            and row["seed"] == 20260825
            and row["max_steps"] in {40, 100, 160, 320}
        ),
        key=lambda row: row["max_steps"],
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        [row["max_steps"] for row in dose],
        [row["heldout_terminal_jspace_accuracy"] for row in dose],
        marker="o",
        label="terminal J-space",
    )
    axis.axhline(
        dose[0]["heldout_initial_jspace_accuracy"],
        color="black",
        linestyle="--",
        label="initial J-space",
    )
    axis.set(xlabel="optimizer steps", ylabel="heldout exact accuracy", ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    dose_path = output_dir / "dose_curve.png"
    figure.savefig(dose_path, dpi=150)
    plt.close(figure)
    plot_paths.append(dose_path)

    scale = sorted(
        (
            row
            for row in rows
            if row["training_condition"] == "jspace"
            and row["max_steps"] in {100, 160}
            and row["seed"] == 20260825
        ),
        key=lambda row: row["data_fraction"],
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        [row["selected_train_rows"] for row in scale],
        [row["heldout_terminal_jspace_accuracy"] for row in scale],
        marker="o",
    )
    axis.axhline(
        scale[0]["heldout_initial_jspace_accuracy"],
        color="black",
        linestyle="--",
        label="initial J-space",
    )
    axis.set(
        xlabel="distinct eligible training rows",
        ylabel="heldout exact accuracy",
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    scale_path = output_dir / "data_scale_curve.png"
    figure.savefig(scale_path, dpi=150)
    plt.close(figure)
    plot_paths.append(scale_path)
    return {
        "sweep_metrics_csv_sha256": sha256_path(csv_path),
        **{f"{path.stem}_sha256": sha256_path(path) for path in plot_paths},
    }


def reduce_results(args: argparse.Namespace) -> dict[str, Any]:
    design = load_design(args.design, args.design_sha256)
    if design["decision"] == "full_decoder_lr_calibration_authorized":
        from jspace_plasticity.recovery_lr_calibration import reduce_calibration

        return reduce_calibration(args, design)
    results = {}
    predictions = {}
    result_hashes = {}
    for arm in design["arms"]:
        arm_dir = args.output_dir / "arms" / f"arm-{arm['index']:02d}-{arm['name']}"
        result_path = arm_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed" or result.get("arm") != arm:
            raise ValueError(f"arm {arm['index']} result does not match design")
        if result.get("experiment_image") != args.experiment_image:
            raise ValueError(f"arm {arm['index']} image mismatch")
        if design["lesion"].get("direction_convention") == "effective_gain":
            if result.get("direction_convention") != "effective_gain":
                raise ValueError(f"arm {arm['index']} is not corrected-gain training")
        prediction_path = arm_dir / "predictions.jsonl"
        _require_sha(
            prediction_path,
            result["predictions_sha256"],
            f"arm {arm['index']} predictions",
        )
        results[arm["name"]] = result
        predictions[arm["name"]] = read_jsonl(prediction_path)
        result_hashes[str(result_path)] = sha256_path(result_path)

    reference_initial = [
        row
        for row in predictions[design["arms"][0]["name"]]
        if row["phase"] == "initial"
    ]
    reference_by_id = {
        (row["dataset"], row["source_id"]): row for row in reference_initial
    }
    initial_fields = (
        "expected_token_id",
        "clean_predicted_token_id",
        "jspace_predicted_token_id",
        "random_predicted_token_id",
    )
    for name, rows in predictions.items():
        initial = [row for row in rows if row["phase"] == "initial"]
        initial_ids = {(row["dataset"], row["source_id"]) for row in initial}
        if set(reference_by_id) != initial_ids:
            raise ValueError(f"initial evaluation rows differ in arm {name}")
        for row in initial:
            reference = reference_by_id[(row["dataset"], row["source_id"])]
            if any(row[field] != reference[field] for field in initial_fields):
                raise ValueError(f"initial predictions drifted in arm {name}")

    arm_metrics = {}
    for arm in design["arms"]:
        name = arm["name"]
        rows = predictions[name]
        initial_heldout = _select_predictions(
            rows,
            phase="initial",
            dataset="synthetic_geo",
            splits={"val", "screen"},
        )
        terminal_heldout = _select_predictions(
            rows,
            phase="terminal",
            dataset="synthetic_geo",
            splits={"val", "screen"},
        )
        initial_screen = [row for row in initial_heldout if row["split"] == "screen"]
        terminal_screen = [row for row in terminal_heldout if row["split"] == "screen"]
        initial_transfer = _select_predictions(
            rows,
            phase="initial",
            dataset="anthropic_probe_swap",
            splits={"transfer"},
        )
        terminal_transfer = _select_predictions(
            rows,
            phase="terminal",
            dataset="anthropic_probe_swap",
            splits={"transfer"},
        )
        arm_metrics[name] = {
            "arm": arm,
            "selected_train_rows": results[name]["selected_train_rows"],
            "eligible_dataset_passes": results[name]["eligible_dataset_passes"],
            "synthetic_heldout_jspace_change": paired_change(
                initial_heldout, terminal_heldout, field="jspace_exact"
            ),
            "synthetic_screen_jspace_change": paired_change(
                initial_screen, terminal_screen, field="jspace_exact"
            ),
            "synthetic_terminal_clean_accuracy": sum(
                row["clean_exact"] for row in terminal_heldout
            )
            / len(terminal_heldout),
            "synthetic_terminal_random_accuracy": sum(
                row["random_exact"] for row in terminal_heldout
            )
            / len(terminal_heldout),
            "anthropic_transfer_jspace_change": paired_change(
                initial_transfer, terminal_transfer, field="jspace_exact"
            ),
            "anthropic_terminal_clean_accuracy": sum(
                row["clean_exact"] for row in terminal_transfer
            )
            / len(terminal_transfer),
        }

    primary_name = design["evaluation"]["primary_arm"]
    primary_terminal = _select_predictions(
        predictions[primary_name],
        phase="terminal",
        dataset="synthetic_geo",
        splits={"val", "screen"},
    )
    control_differences = {}
    sham_name = design["evaluation"].get("sham_arm", "sham-full-s160")
    random_name = design["evaluation"].get("random_arm", "random-full-s160")
    for control in (sham_name, random_name):
        control_terminal = _select_predictions(
            predictions[control],
            phase="terminal",
            dataset="synthetic_geo",
            splits={"val", "screen"},
        )
        control_differences[control] = paired_difference(
            primary_terminal, control_terminal, field="jspace_exact"
        )
    thresholds = design["candidate_recovery_gate"]
    primary = arm_metrics[primary_name]
    clauses = {
        "primary_jspace_net_gain_at_least_0_10": primary[
            "synthetic_heldout_jspace_change"
        ]["net_accuracy_gain"]
        >= thresholds["minimum_primary_jspace_net_gain"],
        "primary_gain_mcnemar_p_at_most_0_05": primary[
            "synthetic_heldout_jspace_change"
        ]["mcnemar_exact_two_sided_p"]
        <= thresholds["maximum_primary_gain_mcnemar_p"],
        "primary_terminal_clean_accuracy_at_least_0_95": primary[
            "synthetic_terminal_clean_accuracy"
        ]
        >= thresholds["minimum_primary_terminal_clean_accuracy"],
        "screen_jspace_net_gain_at_least_0_08": primary[
            "synthetic_screen_jspace_change"
        ]["net_accuracy_gain"]
        >= thresholds["minimum_screen_jspace_net_gain"],
        "primary_minus_sham_terminal_jspace_at_least_0_05": control_differences[
            sham_name
        ]["left_minus_right"]
        >= thresholds["minimum_primary_minus_sham_terminal_jspace"],
        "primary_minus_random_training_terminal_jspace_at_least_0_05": (
            control_differences[random_name]["left_minus_right"]
            >= thresholds["minimum_primary_minus_random_training_terminal_jspace"]
        ),
    }
    candidate = all(clauses.values())
    tracking_hashes = _write_sweep_tracking(args.output_dir, arm_metrics)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "protocol": "synthetic_entity_disjoint_sft_recovery_sweep_graph_v2",
        "experiment_image": args.experiment_image,
        "design_sha256": args.design_sha256,
        "result_sha256": result_hashes,
        "primary_arm": primary_name,
        "direction_convention": design["lesion"].get(
            "direction_convention", "legacy_weight"
        ),
        "arm_metrics": arm_metrics,
        "primary_control_differences": control_differences,
        "candidate_recovery_clauses": clauses,
        "candidate_recovery": candidate,
        "tracking_sha256": tracking_hashes,
        "anthropic_transfer_observed": primary["anthropic_transfer_jspace_change"][
            "net_accuracy_gain"
        ]
        > 0,
        "verdict": (
            "candidate recovery; run fresh-lens anti-evasion and confirmatory seeds"
            if candidate
            else "recovery gate failed; inspect dose/data curves before any follow-up"
        ),
        "claim_boundary": design["claim_boundary"],
        "further_training_authorized": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--design-sha256", required=True)
    parser.add_argument("--triage-dir", required=True, type=Path)
    parser.add_argument("--triage-summary-sha256", required=True)
    parser.add_argument("--eligible-sha256", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--lens", required=True, type=Path)
    parser.add_argument("--lens-sha256", required=True)
    parser.add_argument("--transfer-data", required=True, type=Path)
    parser.add_argument("--transfer-data-sha256", required=True)
    parser.add_argument("--transfer-eligibility", required=True, type=Path)
    parser.add_argument("--transfer-eligibility-sha256", required=True)
    parser.add_argument("--experiment-image", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arm-index", type=int, default=0)
    parser.add_argument("--reduce-only", action="store_true")
    args = parser.parse_args()
    if args.reduce_only:
        reduce_results(args)
    else:
        run_arm(args)


if __name__ == "__main__":
    main()
