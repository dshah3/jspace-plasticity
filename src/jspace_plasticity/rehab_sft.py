"""Answer-only SFT under the accepted Qwen3.5-4B online J-space lesion.

Each process trains one deterministic probe-swap fold on one GPU.  This is a
small, explicitly post-hoc rehabilitation pilot, not RL and not a transfer
benchmark.  Only rows answered correctly by the frozen clean checkpoint can
contribute gradients; validation and heldout rows are evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from jspace_plasticity.config import InterventionConfig, ModelConfig
from jspace_plasticity.evals.two_hop_probe import expected_token_id, load_probe_swap
from jspace_plasticity.intervention import JSpaceAblator
from jspace_plasticity.lens import LensMatrices
from jspace_plasticity.modeling import load_model_and_tokenizer

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LESION_LAYERS = [16, 18, 19, 20, 21, 22]
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(path: Path, expected: str, label: str) -> str:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return observed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_protocol(
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    _require_sha(args.data, args.data_sha256, "probe-swap data")
    _require_sha(args.folds, args.folds_sha256, "fold manifest")
    _require_sha(args.authorization, args.authorization_sha256, "authorization")
    _require_sha(args.lens, args.lens_sha256, "lens")
    rows, _ = load_probe_swap(args.data)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    if (
        folds.get("protocol")
        != "probe_swap_five_fold_answer_only_lesion_rehabilitation"
    ):
        raise ValueError("unexpected fold protocol")
    if folds.get("source", {}).get("dataset_sha256") != args.data_sha256:
        raise ValueError("fold manifest is bound to another dataset")
    if len(folds.get("rows", [])) != len(rows) or len(folds.get("folds", [])) != 5:
        raise ValueError("fold manifest shape is invalid")
    for index, (source, receipt) in enumerate(zip(rows, folds["rows"], strict=True)):
        if receipt["example_index"] != index or receipt["name"] != source["name"]:
            raise ValueError(f"fold/source row mismatch at {index}")
    if authorization.get("decision") != "probe_swap_five_fold_sft_pilot_authorized":
        raise ValueError("authorization does not permit the five-fold SFT pilot")
    checkpoint = authorization.get("checkpoint", {})
    if checkpoint != {"name": args.checkpoint, "revision": args.revision}:
        raise ValueError("authorization checkpoint mismatch")
    lesion = authorization.get("lesion", {})
    expected_lesion = {
        "selection_source": "online_current",
        "projection": "sequential",
        "layers": LESION_LAYERS,
        "k": 10,
        "exclude_output_top_k": 10,
        "strength": 1.0,
    }
    if lesion != expected_lesion:
        raise ValueError("authorization lesion mismatch")
    return rows, folds, authorization


def _fold(folds: dict[str, Any], fold_index: int) -> dict[str, Any]:
    if not 0 <= fold_index < 5:
        raise ValueError("fold must be in 0..4")
    fold = folds["folds"][fold_index]
    if fold["fold"] != fold_index:
        raise ValueError("fold manifest ordering mismatch")
    train = set(fold["train_indices"])
    validation = set(fold["validation_indices"])
    test = set(fold["test_indices"])
    if train & validation or train & test or validation & test:
        raise ValueError("fold partitions overlap")
    if train | validation | test != set(range(90)):
        raise ValueError("fold partitions do not cover all rows")
    eligible = set(fold["eligible_train_indices"])
    if not eligible <= train:
        raise ValueError("gradient-eligible rows must be in train")
    receipt_rows = folds["rows"]
    expected_eligible = {
        index for index in train if receipt_rows[index]["baseline_clean_correct"]
    }
    if eligible != expected_eligible:
        raise ValueError("gradient eligibility does not match baseline receipt")
    return fold


def _intervention() -> InterventionConfig:
    return InterventionConfig(
        enabled=True,
        layers=LESION_LAYERS,
        k=10,
        exclude_output_top_k=10,
        selection_source="online_current",
        projection="sequential",
        strength=1.0,
        control="none",
    )


def _runtime_receipt(
    args: argparse.Namespace, model: Any, resolved: Any
) -> dict[str, Any]:
    tracked = ("torch", "transformers", "accelerate", "trl", "vllm")
    versions = {}
    for package in tracked:
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
            p.requires_grad for p in resolved.final_norm.parameters()
        ),
        "lm_head_trainable": any(
            p.requires_grad for p in resolved.lm_head.parameters()
        ),
        "gradient_checkpointing": False,
        "model_storage_dtype": str(resolved.layers[0].parameters().__next__().dtype),
        "compute_dtype": "torch.bfloat16 autocast",
        "optimizer": "torch.optim.AdamW; FP32 parameters and FP32 optimizer states",
    }


def _encode_prompt(
    tokenizer: Any, prompt: str, device: torch.device
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def _prepare_targets(
    tokenizer: Any,
    rows: list[dict[str, str]],
    fold_manifest: dict[str, Any],
) -> dict[int, int]:
    targets = {}
    for receipt in fold_manifest["rows"]:
        index = int(receipt["example_index"])
        if not receipt["token_compatible"]:
            continue
        token_id, _, reason = expected_token_id(
            tokenizer, rows[index]["prompt"], rows[index]["answer"]
        )
        if reason is not None or token_id != receipt["expected_token_id"]:
            raise ValueError(f"tokenizer target drift at row {index}: {reason}")
        targets[index] = int(token_id)
    if len(targets) != 72:
        raise ValueError(f"expected 72 compatible targets, got {len(targets)}")
    return targets


def _sample_indices(numel: int, count: int, *, device: torch.device) -> torch.Tensor:
    """Return exact, in-bounds evenly spaced indices without float rounding."""

    if numel < 1 or count < 1 or count > numel:
        raise ValueError("sample dimensions must satisfy 1 <= count <= numel")
    if count == 1:
        return torch.zeros(1, dtype=torch.int64, device=device)
    # Float32 cannot represent every integer once a tensor exceeds 2**24
    # elements. An earlier linspace implementation could therefore round the
    # final numel-1 endpoint up to numel for Qwen's largest matrices.
    numerator = torch.arange(count, dtype=torch.int64, device=device) * (numel - 1)
    return torch.div(numerator, count - 1, rounding_mode="floor")


def _parameter_probe(
    model: Any, values_per_parameter: int = 32
) -> tuple[list[str], torch.Tensor]:
    names = []
    values = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.numel() == 0:
            continue
        flat = parameter.detach().reshape(-1)
        count = min(values_per_parameter, flat.numel())
        indices = _sample_indices(flat.numel(), count, device=flat.device)
        names.append(name)
        values.append(flat[indices].float().cpu())
        if len(names) == 16:
            break
    if not values:
        raise ValueError("no trainable parameters available for update probe")
    return names, torch.cat(values)


def _memory() -> dict[str, float]:
    gib = 1024**3
    return {
        "allocated_gib": torch.cuda.memory_allocated() / gib,
        "reserved_gib": torch.cuda.memory_reserved() / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / gib,
    }


@torch.inference_mode()
def _evaluate(
    model: Any,
    tokenizer: Any,
    ablator: JSpaceAblator,
    rows: list[dict[str, str]],
    folds: dict[str, Any],
    fold: dict[str, Any],
    targets: dict[int, int],
    indices: list[int],
    phase: str,
) -> list[dict[str, Any]]:
    model.eval()
    split_by_index = {
        **{i: "train" for i in fold["train_indices"]},
        **{i: "validation" for i in fold["validation_indices"]},
        **{i: "test" for i in fold["test_indices"]},
    }
    records = []
    for index in indices:
        receipt = folds["rows"][index]
        record = {
            "phase": phase,
            "fold": fold["fold"],
            "split": split_by_index[index],
            "example_index": index,
            "name": rows[index]["name"],
            "category": rows[index]["category"],
            "baseline_clean_correct": receipt["baseline_clean_correct"],
            "baseline_lesion_broken": receipt["baseline_lesion_broken"],
            "token_compatible": index in targets,
            "expected_token_id": targets.get(index),
            "clean_predicted_token_id": None,
            "lesion_predicted_token_id": None,
            "clean_exact": None,
            "lesion_exact": None,
        }
        if index in targets:
            encoded = _encode_prompt(tokenizer, rows[index]["prompt"], model.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                plan = ablator.build_plan(
                    model, encoded["input_ids"], encoded["attention_mask"]
                )
                clean_logits = plan.clean_next_logits[0].float()
                clean_predicted = int(clean_logits.argmax())
                with ablator.apply(plan):
                    lesion_logits = (
                        model(**encoded, use_cache=False).logits[0, -1].float()
                    )
            lesion_predicted = int(lesion_logits.argmax())
            expected = targets[index]
            record.update(
                clean_predicted_token_id=clean_predicted,
                lesion_predicted_token_id=lesion_predicted,
                clean_predicted_token=tokenizer.decode([clean_predicted]),
                lesion_predicted_token=tokenizer.decode([lesion_predicted]),
                clean_exact=float(clean_predicted == expected),
                lesion_exact=float(lesion_predicted == expected),
                clean_gold_margin=float(clean_logits[expected] - clean_logits.max()),
                lesion_gold_margin=float(lesion_logits[expected] - lesion_logits.max()),
            )
        records.append(record)
    return records


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for split in ("train", "validation", "test", "all"):
        split_rows = (
            records if split == "all" else [r for r in records if r["split"] == split]
        )
        cohorts = {
            "compatible": [r for r in split_rows if r["token_compatible"]],
            "baseline_clean_correct": [
                r for r in split_rows if r["baseline_clean_correct"]
            ],
            "baseline_lesion_broken": [
                r for r in split_rows if r["baseline_lesion_broken"]
            ],
        }
        result[split] = {}
        for name, cohort in cohorts.items():
            result[split][name] = {
                "rows": len(cohort),
                "clean_accuracy": (
                    sum(float(r["clean_exact"]) for r in cohort) / len(cohort)
                    if cohort
                    else None
                ),
                "lesion_accuracy": (
                    sum(float(r["lesion_exact"]) for r in cohort) / len(cohort)
                    if cohort
                    else None
                ),
            }
    return result


def _evaluation_indices(
    fold: dict[str, Any], folds: dict[str, Any], limit: int
) -> list[int]:
    if limit == 0:
        return list(range(90))
    pools = [
        list(fold["eligible_train_indices"]),
        [
            i
            for i in fold["validation_indices"]
            if folds["rows"][i]["baseline_clean_correct"]
        ],
        [i for i in fold["test_indices"] if folds["rows"][i]["baseline_clean_correct"]],
    ]
    selected = []
    while len(selected) < limit and any(pools):
        for pool in pools:
            if pool and len(selected) < limit:
                selected.append(pool.pop(0))
    return sorted(set(selected))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if args.checkpoint != MODEL_ID or args.revision != MODEL_REVISION:
        raise ValueError("this pilot is pinned to the accepted Qwen3.5-4B checkpoint")
    if args.max_steps < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("training steps and accumulation must be positive")
    if args.mode == "preflight" and (args.max_steps > 2 or args.save_model):
        raise ValueError("preflight is limited to two steps and cannot save a model")
    if args.mode == "train" and args.eval_limit != 0:
        raise ValueError("full training must evaluate all 90 rows")
    args.output_dir.mkdir(parents=True)
    _write_json(args.output_dir / "status.json", {"status": "starting"})

    rows, folds, authorization = _load_protocol(args)
    fold = _fold(folds, args.fold)
    random.seed(args.seed + args.fold)
    torch.manual_seed(args.seed + args.fold)
    torch.cuda.manual_seed_all(args.seed + args.fold)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()

    model_config = ModelConfig(
        name_or_path=args.checkpoint,
        revision=args.revision,
        # Keep trainable master weights in FP32. BF16 compute is applied with
        # autocast below; loading storage weights as BF16 would make 1e-5 AdamW
        # updates disappear into BF16 rounding for many decoder parameters.
        dtype="float32",
        attn_implementation="sdpa",
        gradient_checkpointing=False,
        freeze_output_head=True,
    )
    model, tokenizer, resolved = load_model_and_tokenizer(model_config, device)
    runtime = _runtime_receipt(args, model, resolved)
    if runtime["final_norm_trainable"] or runtime["lm_head_trainable"]:
        raise ValueError("final norm and unembedding must be frozen")
    lens = LensMatrices.load(args.lens)
    if lens.n_prompts != 1000:
        raise ValueError(f"expected published n=1000 lens, got {lens.n_prompts}")
    ablator = JSpaceAblator(
        resolved,
        lens,
        _intervention(),
        device=device,
        dtype=torch.bfloat16,
    )
    targets = _prepare_targets(tokenizer, rows, folds)
    eligible = list(fold["eligible_train_indices"])
    if len(eligible) < 30:
        raise ValueError("fold has fewer than 30 baseline-clean-correct train rows")
    eval_indices = _evaluation_indices(fold, folds, args.eval_limit)
    initial_predictions = _evaluate(
        model, tokenizer, ablator, rows, folds, fold, targets, eval_indices, "initial"
    )
    before_names, before_values = _parameter_probe(model)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
    )
    optimizer.zero_grad(set_to_none=True)
    metrics = []
    order: list[int] = []
    cursor = 0
    model.train()
    for step in range(args.max_steps):
        losses = []
        selected_rows = []
        for _ in range(args.gradient_accumulation_steps):
            if cursor >= len(order):
                order = eligible.copy()
                random.Random(args.seed + args.fold * 10_000 + step).shuffle(order)
                cursor = 0
            index = order[cursor]
            cursor += 1
            selected_rows.append(index)
            encoded = _encode_prompt(tokenizer, rows[index]["prompt"], device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                plan = ablator.build_plan(
                    model, encoded["input_ids"], encoded["attention_mask"]
                )
                with ablator.apply(plan):
                    logits = model(**encoded, use_cache=False).logits[:, -1].float()
                    target = torch.tensor(
                        [targets[index]], dtype=torch.long, device=device
                    )
                    loss = F.cross_entropy(logits, target)
                    (loss / args.gradient_accumulation_steps).backward()
            if set(plan.selected_token_ids) != set(LESION_LAYERS):
                raise RuntimeError("online-current lesion did not fire at every layer")
            losses.append(float(loss.detach()))
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, args.max_grad_norm
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step + 1}")
        warmup_scale = min(1.0, (step + 1) / max(1, args.warmup_steps))
        learning_rate = args.learning_rate * warmup_scale
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        row = {
            "step": step + 1,
            "loss": sum(losses) / len(losses),
            "grad_norm_pre_clip": float(grad_norm),
            "learning_rate": learning_rate,
            "example_indices": json.dumps(selected_rows),
            **_memory(),
        }
        if not math.isfinite(row["loss"]):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        metrics.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

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
        raise RuntimeError(
            "no sampled FP32 parameter changed; the optimizer path is ineffective"
        )

    final_predictions = _evaluate(
        model, tokenizer, ablator, rows, folds, fold, targets, eval_indices, "terminal"
    )
    with (args.output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in initial_predictions + final_predictions:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    checkpoint = None
    if args.save_model:
        checkpoint_dir = args.output_dir / "checkpoint-terminal"
        model.save_pretrained(
            checkpoint_dir,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        tokenizer.save_pretrained(checkpoint_dir)
        checkpoint = {
            "path": str(checkpoint_dir),
            "optimizer_saved": False,
            "estimated_unique_bytes": sum(
                path.stat().st_size
                for path in checkpoint_dir.rglob("*")
                if path.is_file()
            ),
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "mode": args.mode,
        "fold": args.fold,
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.revision,
        "objective": "single_token_answer_only_teacher_forced_cross_entropy",
        "no_chat_template": True,
        "thinking_disabled": True,
        "rl": False,
        "lesion": {
            "selection_source": "online_current",
            "projection": "sequential",
            "layers": LESION_LAYERS,
            "k": 10,
            "exclude_output_top_k": 10,
            "strength": 1.0,
            "always_on_during_gradient_forwards": True,
        },
        "source_hashes": {
            "data": args.data_sha256,
            "folds": args.folds_sha256,
            "authorization": args.authorization_sha256,
            "lens": args.lens_sha256,
        },
        "authorization": authorization,
        "split": {
            "train_rows": len(fold["train_indices"]),
            "gradient_eligible_train_rows": len(eligible),
            "validation_rows": len(fold["validation_indices"]),
            "test_rows": len(fold["test_indices"]),
            "evaluated_rows": len(eval_indices),
        },
        "optimization": {
            "max_steps": args.max_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "max_grad_norm": args.max_grad_norm,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "seed": args.seed + args.fold,
        },
        "runtime": runtime,
        "memory": _memory(),
        "parameter_update_probe": update_probe,
        "initial": _summarize(initial_predictions),
        "terminal": _summarize(final_predictions),
        "saved_checkpoint": checkpoint,
        "claim_boundary": authorization["claim_boundary"],
    }
    _write_json(args.output_dir / "result.json", result)
    _write_json(args.output_dir / "status.json", {"status": "completed"})
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--folds-sha256", required=True)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--lens", required=True, type=Path)
    parser.add_argument("--lens-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=0,
        help="0 evaluates all rows; nonzero is allowed only for preflight",
    )
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
