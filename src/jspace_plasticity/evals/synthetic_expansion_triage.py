"""Triage entity-disjoint synthetic multihop rows for lesion-on SFT.

The frozen GeoNames corpus previously failed a population-level clean gate
because Qwen does not know every city fact and many answers are multi-token.
Rehabilitation, however, is explicitly limited to capabilities the clean model
already exhibits.  This evaluation therefore identifies exact-path,
single-token, clean-correct rows and measures J-space versus prompt-resampled
random retention on those rows.  It never trains or saves a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from jspace_plasticity.config import InterventionConfig, ModelConfig
from jspace_plasticity.evals.two_hop_probe import expected_token_id
from jspace_plasticity.intervention import JSpaceAblator
from jspace_plasticity.lens import LensMatrices
from jspace_plasticity.modeling import load_model_and_tokenizer
from jspace_plasticity.tasks.closedbook_geo import load_split

SCHEMA_VERSION = 1
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LESION_LAYERS = [16, 18, 19, 20, 21, 22]
SPLITS = ("train", "val", "screen")
CONDITIONS = ("jspace", "matched_random")
EXPECTED_WORLD_SIZE = len(SPLITS) * len(CONDITIONS)
MIN_CLEAN_CORRECT = {"train": 160, "val": 64, "screen": 64}
MIN_CLEAN_CORRECT_CLUSTERS = {"train": 30, "val": 10, "screen": 10}
MAX_EVAL_JSPACE_RETENTION = 0.75
MIN_EVAL_RANDOM_RETENTION = 0.80
MIN_EVAL_RANDOM_MINUS_JSPACE = 0.10
MAX_MCNEMAR_P = 0.05


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workload(rank_index: int, world_size: int = EXPECTED_WORLD_SIZE) -> dict[str, str]:
    if world_size != EXPECTED_WORLD_SIZE:
        raise ValueError(f"synthetic triage requires world_size={EXPECTED_WORLD_SIZE}")
    if not 0 <= rank_index < world_size:
        raise ValueError("rank index is outside world size")
    return {
        "split": SPLITS[rank_index // len(CONDITIONS)],
        "condition": CONDITIONS[rank_index % len(CONDITIONS)],
    }


def _require_preflight(
    path: Path,
    expected_sha256: str,
    expected_image: str,
) -> dict[str, Any]:
    if sha256_path(path) != expected_sha256:
        raise ValueError("SFT preflight result SHA-256 mismatch")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or result.get("mode") != "preflight":
        raise ValueError("SFT preflight did not complete")
    runtime = result.get("runtime", {})
    if runtime.get("experiment_image") != expected_image:
        raise ValueError("SFT preflight image mismatch")
    if runtime.get("model_storage_dtype") != "torch.float32":
        raise ValueError("SFT preflight did not use FP32 master parameters")
    if runtime.get("compute_dtype") != "torch.bfloat16 autocast":
        raise ValueError("SFT preflight did not use BF16 autocast")
    if result.get("parameter_update_probe", {}).get("changed_values", 0) < 1:
        raise ValueError("SFT preflight did not demonstrate a parameter update")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "experiment_image": expected_image,
        "peak_allocated_gib": result.get("memory", {}).get("peak_allocated_gib"),
    }


def intervention_config(condition: str, lens_path: Path) -> InterventionConfig:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown triage condition: {condition}")
    return InterventionConfig(
        enabled=True,
        lens_path=str(lens_path),
        layers=LESION_LAYERS,
        k=10,
        exclude_output_top_k=10,
        selection_source="online_current",
        projection="sequential",
        strength=1.0,
        control="none" if condition == "jspace" else "matched_random",
        control_resample="per_example",
        random_seed=20260825,
    )


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    compatible = [row for row in records if row["token_compatible"]]
    clean_correct = [row for row in compatible if row["clean_exact"] == 1.0]
    retained = sum(float(row["intervention_exact"]) for row in clean_correct)
    return {
        "rows": len(records),
        "compatible_rows": len(compatible),
        "compatible_frac": len(compatible) / len(records) if records else 0.0,
        "clean_correct_rows": len(clean_correct),
        "clean_correct_clusters": len({row["cluster_id"] for row in clean_correct}),
        "retained_correct_rows": int(retained),
        "retention": retained / len(clean_correct) if clean_correct else 0.0,
    }


def grouped_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[f"{row['relation']}/{row['template_id']}"].append(row)
    return {name: summarize(rows) for name, rows in sorted(grouped.items())}


@torch.inference_mode()
def evaluate_rows(
    model: Any,
    tokenizer: Any,
    ablator: JSpaceAblator,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for index, row in enumerate(rows):
        expected, continuation, reason = expected_token_id(
            tokenizer, row["prompt"], row["answer"]
        )
        record = {
            **row,
            "split_index": index,
            "answer_continuation": continuation,
            "token_compatible": expected is not None,
            "incompatible_reason": reason or "",
            "expected_token_id": expected,
            "clean_predicted_token_id": None,
            "intervention_predicted_token_id": None,
            "clean_exact": None,
            "intervention_exact": None,
        }
        if expected is not None:
            encoded = tokenizer(row["prompt"], return_tensors="pt")
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                plan = ablator.build_plan(
                    model, encoded["input_ids"], encoded["attention_mask"]
                )
                clean_logits = plan.clean_next_logits[0].float()
                with ablator.apply(plan):
                    intervention_logits = (
                        model(**encoded, use_cache=False).logits[0, -1].float()
                    )
            if set(plan.selected_token_ids) != set(LESION_LAYERS):
                raise RuntimeError("online-current lesion did not fire at every layer")
            clean_predicted = int(clean_logits.argmax())
            intervention_predicted = int(intervention_logits.argmax())
            record.update(
                clean_predicted_token_id=clean_predicted,
                intervention_predicted_token_id=intervention_predicted,
                clean_predicted_token=tokenizer.decode([clean_predicted]),
                intervention_predicted_token=tokenizer.decode([intervention_predicted]),
                clean_exact=float(clean_predicted == expected),
                intervention_exact=float(intervention_predicted == expected),
            )
        records.append(record)
    return records


def run_rank(args: argparse.Namespace) -> dict[str, Any]:
    assigned = workload(args.rank_index, args.world_size)
    rank_dir = args.output_dir / "ranks" / f"rank-{args.rank_index:02d}"
    if rank_dir.exists():
        raise FileExistsError(f"refusing to overwrite triage rank: {rank_dir}")
    preflight = _require_preflight(
        args.preflight_result, args.preflight_sha256, args.preflight_image
    )
    if os.environ.get("EXPERIMENT_IMAGE") != args.experiment_image:
        raise ValueError("runtime experiment image does not match the CLI receipt")
    rows, manifest = load_split(
        args.data, expected_sha256=args.data_sha256, split=assigned["split"]
    )
    model_config = ModelConfig(
        name_or_path=args.checkpoint,
        revision=args.revision,
        dtype="float32",
        attn_implementation="sdpa",
        gradient_checkpointing=False,
        freeze_output_head=True,
    )
    model, tokenizer, resolved = load_model_and_tokenizer(
        model_config, torch.device("cuda")
    )
    model.eval()
    if sha256_path(args.lens) != args.lens_sha256:
        raise ValueError("published lens SHA-256 mismatch")
    lens = LensMatrices.load(args.lens)
    if lens.n_prompts != 1000:
        raise ValueError(f"expected an n=1000 published lens, got {lens.n_prompts}")
    ablator = JSpaceAblator(
        resolved,
        lens,
        intervention_config(assigned["condition"], args.lens),
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    records = evaluate_rows(model, tokenizer, ablator, rows)
    rank_dir.mkdir(parents=True)
    predictions_path = rank_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "protocol": "synthetic_expansion_clean_correct_lesion_triage",
        "rank_index": args.rank_index,
        "world_size": args.world_size,
        **assigned,
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.revision,
        "experiment_image": args.experiment_image,
        "preflight": preflight,
        "data": {
            "path": str(args.data),
            "sha256": args.data_sha256,
            "split_rows": manifest["split_rows"],
            "entity_splits": manifest["entity_splits"],
        },
        "lens": {
            "path": str(args.lens),
            "sha256": args.lens_sha256,
            "n_prompts": lens.n_prompts,
        },
        "intervention": {
            "layers": LESION_LAYERS,
            "k": 10,
            "exclude_output_top_k": 10,
            "selection_source": "online_current",
            "projection": "sequential",
            "condition": assigned["condition"],
        },
        "metrics": summarize(records),
        "cells": grouped_metrics(records),
        "predictions_sha256": sha256_path(predictions_path),
        "training_authorized": False,
    }
    (rank_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def paired_metrics(
    jspace: list[dict[str, Any]], random: list[dict[str, Any]]
) -> dict[str, Any]:
    random_by_id = {row["source_id"]: row for row in random}
    pairs = []
    for row in jspace:
        other = random_by_id[row["source_id"]]
        clean_fields = (
            "token_compatible",
            "expected_token_id",
            "clean_predicted_token_id",
            "clean_exact",
        )
        if any(row[field] != other[field] for field in clean_fields):
            raise ValueError(f"clean-path drift for {row['source_id']}")
        if row["token_compatible"] and row["clean_exact"] == 1.0:
            pairs.append(
                (int(row["intervention_exact"]), int(other["intervention_exact"]))
            )
    if not pairs:
        raise ValueError("paired triage has no clean-correct rows")
    both_correct = sum(j == 1 and r == 1 for j, r in pairs)
    j_only = sum(j == 1 and r == 0 for j, r in pairs)
    random_only = sum(j == 0 and r == 1 for j, r in pairs)
    both_wrong = sum(j == 0 and r == 0 for j, r in pairs)
    count = len(pairs)
    discordant = j_only + random_only
    if discordant:
        lower = (
            sum(
                math.comb(discordant, value)
                for value in range(min(j_only, random_only) + 1)
            )
            / 2**discordant
        )
        p_value = min(1.0, 2 * lower)
    else:
        p_value = 1.0
    return {
        "clean_correct_rows": count,
        "both_correct": both_correct,
        "jspace_only_correct": j_only,
        "random_only_correct": random_only,
        "both_wrong": both_wrong,
        "jspace_retention": (both_correct + j_only) / count,
        "random_retention": (both_correct + random_only) / count,
        "random_minus_jspace": (random_only - j_only) / count,
        "mcnemar_exact_two_sided_p": p_value,
    }


def reduce_results(args: argparse.Namespace) -> dict[str, Any]:
    expected = [workload(rank, args.world_size) for rank in range(args.world_size)]
    results: dict[tuple[str, str], dict[str, Any]] = {}
    predictions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    hashes = {}
    for rank, assigned in enumerate(expected):
        rank_dir = args.output_dir / "ranks" / f"rank-{rank:02d}"
        result_path = rank_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        key = (assigned["split"], assigned["condition"])
        if result.get("status") != "completed" or any(
            result.get(field) != value for field, value in assigned.items()
        ):
            raise ValueError(f"rank {rank} result does not match its workload")
        if result.get("experiment_image") != args.experiment_image:
            raise ValueError(f"rank {rank} used another experiment image")
        results[key] = result
        predictions[key] = _read_jsonl(rank_dir / "predictions.jsonl")
        hashes[str(result_path)] = sha256_path(result_path)

    paired = {
        split: paired_metrics(
            predictions[(split, "jspace")],
            predictions[(split, "matched_random")],
        )
        for split in SPLITS
    }
    evaluation_j = predictions[("val", "jspace")] + predictions[("screen", "jspace")]
    evaluation_r = (
        predictions[("val", "matched_random")]
        + predictions[("screen", "matched_random")]
    )
    paired_evaluation = paired_metrics(evaluation_j, evaluation_r)
    clauses = {
        **{
            f"{split}_clean_correct_at_least_{MIN_CLEAN_CORRECT[split]}": paired[split][
                "clean_correct_rows"
            ]
            >= MIN_CLEAN_CORRECT[split]
            for split in SPLITS
        },
        **{
            f"{split}_clusters_at_least_{MIN_CLEAN_CORRECT_CLUSTERS[split]}": results[
                (split, "jspace")
            ]["metrics"]["clean_correct_clusters"]
            >= MIN_CLEAN_CORRECT_CLUSTERS[split]
            for split in SPLITS
        },
        "evaluation_jspace_retention_at_most_0_75": paired_evaluation[
            "jspace_retention"
        ]
        <= MAX_EVAL_JSPACE_RETENTION,
        "evaluation_random_retention_at_least_0_80": paired_evaluation[
            "random_retention"
        ]
        >= MIN_EVAL_RANDOM_RETENTION,
        "evaluation_random_minus_jspace_at_least_0_10": paired_evaluation[
            "random_minus_jspace"
        ]
        >= MIN_EVAL_RANDOM_MINUS_JSPACE,
        "evaluation_mcnemar_p_at_most_0_05": paired_evaluation[
            "mcnemar_exact_two_sided_p"
        ]
        <= MAX_MCNEMAR_P,
    }
    eligible_train = [
        {
            "source_id": row["source_id"],
            "split_index": row["split_index"],
            "cluster_id": row["cluster_id"],
            "relation": row["relation"],
            "template_id": row["template_id"],
            "expected_token_id": row["expected_token_id"],
            "baseline_jspace_correct": bool(row["intervention_exact"]),
        }
        for row in predictions[("train", "jspace")]
        if row["token_compatible"] and row["clean_exact"] == 1.0
    ]
    eligibility_path = args.output_dir / "eligible_train_rows.jsonl"
    eligibility_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in eligible_train),
        encoding="utf-8",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "protocol": "synthetic_expansion_clean_correct_lesion_triage",
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.revision,
        "experiment_image": args.experiment_image,
        "data_sha256": args.data_sha256,
        "lens_sha256": args.lens_sha256,
        "preflight_sha256": args.preflight_sha256,
        "rank_result_sha256": hashes,
        "paired_by_split": paired,
        "paired_evaluation": paired_evaluation,
        "clauses": clauses,
        "triage_passed": all(clauses.values()),
        "eligible_train_rows": len(eligible_train),
        "eligible_train_rows_sha256": sha256_path(eligibility_path),
        "training_authorized": False,
        "next_step": (
            "freeze data-scale and duration sweep"
            if all(clauses.values())
            else "do not train on this synthetic distribution"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--lens", required=True, type=Path)
    parser.add_argument("--lens-sha256", required=True)
    parser.add_argument("--preflight-result", required=True, type=Path)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--preflight-image", required=True)
    parser.add_argument("--experiment-image", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rank-index", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=EXPECTED_WORLD_SIZE)
    parser.add_argument("--reduce-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint != MODEL_ID or args.revision != MODEL_REVISION:
        parser.error("synthetic triage is pinned to the Qwen3.5-4B checkpoint")
    if args.world_size != EXPECTED_WORLD_SIZE:
        parser.error(f"synthetic triage requires world-size {EXPECTED_WORLD_SIZE}")
    if args.reduce_only:
        reduce_results(args)
    else:
        run_rank(args)


if __name__ == "__main__":
    main()
