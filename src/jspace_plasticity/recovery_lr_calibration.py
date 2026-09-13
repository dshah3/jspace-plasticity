"""Validation-only selection for the full-decoder learning-rate calibration."""

from __future__ import annotations

import json
from typing import Any

DECISION = "full_decoder_lr_calibration_authorized"
STEPS = (10, 25, 50, 100)
RATES = (1e-6, 3e-7)
NAMES = ("j-full-lr1e-6", "j-full-lr3e-7")


def validate_design(design: dict[str, Any]) -> None:
    if design["evaluation"].get("selection_split") != "val":
        raise ValueError("calibration selection must use validation only")
    if design["evaluation"].get("milestones") != list(STEPS):
        raise ValueError("calibration milestones drifted")
    if design["evaluation"].get("minimum_clean_accuracy") != 0.95:
        raise ValueError("clean retention floor must be 0.95")
    if design["training"].get("trainable_policy") != "all_current_decoder_parameters":
        raise ValueError("calibration must not introduce additional freezing")
    if design["evaluation"].get("checkpoint_retention") != "best_qualifying_per_arm":
        raise ValueError(
            "calibration must retain only best qualifying checkpoint per arm"
        )
    if design["training"].get("learning_rate") is not None:
        raise ValueError("calibration learning rates must be specified per arm")
    for arm, name, rate in zip(design["arms"], NAMES, RATES, strict=True):
        if (
            arm["name"] != name
            or arm["learning_rate"] != rate
            or arm["max_steps"] != 100
            or arm["training_condition"] != "jspace"
            or arm["data_fraction"] != 1.0
            or arm["seed"] != 20260825
            or arm["expected_train_rows"] != 189
            or arm["expected_train_clusters"] != 45
            or arm["save_checkpoint"]
        ):
            raise ValueError("calibration arm assignment drifted")


def validation_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    if len(rows) != 64 or len({r["source_id"] for r in rows}) != 64:
        raise ValueError("expected exactly 64 unique validation rows")
    for row in rows:
        if row["split"] != "val" or row["dataset"] != "synthetic_geo":
            raise ValueError("selection may not read screen or transfer predictions")
        for condition in ("clean", "jspace", "random"):
            if row[f"{condition}_exact"] != int(
                row[f"{condition}_predicted_token_id"] == row["expected_token_id"]
            ):
                raise ValueError("prediction correctness flag mismatch")
    return {
        "rows": len(rows),
        **{
            f"{c}_correct": sum(int(r[f"{c}_exact"]) for r in rows)
            for c in ("clean", "jspace", "random")
        },
    }


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        c
        for c in candidates
        if c["clean_correct"] / c["rows"] >= 0.95
        and c["jspace_correct"] > c["initial_jspace_correct"]
    ]
    # Fixed tie order: recovery, clean retention, fewer steps, smaller LR.
    return (
        max(
            eligible,
            key=lambda c: (
                c["jspace_correct"] - c["initial_jspace_correct"],
                c["clean_correct"],
                -c["step"],
                -c["learning_rate"],
            ),
        )
        if eligible
        else None
    )


def reduce_calibration(args: Any, design: dict[str, Any]) -> dict[str, Any]:
    from jspace_plasticity.synthetic_recovery_sft import (
        _require_sha,
        read_jsonl,
        sha256_path,
        write_json,
    )

    candidates, result_hashes = [], {}
    reference_initial = None
    for arm in design["arms"]:
        directory = args.output_dir / "arms" / f"arm-{arm['index']:02d}-{arm['name']}"
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text())
        if (
            result["status"] != "completed"
            or result["arm"] != arm
            or result["design_sha256"] != args.design_sha256
            or result["experiment_image"] != args.experiment_image
            or result["direction_convention"] != "effective_gain"
        ):
            raise ValueError("calibration result provenance mismatch")
        if result["runtime"]["trainable_parameters"] != 3570049536:
            raise ValueError("unexpected trainable parameter count")
        _require_sha(
            directory / "predictions.jsonl",
            result["predictions_sha256"],
            "initial predictions",
        )
        initial = read_jsonl(directory / "predictions.jsonl")
        baseline = validation_counts(initial)
        ordered = sorted(initial, key=lambda r: r["source_id"])
        if reference_initial is not None and ordered != reference_initial:
            raise ValueError("initial predictions differ across LR arms")
        reference_initial = ordered
        if [r["step"] for r in result["calibration_milestones"]] != list(STEPS):
            raise ValueError("missing or duplicated calibration milestones")
        for receipt in result["calibration_milestones"]:
            step = receipt["step"]
            stage = directory / "milestones" / f"step-{step:04d}"
            _require_sha(
                stage / "predictions.jsonl",
                receipt["predictions_sha256"],
                "milestone predictions",
            )
            counts = validation_counts(read_jsonl(stage / "predictions.jsonl"))
            if counts != receipt["counts"]:
                raise ValueError("milestone aggregate mismatch")
            if {r["source_id"] for r in read_jsonl(stage / "predictions.jsonl")} != {
                r["source_id"] for r in initial
            }:
                raise ValueError("milestone cohort drifted")
            if receipt["checkpoint"] is not None:
                _require_sha(
                    stage / "checkpoint-manifest.json",
                    receipt["checkpoint"]["manifest_sha256"],
                    "checkpoint manifest",
                )
            candidates.append(
                {
                    "arm": arm["name"],
                    "learning_rate": arm["learning_rate"],
                    "step": step,
                    **counts,
                    "initial_jspace_correct": baseline["jspace_correct"],
                    "checkpoint": receipt["checkpoint"]["path"]
                    if receipt["checkpoint"]
                    else None,
                    "checkpoint_retained": (stage / "checkpoint-terminal").is_dir(),
                }
            )
        result_hashes[str(result_path)] = sha256_path(result_path)
    selected = select_candidate(candidates)
    if selected is not None and not selected["checkpoint_retained"]:
        raise ValueError("selected checkpoint is missing")
    for arm in design["arms"]:
        arm_best = select_candidate([c for c in candidates if c["arm"] == arm["name"]])
        retained = [
            c
            for c in candidates
            if c["arm"] == arm["name"] and c["checkpoint_retained"]
        ]
        if len(retained) != int(arm_best is not None):
            raise ValueError("checkpoint retention count mismatch")
        if arm_best is not None and not arm_best["checkpoint_retained"]:
            raise ValueError("best checkpoint for an arm is missing")
    summary = {
        "status": "completed",
        "protocol": DECISION,
        "design_sha256": args.design_sha256,
        "experiment_image": args.experiment_image,
        "result_sha256": result_hashes,
        "candidates": candidates,
        "selected": selected,
        "selection_split": "val",
        "minimum_clean_accuracy": 0.95,
        "verdict": (
            (
                "validation candidate identified; "
                "matched controls and another seed required"
            )
            if selected
            else (
                "no checkpoint improved lesion accuracy "
                "while retaining 95% clean validation accuracy"
            )
        ),
        "claim_boundary": (
            "Exploratory calibration only. No screen/transfer selection "
            "or confirmation; no recovery guarantee."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
