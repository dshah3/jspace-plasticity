"""Evaluate whether Qwen3.5-4B recovery exploited the top-k lesion boundary.

This module never trains.  It crosses the clean, primary-recovered, and
high-dose-recovered checkpoints with their own exact 500-prompt fresh lenses at
k in {10, 12, 16, 20, 32, 50}.  Every condition also runs a prompt-resampled,
norm-matched random control.  An audit-only top-50 ranking is captured while
only the requested first k directions are projected out, so the k=10 condition
directly reveals directions immediately below the original cutoff.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from jspace_plasticity.config import ModelConfig
from jspace_plasticity.intervention import AblationPlan, JSpaceAblator
from jspace_plasticity.lens import LensMatrices
from jspace_plasticity.lens.fit_exact_dp import validate_model_manifest
from jspace_plasticity.modeling import load_model_and_tokenizer
from jspace_plasticity.synthetic_recovery_sft import (
    LESION_LAYERS,
    MODEL_REVISION,
    _encode,
    _normalized_synthetic_rows,
    _transfer_rows,
    intervention_config,
    load_triage,
    paired_difference,
    read_jsonl,
    sha256_path,
    write_json,
)
from jspace_plasticity.tasks.closedbook_geo import load_split

SCHEMA_VERSION = 1
EXPECTED_WORLD_SIZE = 8
EXPECTED_K_VALUES = (10, 12, 16, 20, 32, 50)
EXPECTED_MODELS = ("base", "primary", "high_dose")
PRIOR_CONDITION_BY_MODEL = {
    "base": "base-own-fresh",
    "primary": "primary-own-fresh",
    "high_dose": "high-own-fresh",
}


def _require_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256_path(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return observed


def _expected_conditions() -> list[dict[str, Any]]:
    rows = []
    index = 0
    labels = {"base": "base", "primary": "primary", "high_dose": "high"}
    lenses = {
        "base": "base_fresh",
        "primary": "primary_fresh",
        "high_dose": "high_fresh",
    }
    for k in EXPECTED_K_VALUES:
        for model in EXPECTED_MODELS:
            rows.append(
                {
                    "index": index,
                    "name": f"{labels[model]}-k{k}",
                    "model": model,
                    "lens": lenses[model],
                    "k": k,
                }
            )
            index += 1
    return rows


def load_design(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require_sha(path, expected_sha256, "rank-cutoff design")
    design = json.loads(path.read_text(encoding="utf-8"))
    if design.get("decision") != "fresh_lens_rank_cutoff_diagnostic_authorized":
        raise ValueError("design does not authorize the rank-cutoff diagnostic")
    if design.get("evaluation_conditions") != _expected_conditions():
        raise ValueError("rank-cutoff condition matrix drifted")
    intervention = design.get("intervention", {})
    if intervention != {
        "selection_source": "online_current",
        "projection": "sequential",
        "layers": LESION_LAYERS,
        "k_values": list(EXPECTED_K_VALUES),
        "audit_top_k": 50,
        "audit_ordering": (
            "exact projected top-k prefix, followed by the highest-scoring "
            "remaining eligible directions"
        ),
        "exclude_output_top_k": 10,
        "strength": 1.0,
        "matched_random_resampling": "per_example",
    }:
        raise ValueError("rank-cutoff intervention drifted")
    if design.get("evaluation", {}).get("training") is not False:
        raise ValueError("rank-cutoff diagnostic must not train")
    return design


def condition_for(design: dict[str, Any], index: int) -> dict[str, Any]:
    conditions = design["evaluation_conditions"]
    if not 0 <= index < len(conditions):
        raise ValueError("condition index is outside the frozen design")
    condition = conditions[index]
    if condition["index"] != index:
        raise ValueError("condition ordering drifted")
    return condition


def condition_indices_for_worker(
    *, worker_index: int, world_size: int, condition_count: int
) -> list[int]:
    if world_size != EXPECTED_WORLD_SIZE:
        raise ValueError(f"world_size must be {EXPECTED_WORLD_SIZE}")
    if not 0 <= worker_index < world_size:
        raise ValueError("worker index is outside world size")
    return list(range(worker_index, condition_count, world_size))


def _model_receipt(design: dict[str, Any], key: str) -> dict[str, Any]:
    receipt = design["models"][key]
    if receipt["revision"] != MODEL_REVISION:
        raise ValueError(f"model revision drift for {key}")
    if key != "base":
        validate_model_manifest(
            Path(receipt["manifest_path"]),
            expected_sha256=receipt["manifest_sha256"],
            model_dir=Path(receipt["path"]),
        )
    return receipt


def _lens_receipt(design: dict[str, Any], key: str) -> dict[str, Any]:
    receipt = design["fresh_lenses"][key]
    path = Path(receipt["path"])
    _require_sha(path, receipt["sha256"], f"{key} lens")
    return {
        "key": key,
        "path": str(path),
        "sha256": receipt["sha256"],
        "n_prompts": int(design["fresh_lenses"]["stage_prompts"]),
    }


def _intervention_config(condition: str, lens_path: Path, k: int) -> Any:
    return replace(intervention_config(condition, lens_path), k=k)


def _semantic_token_candidates(tokenizer: Any, text: str) -> list[dict[str, Any]]:
    candidates: dict[int, dict[str, Any]] = {}
    for rendered in (text, f" {text}"):
        for token_id in tokenizer.encode(rendered, add_special_tokens=False):
            token_id = int(token_id)
            candidates.setdefault(
                token_id,
                {
                    "token_id": token_id,
                    "token": tokenizer.convert_ids_to_tokens(token_id),
                    "decoded": tokenizer.decode([token_id]),
                },
            )
    if not candidates:
        raise ValueError(f"intermediate has no token candidates: {text!r}")
    return list(candidates.values())


def _rank_audit(
    plan: AblationPlan,
    tokenizer: Any,
    *,
    intermediate: str,
    expected_token_id: int,
    intervention_k: int,
) -> dict[str, Any]:
    if plan.ranked_token_ids is None or plan.audit_top_k is None:
        raise RuntimeError("rank audit was not requested")
    candidate_rows = _semantic_token_candidates(tokenizer, intermediate)
    candidate_ids = {row["token_id"] for row in candidate_rows}
    blocked = [int(value) for value in plan.blocked_token_ids[0, -1].tolist()]
    layers: dict[str, Any] = {}
    best_ranks = []
    any_top10 = False
    any_boundary = False
    any_selected = False
    for layer in LESION_LAYERS:
        ranked = [int(value) for value in plan.ranked_token_ids[layer][0, -1].tolist()]
        selected = [
            int(value) for value in plan.selected_token_ids[layer][0, -1].tolist()
        ]
        if selected != ranked[:intervention_k]:
            raise RuntimeError(
                f"audit ranking changed projected directions at layer {layer}"
            )
        ranks = {
            str(token_id): ranked.index(token_id) + 1
            for token_id in candidate_ids
            if token_id in ranked
        }
        best_rank = min(ranks.values()) if ranks else None
        if best_rank is not None:
            best_ranks.append(best_rank)
            any_top10 = any_top10 or best_rank <= 10
            any_boundary = any_boundary or 11 <= best_rank <= 12
            any_selected = any_selected or best_rank <= intervention_k
        layers[str(layer)] = {
            "ranked_token_ids": ranked,
            "intermediate_candidate_ranks": ranks,
            "intermediate_best_rank": best_rank,
            "intermediate_selected": best_rank is not None
            and best_rank <= intervention_k,
        }
    return {
        "intermediate": intermediate,
        "intermediate_token_candidates": candidate_rows,
        "intermediate_output_blocked": bool(candidate_ids.intersection(blocked)),
        "intermediate_output_blocked_token_ids": sorted(
            candidate_ids.intersection(blocked)
        ),
        "answer_output_blocked": expected_token_id in blocked,
        "answer_output_blocked_rank": (
            blocked.index(expected_token_id) + 1
            if expected_token_id in blocked
            else None
        ),
        "intermediate_best_rank_any_layer": min(best_ranks) if best_ranks else None,
        "intermediate_seen_top10_any_layer": any_top10,
        "intermediate_seen_rank11_12_any_layer": any_boundary,
        "intermediate_rank11_12_without_top10": any_boundary and not any_top10,
        "intermediate_selected_any_layer": any_selected,
        "layers": layers,
    }


def _load_rows(
    design: dict[str, Any], tokenizer: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence = design["evaluation_evidence"]
    cohorts, _, _ = load_triage(
        Path(evidence["triage_dir"]),
        summary_sha256=evidence["triage_summary_sha256"],
        eligible_sha256=evidence["eligible_train_rows_sha256"],
    )
    split_rows = {
        split: load_split(
            Path(evidence["synthetic_data_path"]),
            expected_sha256=evidence["synthetic_data_sha256"],
            split=split,
        )[0]
        for split in ("val", "screen")
    }
    normalized = _normalized_synthetic_rows(tokenizer, split_rows, cohorts)
    source_by_id = {
        row["source_id"]: row for rows in split_rows.values() for row in rows
    }
    synthetic = normalized["val"] + normalized["screen"]
    for row in synthetic:
        row["intermediate"] = source_by_id[row["source_id"]]["intermediate"]

    transfer = _transfer_rows(
        tokenizer,
        Path(evidence["anthropic_data_path"]),
        evidence["anthropic_data_sha256"],
        Path(evidence["anthropic_eligibility_path"]),
        evidence["anthropic_eligibility_sha256"],
    )
    raw_transfer = json.loads(
        Path(evidence["anthropic_data_path"]).read_text(encoding="utf-8")
    )["items"]
    intermediate_by_name = {row["name"]: row["intermediate"] for row in raw_transfer}
    for row in transfer:
        row["intermediate"] = intermediate_by_name[row["cluster_id"]]
    if len(synthetic) != 129 or len(transfer) != 49:
        raise ValueError("frozen rank-cutoff cohort row count drifted")
    return synthetic, transfer


@torch.inference_mode()
def evaluate_condition(
    model: Any,
    tokenizer: Any,
    jspace: JSpaceAblator,
    random_control: JSpaceAblator,
    rows: list[dict[str, Any]],
    *,
    k: int,
    audit_top_k: int,
) -> list[dict[str, Any]]:
    model.eval()
    records = []
    for row in rows:
        encoded = _encode(tokenizer, row["prompt"], model.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            j_plan = jspace.build_plan(
                model,
                encoded["input_ids"],
                encoded["attention_mask"],
                audit_top_k=audit_top_k,
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
        if int(clean_logits.argmax()) != int(random_clean_logits.argmax()):
            raise RuntimeError("clean argmax drifted across intervention plans")
        target = int(row["expected_token_id"])
        clean_predicted = int(clean_logits.argmax())
        j_predicted = int(j_logits.argmax())
        random_predicted = int(random_logits.argmax())
        records.append(
            {
                "dataset": row["dataset"],
                "split": row["split"],
                "source_id": row["source_id"],
                "cluster_id": row["cluster_id"],
                "relation": row["relation"],
                "expected_token_id": target,
                "clean_predicted_token_id": clean_predicted,
                "jspace_predicted_token_id": j_predicted,
                "random_predicted_token_id": random_predicted,
                "clean_exact": int(clean_predicted == target),
                "jspace_exact": int(j_predicted == target),
                "random_exact": int(random_predicted == target),
                "clean_gold_margin": float(clean_logits[target] - clean_logits.max()),
                "jspace_gold_margin": float(j_logits[target] - j_logits.max()),
                "random_gold_margin": float(
                    random_logits[target] - random_logits.max()
                ),
                "rank_audit": _rank_audit(
                    j_plan,
                    tokenizer,
                    intermediate=row["intermediate"],
                    expected_token_id=target,
                    intervention_k=k,
                ),
            }
        )
    return records


def _accuracy(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        raise ValueError("cannot score an empty row group")
    return sum(int(row[field]) for row in rows) / len(rows)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row["dataset"]].append(row)
    result = {}
    for name, rows in sorted(groups.items()):
        audits = [row["rank_audit"] for row in rows]
        result[name] = {
            "rows": len(rows),
            "clean_accuracy": _accuracy(rows, "clean_exact"),
            "jspace_accuracy": _accuracy(rows, "jspace_exact"),
            "random_accuracy": _accuracy(rows, "random_exact"),
            "random_minus_jspace": _accuracy(rows, "random_exact")
            - _accuracy(rows, "jspace_exact"),
            "answer_output_blocked_frac": sum(
                int(row["answer_output_blocked"]) for row in audits
            )
            / len(audits),
            "intermediate_output_blocked_frac": sum(
                int(row["intermediate_output_blocked"]) for row in audits
            )
            / len(audits),
            "intermediate_seen_top10_any_layer_frac": sum(
                int(row["intermediate_seen_top10_any_layer"]) for row in audits
            )
            / len(audits),
            "intermediate_seen_rank11_12_any_layer_frac": sum(
                int(row["intermediate_seen_rank11_12_any_layer"]) for row in audits
            )
            / len(audits),
            "intermediate_rank11_12_without_top10_frac": sum(
                int(row["intermediate_rank11_12_without_top10"]) for row in audits
            )
            / len(audits),
        }
    return result


def _verify_k10_parity(
    records: list[dict[str, Any]], design: dict[str, Any], model_key: str
) -> dict[str, Any]:
    source = design["source_result"]
    summary_path = Path(source["path"])
    _require_sha(summary_path, source["sha256"], "source anti-stale-lens summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prior_name = PRIOR_CONDITION_BY_MODEL[model_key]
    matches = [
        (Path(path), digest)
        for path, digest in summary["result_sha256"].items()
        if Path(path).parent.name.endswith(f"-{prior_name}")
    ]
    if len(matches) != 1:
        raise ValueError(f"could not resolve prior condition {prior_name}")
    result_path, result_sha = matches[0]
    _require_sha(result_path, result_sha, f"prior {prior_name} result")
    prior_result = json.loads(result_path.read_text(encoding="utf-8"))
    predictions_path = result_path.parent / "predictions.jsonl"
    _require_sha(
        predictions_path,
        prior_result["predictions_sha256"],
        f"prior {prior_name} predictions",
    )
    prior = {
        (row["dataset"], row["source_id"]): row for row in read_jsonl(predictions_path)
    }
    fields = (
        "clean_predicted_token_id",
        "jspace_predicted_token_id",
        "random_predicted_token_id",
    )
    mismatches = []
    compared = 0
    for row in records:
        expected = prior[(row["dataset"], row["source_id"])]
        for field in fields:
            compared += 1
            if int(row[field]) != int(expected[field]):
                mismatches.append(
                    {
                        "dataset": row["dataset"],
                        "source_id": row["source_id"],
                        "field": field,
                        "expected": int(expected[field]),
                        "observed": int(row[field]),
                    }
                )
    return {
        "prior_condition": prior_name,
        "compared_values": compared,
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def run_condition(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite condition: {args.output_dir}")
    design = load_design(args.design, args.design_sha256)
    condition = condition_for(design, args.condition_index)
    if os.environ.get("EXPERIMENT_IMAGE") != args.experiment_image:
        raise ValueError("runtime image does not match immutable CLI image")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    model_receipt = _model_receipt(design, condition["model"])
    model, tokenizer, resolved = load_model_and_tokenizer(
        ModelConfig(
            name_or_path=model_receipt["path"],
            revision=model_receipt["revision"],
            dtype="float32",
            attn_implementation="sdpa",
            gradient_checkpointing=False,
            freeze_output_head=True,
        ),
        device,
    )
    synthetic, transfer = _load_rows(design, tokenizer)
    lens_receipt = _lens_receipt(design, condition["lens"])
    lens = LensMatrices.load(lens_receipt["path"])
    if lens.n_prompts != lens_receipt["n_prompts"]:
        raise ValueError("loaded lens prompt count disagrees with receipt")
    lens_path = Path(lens_receipt["path"])
    jspace = JSpaceAblator(
        resolved,
        lens,
        _intervention_config("jspace", lens_path, condition["k"]),
        device=device,
        dtype=torch.bfloat16,
    )
    random_control = JSpaceAblator(
        resolved,
        lens,
        _intervention_config("matched_random", lens_path, condition["k"]),
        device=device,
        dtype=torch.bfloat16,
    )
    records = evaluate_condition(
        model,
        tokenizer,
        jspace,
        random_control,
        synthetic + transfer,
        k=condition["k"],
        audit_top_k=int(design["intervention"]["audit_top_k"]),
    )
    parity = (
        _verify_k10_parity(records, design, condition["model"])
        if condition["k"] == 10
        else None
    )
    if parity is not None and not parity["passed"]:
        raise RuntimeError(f"k=10 audit-path parity failed: {parity}")
    args.output_dir.mkdir(parents=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "protocol": "q35_recovery_rank_cutoff_sweep",
        "training": False,
        "condition": condition,
        "experiment_image": args.experiment_image,
        "design_sha256": args.design_sha256,
        "model_receipt": model_receipt,
        "lens_receipt": lens_receipt,
        "evaluation": summarize(records),
        "k10_prior_parity": parity,
        "predictions_sha256": sha256_path(predictions_path),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def classify_rank_11_12(
    metrics: dict[str, dict[str, Any]], gate: dict[str, float]
) -> tuple[str, dict[str, bool]]:
    primary10 = metrics["primary-k10"]["synthetic_geo"]
    primary12 = metrics["primary-k12"]["synthetic_geo"]
    high10 = metrics["high-k10"]["synthetic_geo"]
    high12 = metrics["high-k12"]["synthetic_geo"]
    clauses = {
        "primary_k12_accuracy_at_least_0_70": primary12["jspace_accuracy"]
        >= gate["minimum_primary_k12_accuracy"],
        "primary_k10_to_k12_drop_at_most_0_10": (
            primary10["jspace_accuracy"] - primary12["jspace_accuracy"]
            <= gate["maximum_primary_k10_to_k12_drop"]
        ),
        "primary_k12_random_at_least_0_75": primary12["random_accuracy"]
        >= gate["minimum_primary_k12_random_accuracy"],
        "high_k12_accuracy_at_least_0_70": high12["jspace_accuracy"]
        >= gate["minimum_high_k12_accuracy"],
        "high_k10_to_k12_drop_at_most_0_10": (
            high10["jspace_accuracy"] - high12["jspace_accuracy"]
            <= gate["maximum_high_k10_to_k12_drop"]
        ),
    }
    if all(clauses.values()):
        verdict = "simple_rank_11_12_evasion_not_supported"
    elif (
        primary10["jspace_accuracy"] - primary12["jspace_accuracy"] >= 0.20
        and primary12["random_accuracy"] >= gate["minimum_primary_k12_random_accuracy"]
    ):
        verdict = "simple_rank_11_12_evasion_supported"
    else:
        verdict = "mixed_rank_11_12_diagnostic"
    return verdict, clauses


def reduce_results(args: argparse.Namespace) -> dict[str, Any]:
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("refusing to overwrite rank-cutoff summary")
    design = load_design(args.design, args.design_sha256)
    results = {}
    predictions = {}
    result_hashes = {}
    for condition in design["evaluation_conditions"]:
        condition_dir = (
            args.output_dir
            / "conditions"
            / f"condition-{condition['index']:02d}-{condition['name']}"
        )
        result_path = condition_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed" or result.get("condition") != condition:
            raise ValueError(f"condition result drift: {condition['name']}")
        if result.get("experiment_image") != args.experiment_image:
            raise ValueError(f"condition image drift: {condition['name']}")
        predictions_path = condition_dir / "predictions.jsonl"
        _require_sha(
            predictions_path,
            result["predictions_sha256"],
            f"{condition['name']} predictions",
        )
        results[condition["name"]] = result
        predictions[condition["name"]] = read_jsonl(predictions_path)
        result_hashes[str(result_path)] = sha256_path(result_path)
    k10_parity = {
        name: result["k10_prior_parity"]
        for name, result in results.items()
        if result["k10_prior_parity"] is not None
    }
    if len(k10_parity) != 3 or not all(row["passed"] for row in k10_parity.values()):
        raise ValueError("k=10 audit path did not reproduce the frozen source result")
    metrics = {name: result["evaluation"] for name, result in results.items()}
    paired = {}
    for model_label in ("base", "primary", "high"):
        k10_rows = [
            row
            for row in predictions[f"{model_label}-k10"]
            if row["dataset"] == "synthetic_geo"
        ]
        for k in EXPECTED_K_VALUES[1:]:
            other = [
                row
                for row in predictions[f"{model_label}-k{k}"]
                if row["dataset"] == "synthetic_geo"
            ]
            paired[f"{model_label}_k10_vs_k{k}_jspace"] = paired_difference(
                k10_rows, other, field="jspace_exact"
            )
    verdict, clauses = classify_rank_11_12(metrics, design["rank_11_12_gate"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "protocol": "q35_recovery_rank_cutoff_sweep",
        "training": False,
        "experiment_image": args.experiment_image,
        "design_sha256": args.design_sha256,
        "result_sha256": result_hashes,
        "k10_prior_parity": k10_parity,
        "condition_metrics": metrics,
        "paired_k_sweep": paired,
        "rank_11_12_gate_clauses": clauses,
        "verdict": verdict,
        "interpretation": design["interpretation"][
            "not_supported"
            if verdict == "simple_rank_11_12_evasion_not_supported"
            else "supported"
            if verdict == "simple_rank_11_12_evasion_supported"
            else "mixed"
        ],
        "claim_boundary": design["claim_boundary"],
        "further_training_authorized": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--design-sha256", required=True)
    parser.add_argument("--experiment-image", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition-index", type=int, default=0)
    parser.add_argument("--reduce-only", action="store_true")
    args = parser.parse_args()
    if args.reduce_only:
        reduce_results(args)
    else:
        run_condition(args)


if __name__ == "__main__":
    main()
