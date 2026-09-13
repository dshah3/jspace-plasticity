"""Preregistered clean capability gate for geography composition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from jspace_plasticity.evals.two_hop_probe import (
    QWEN3_8B_REVISION,
    evaluate,
)

# Updated only after the mechanically uniform trailing-space boundary correction.
GEOGRAPHY_DATA_SHA256 = (
    "84452211a8e50e10f2b2d427aa6fa47df271956e90b356bd63e498b215488731"
)
EXPECTED_RELATIONS = frozenset({"capital", "country", "currency", "region"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_geography(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _sha256(path)
    if digest != GEOGRAPHY_DATA_SHA256 or manifest["sha256"] != digest:
        raise ValueError("geography confirmation dataset SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["items"]
    if (
        payload["split"] != "clean_confirmation_eval_only"
        or len(payload["entities"]) != 32
        or len(rows) != 128
        or set(payload["relations"]) != EXPECTED_RELATIONS
    ):
        raise ValueError("unexpected geography confirmation dataset contract")
    relation_counts: dict[str, int] = defaultdict(int)
    entity_relations: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["prompt"] != row["prompt"].rstrip():
            raise ValueError("geography prompts must not end in whitespace")
        relation_counts[row["relation"]] += 1
        entity_relations[row["iso"]].add(row["relation"])
    if relation_counts != {relation: 32 for relation in EXPECTED_RELATIONS}:
        raise ValueError(f"unexpected relation counts: {relation_counts}")
    if len(entity_relations) != 32 or any(
        relations != EXPECTED_RELATIONS for relations in entity_relations.values()
    ):
        raise ValueError("each geography entity must have all four relations")
    return rows, manifest


def relation_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["relation"]].append(row)
    result = {}
    for relation in sorted(EXPECTED_RELATIONS):
        rows = grouped[relation]
        compatible = [row for row in rows if row["token_compatible"]]
        result[relation] = {
            "rows": float(len(rows)),
            "compatible_rows": float(len(compatible)),
            "coverage": len(compatible) / len(rows) if rows else 0.0,
            "accuracy": (
                sum(float(row["exact"]) for row in compatible) / len(compatible)
                if compatible
                else 0.0
            ),
        }
    return result


def capability_verdict(
    records: list[dict[str, Any]],
    *,
    min_overall_coverage: float = 0.8,
    min_relation_coverage: float = 0.75,
    min_overall_accuracy: float = 0.8,
    min_macro_accuracy: float = 0.8,
    min_relation_accuracy: float = 0.7,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")
    metrics = relation_metrics(records)
    compatible = [row for row in records if row["token_compatible"]]
    coverage = len(compatible) / len(records)
    accuracy = (
        sum(float(row["exact"]) for row in compatible) / len(compatible)
        if compatible
        else 0.0
    )
    macro_accuracy = sum(row["accuracy"] for row in metrics.values()) / len(metrics)
    relation_coverage_floor = min(row["coverage"] for row in metrics.values())
    relation_accuracy_floor = min(row["accuracy"] for row in metrics.values())
    clauses = {
        "overall_tokenization_coverage": coverage >= min_overall_coverage,
        "per_relation_tokenization_coverage": (
            relation_coverage_floor >= min_relation_coverage
        ),
        "overall_clean_accuracy": accuracy >= min_overall_accuracy,
        "macro_relation_clean_accuracy": macro_accuracy >= min_macro_accuracy,
        "per_relation_clean_accuracy": relation_accuracy_floor
        >= min_relation_accuracy,
    }
    return {
        "passed": all(clauses.values()),
        "clauses": clauses,
        "rows": len(records),
        "compatible_rows": len(compatible),
        "coverage": coverage,
        "accuracy": accuracy,
        "macro_accuracy": macro_accuracy,
        "relation_coverage_floor": relation_coverage_floor,
        "relation_accuracy_floor": relation_accuracy_floor,
        "thresholds": {
            "min_overall_coverage": min_overall_coverage,
            "min_relation_coverage": min_relation_coverage,
            "min_overall_accuracy": min_overall_accuracy,
            "min_macro_accuracy": min_macro_accuracy,
            "min_relation_accuracy": min_relation_accuracy,
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, metrics: dict[str, dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    relations = sorted(metrics)
    coverage = [metrics[relation]["coverage"] for relation in relations]
    accuracy = [metrics[relation]["accuracy"] for relation in relations]
    positions = list(range(len(relations)))
    figure, axis = plt.subplots(figsize=(7, 4.5))
    width = 0.36
    axis.bar(
        [position - width / 2 for position in positions],
        coverage,
        width,
        label="token coverage",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        accuracy,
        width,
        label="clean accuracy",
    )
    axis.axhline(0.8, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(positions, relations)
    axis.set_ylim(0, 1.03)
    axis.set_ylabel("fraction")
    axis.set_title("Qwen3-8B geography-composition confirmation")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    source_rows, data_manifest = load_geography(args.data)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, revision=args.revision
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        revision=args.revision,
        dtype="bfloat16",
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    records = evaluate(model, tokenizer, source_rows, batch_size=args.batch_size)
    metrics = relation_metrics(records)
    verdict = capability_verdict(records)
    payload = {
        "schema_version": 1,
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.revision,
        "protocol": "raw_full_vocab_greedy_next_token",
        "batch_size": args.batch_size,
        "dataset": data_manifest,
        "prompt_sha256": hashlib.sha256(
            "\n\0\n".join(row["prompt"] for row in source_rows).encode()
        ).hexdigest(),
        "capability_gate": verdict,
        "relation_metrics": metrics,
    }
    args.output_dir.mkdir(parents=True)
    _write_csv(args.output_dir / "predictions.csv", records)
    (args.output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(args.output_dir / "capability_gate.png", metrics)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default=QWEN3_8B_REVISION)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
