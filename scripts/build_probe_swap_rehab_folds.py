#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Freeze outcome-independent folds plus baseline eligibility for rehab SFT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_SEED = 20260825


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(seed: int, purpose: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{purpose}:{value}".encode()).hexdigest()


def assign_test_folds(rows: list[dict[str, Any]], seed: int) -> dict[int, int]:
    """Stratify categories deterministically while keeping exactly 18/fold."""

    by_category: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_category[row["category"]].append(index)
    fold_rows: list[list[int]] = [[] for _ in range(5)]
    category_counts = [Counter() for _ in range(5)]
    categories = sorted(by_category, key=lambda c: (-len(by_category[c]), c))
    for category in categories:
        indices = sorted(
            by_category[category],
            key=lambda i: stable_key(seed, f"test-row:{category}", rows[i]["name"]),
        )
        for index in indices:
            candidates = [fold for fold in range(5) if len(fold_rows[fold]) < 18]
            fold = min(
                candidates,
                key=lambda f: (
                    category_counts[f][category],
                    len(fold_rows[f]),
                    stable_key(
                        seed, f"test-fold:{category}:{rows[index]['name']}", str(f)
                    ),
                ),
            )
            fold_rows[fold].append(index)
            category_counts[fold][category] += 1
    sizes = [len(indices) for indices in fold_rows]
    if sizes != [18] * 5:
        raise RuntimeError(f"test fold sizes are not balanced: {sizes}")
    assignment = {
        index: fold for fold, indices in enumerate(fold_rows) for index in indices
    }
    if set(assignment) != set(range(len(rows))):
        raise RuntimeError("test fold assignment is incomplete")
    return assignment


def _bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"expected CSV boolean, got {value!r}")
    return value == "True"


def build(
    *,
    data_path: Path,
    clean_predictions_path: Path,
    lesion_predictions_path: Path,
    seed: int,
) -> dict[str, Any]:
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    rows = payload["items"]
    if len(rows) != 90:
        raise ValueError(f"expected 90 probe-swap rows, got {len(rows)}")
    clean_rows = list(csv.DictReader(clean_predictions_path.open(encoding="utf-8")))
    lesion_rows = [
        json.loads(line)
        for line in lesion_predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(clean_rows) != len(rows) or len(lesion_rows) != len(rows):
        raise ValueError("prediction receipts must each contain all 90 source rows")

    outcomes = []
    for index, source in enumerate(rows):
        clean = clean_rows[index]
        lesion = lesion_rows[index]
        if (
            int(clean["example_index"]) != index
            or int(lesion["example_index"]) != index
            or clean["name"] != source["name"]
            or lesion["name"] != source["name"]
            or clean["prompt"] != source["prompt"]
            or lesion["prompt"] != source["prompt"]
            or clean["answer"] != source["answer"]
            or lesion["answer"] != source["answer"]
        ):
            raise ValueError(f"prediction/source mismatch at row {index}")
        compatible = _bool(clean["token_compatible"])
        lesion_compatible = bool(lesion["token_compatible"])
        if compatible != lesion_compatible:
            raise ValueError(f"token compatibility drift at row {index}")
        clean_exact = compatible and float(clean["exact"]) == 1.0
        exact_path_clean = (
            compatible and float(lesion.get("exact_path_clean_exact", 0.0)) == 1.0
        )
        if clean_exact != exact_path_clean:
            raise ValueError(
                f"batch-8 and exact-path clean-correct status differs at row {index}"
            )
        lesion_exact = compatible and float(lesion["exact"]) == 1.0
        outcomes.append(
            {
                "example_index": index,
                "name": source["name"],
                "token_compatible": compatible,
                "expected_token_id": (
                    int(clean["expected_token_id"]) if compatible else None
                ),
                "baseline_clean_correct": clean_exact,
                "baseline_lesion_correct": lesion_exact if clean_exact else None,
                "baseline_lesion_broken": clean_exact and not lesion_exact,
            }
        )

    if sum(row["baseline_clean_correct"] for row in outcomes) != 49:
        raise ValueError("expected exactly 49 baseline-clean-correct prompts")
    if sum(row["baseline_lesion_broken"] for row in outcomes) != 27:
        raise ValueError("expected exactly 27 lesion-broken prompts")

    test_assignment = assign_test_folds(rows, seed)
    folds = []
    for fold in range(5):
        test = sorted(i for i, assigned in test_assignment.items() if assigned == fold)
        remaining = [i for i in range(len(rows)) if i not in set(test)]
        validation = sorted(
            sorted(
                remaining,
                key=lambda i: stable_key(seed, f"validation:{fold}", rows[i]["name"]),
            )[:9]
        )
        validation_set = set(validation)
        train = sorted(i for i in remaining if i not in validation_set)
        eligible_train = [i for i in train if outcomes[i]["baseline_clean_correct"]]
        if (len(train), len(validation), len(test)) != (63, 9, 18):
            raise RuntimeError(f"invalid split sizes for fold {fold}")
        folds.append(
            {
                "fold": fold,
                "train_indices": train,
                "validation_indices": validation,
                "test_indices": test,
                "eligible_train_indices": eligible_train,
                "eligible_train_rows": len(eligible_train),
                "clean_correct_validation_rows": sum(
                    outcomes[i]["baseline_clean_correct"] for i in validation
                ),
                "clean_correct_test_rows": sum(
                    outcomes[i]["baseline_clean_correct"] for i in test
                ),
                "lesion_broken_test_rows": sum(
                    outcomes[i]["baseline_lesion_broken"] for i in test
                ),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "protocol": "probe_swap_five_fold_answer_only_lesion_rehabilitation",
        "source": {
            "dataset_path": str(data_path),
            "dataset_sha256": sha256(data_path),
            "clean_predictions_path": str(clean_predictions_path),
            "clean_predictions_sha256": sha256(clean_predictions_path),
            "lesion_predictions_path": str(lesion_predictions_path),
            "lesion_predictions_sha256": sha256(lesion_predictions_path),
        },
        "fold_construction": {
            "test": (
                "outcome-independent deterministic category-stratified assignment; "
                "each of 90 rows is held out exactly once"
            ),
            "validation": (
                "nine deterministic hash-selected rows from the 72 non-test rows"
            ),
            "train": "the remaining 63 rows",
            "gradient_eligibility": (
                "train rows only when the frozen baseline was token-compatible "
                "and exact under the exact-path clean forward"
            ),
        },
        "rows": outcomes,
        "folds": folds,
        "primary_cohorts": {
            "baseline_clean_correct": 49,
            "baseline_lesion_broken": 27,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--clean-predictions", required=True, type=Path)
    parser.add_argument("--lesion-predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = build(
        data_path=args.data,
        clean_predictions_path=args.clean_predictions,
        lesion_predictions_path=args.lesion_predictions,
        seed=args.seed,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise FileExistsError(
            f"refusing to overwrite different fold file: {args.output}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"sha256 {hashlib.sha256(rendered.encode()).hexdigest()}")
    for fold in result["folds"]:
        print(
            f"fold {fold['fold']}: eligible_train={fold['eligible_train_rows']} "
            f"clean_val={fold['clean_correct_validation_rows']} "
            f"clean_test={fold['clean_correct_test_rows']} "
            f"broken_test={fold['lesion_broken_test_rows']}"
        )


if __name__ == "__main__":
    main()
