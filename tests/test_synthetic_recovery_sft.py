from __future__ import annotations

import hashlib
import json
import math
from argparse import Namespace
from pathlib import Path

import pytest

from jspace_plasticity.synthetic_recovery_sft import (
    EXPECTED_WORLD_SIZE,
    arm_for,
    load_design,
    paired_change,
    paired_difference,
    reduce_results,
    select_training_rows,
)

ROOT = Path(__file__).parents[1]
DESIGN = ROOT / "data/evals/q35-synthetic-recovery-sweep-20260825.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_design_has_eight_unique_arms_and_one_primary() -> None:
    design = load_design(DESIGN, _sha256(DESIGN))
    assert len(design["arms"]) == EXPECTED_WORLD_SIZE
    assert [arm_for(design, index)["index"] for index in range(8)] == list(range(8))
    assert sum(arm["name"] == "j-full-s160-primary" for arm in design["arms"]) == 1
    assert sum(bool(arm["save_checkpoint"]) for arm in design["arms"]) == 2


def test_design_hash_is_enforced(tmp_path: Path) -> None:
    copied = tmp_path / "design.json"
    copied.write_text(DESIGN.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_design(copied, "0" * 64)


def _eligibility_fixture() -> tuple[list[dict], list[dict]]:
    rows = []
    eligibility = []
    for cluster in ("AA", "BB", "CC", "DD"):
        for index in range(3):
            source_id = f"{cluster}-{index}"
            rows.append(
                {
                    "source_id": source_id,
                    "cluster_id": cluster,
                    "prompt": f"Fact: {source_id} is",
                    "answer": "answer",
                }
            )
            eligibility.append(
                {
                    "source_id": source_id,
                    "cluster_id": cluster,
                    "expected_token_id": index,
                }
            )
    return rows, eligibility


def test_data_scale_selection_is_nested_and_keeps_whole_clusters() -> None:
    rows, eligibility = _eligibility_fixture()
    quarter, quarter_clusters = select_training_rows(rows, eligibility, 0.25)
    half, half_clusters = select_training_rows(rows, eligibility, 0.5)
    full, full_clusters = select_training_rows(rows, eligibility, 1.0)
    assert len(quarter_clusters) == math.ceil(0.25 * 4)
    assert len(half_clusters) == math.ceil(0.5 * 4)
    assert len(full_clusters) == 4
    assert set(quarter_clusters) <= set(half_clusters) <= set(full_clusters)
    for selected, clusters in (
        (quarter, quarter_clusters),
        (half, half_clusters),
        (full, full_clusters),
    ):
        assert {row["cluster_id"] for row in selected} == set(clusters)
        assert len(selected) == 3 * len(clusters)


def test_data_scale_selection_rejects_receipt_outside_train_rows() -> None:
    rows, eligibility = _eligibility_fixture()
    eligibility.append(
        {"source_id": "missing", "cluster_id": "ZZ", "expected_token_id": 1}
    )
    with pytest.raises(ValueError, match="outside the train split"):
        select_training_rows(rows, eligibility, 1.0)


def _record(source_id: str, exact: int) -> dict:
    return {"source_id": source_id, "jspace_exact": exact}


def test_paired_change_counts_recoveries_and_regressions() -> None:
    initial = [_record("a", 0), _record("b", 0), _record("c", 1), _record("d", 1)]
    terminal = [_record("a", 1), _record("b", 0), _record("c", 0), _record("d", 1)]
    result = paired_change(initial, terminal, field="jspace_exact")
    assert result["recovered"] == 1
    assert result["regressed"] == 1
    assert result["net_accuracy_gain"] == 0
    assert result["mcnemar_exact_two_sided_p"] == 1.0


def test_paired_difference_is_left_minus_right() -> None:
    left = [_record("a", 1), _record("b", 1), _record("c", 0), _record("d", 1)]
    right = [_record("a", 0), _record("b", 1), _record("c", 0), _record("d", 0)]
    result = paired_difference(left, right, field="jspace_exact")
    assert result["left_only_correct"] == 2
    assert result["right_only_correct"] == 0
    assert result["left_minus_right"] == 0.5


def test_paired_helpers_reject_different_rows() -> None:
    with pytest.raises(ValueError, match="rows differ"):
        paired_change([_record("a", 0)], [_record("b", 1)], field="jspace_exact")
    with pytest.raises(ValueError, match="rows differ"):
        paired_difference([_record("a", 0)], [_record("b", 1)], field="jspace_exact")


def test_design_documents_no_rl_no_thinking_and_heldout_is_gradient_free() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    assert design["training"]["kind"].endswith("teacher_forced_cross_entropy")
    assert design["training"]["thinking"] is False
    assert design["training"]["chat_template"] is False
    assert design["training"]["rl"] is False
    assert "never" in design["training"]["test_use"]


@pytest.mark.parametrize(
    "design_path",
    [
        DESIGN,
        ROOT / "data/evals/q35-corrected-gain-retraining-20260905.json",
        ROOT / "data/evals/q35-final-capability-20260905.json",
    ],
)
def test_reducer_builds_paired_gate_from_all_frozen_arms(
    tmp_path: Path, design_path: Path
) -> None:
    design = json.loads(design_path.read_text(encoding="utf-8"))
    image = "registry.example/recovery@sha256:" + "a" * 64
    output = tmp_path / "run"

    def prediction(
        source_id: str,
        *,
        phase: str,
        dataset: str,
        split: str,
        jspace_exact: int,
    ) -> dict:
        return {
            "source_id": source_id,
            "phase": phase,
            "dataset": dataset,
            "split": split,
            "expected_token_id": 1,
            "clean_predicted_token_id": 1,
            "jspace_predicted_token_id": jspace_exact,
            "random_predicted_token_id": 1,
            "clean_exact": 1,
            "jspace_exact": jspace_exact,
            "random_exact": 1,
        }

    for arm in design["arms"]:
        arm_dir = output / "arms" / f"arm-{arm['index']:02d}-{arm['name']}"
        arm_dir.mkdir(parents=True)
        terminal_j = int(arm["training_condition"] == "jspace")
        rows = []
        for index in range(6):
            split = "val" if index < 3 else "screen"
            source_id = f"heldout-{index}"
            rows.append(
                prediction(
                    source_id,
                    phase="initial",
                    dataset="synthetic_geo",
                    split=split,
                    jspace_exact=0,
                )
            )
            rows.append(
                prediction(
                    source_id,
                    phase="terminal",
                    dataset="synthetic_geo",
                    split=split,
                    jspace_exact=terminal_j,
                )
            )
        for index in range(2):
            for phase in ("initial", "terminal"):
                rows.append(
                    prediction(
                        f"transfer-{index}",
                        phase=phase,
                        dataset="anthropic_probe_swap",
                        split="transfer",
                        jspace_exact=int(phase == "terminal" and terminal_j == 1),
                    )
                )
        predictions_path = arm_dir / "predictions.jsonl"
        predictions_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        result = {
            "status": "completed",
            "arm": arm,
            "experiment_image": image,
            "direction_convention": design["lesion"].get(
                "direction_convention", "legacy_weight"
            ),
            "selected_train_rows": arm["expected_train_rows"],
            "eligible_dataset_passes": 1.0,
            "predictions_sha256": _sha256(predictions_path),
        }
        (arm_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    summary = reduce_results(
        Namespace(
            design=design_path,
            design_sha256=_sha256(design_path),
            output_dir=output,
            experiment_image=image,
        )
    )
    assert summary["candidate_recovery"] is True
    assert all(summary["candidate_recovery_clauses"].values())
    primary = summary["arm_metrics"][design["evaluation"]["primary_arm"]]
    assert primary["synthetic_heldout_jspace_change"]["recovered"] == 6
    assert (
        summary["primary_control_differences"][
            design["evaluation"].get("sham_arm", "sham-full-s160")
        ]["left_minus_right"]
        == 1.0
    )


def test_corrected_retraining_design_uses_four_arms_and_effective_gain():
    from jspace_plasticity.synthetic_recovery_sft import intervention_config

    path = ROOT / "data/evals/q35-corrected-gain-retraining-20260905.json"
    design = load_design(path, _sha256(path))
    assert len(design["arms"]) == 4
    assert all(a["save_checkpoint"] and a["max_steps"] == 160 for a in design["arms"])
    assert arm_for(design, 3)["training_condition"] == "sham"
    with pytest.raises(ValueError):
        arm_for(design, 4)
    config = intervention_config(
        "jspace",
        Path("lens.pt"),
        direction_convention=design["lesion"]["direction_convention"],
    )
    assert config.direction_convention == "effective_gain"
    assert (
        intervention_config("jspace", Path("lens.pt")).direction_convention
        == "legacy_weight"
    )


def test_corrected_initialization_keeps_clean_parity_but_remeasures_lesion():
    from jspace_plasticity.synthetic_recovery_sft import verify_initial_synthetic

    baseline = dict(
        clean_predicted_token_id=1,
        jspace_predicted_token_id=2,
        random_predicted_token_id=3,
    )
    row = dict(baseline, source_id="a", split="val", jspace_predicted_token_id=4)
    cohorts = {"val": {"a": baseline}}
    verify_initial_synthetic([row], cohorts, direction_convention="effective_gain")
    with pytest.raises(ValueError, match="parity failed"):
        verify_initial_synthetic([row], cohorts)
    row["clean_predicted_token_id"] = 5
    with pytest.raises(ValueError, match="parity failed"):
        verify_initial_synthetic([row], cohorts, direction_convention="effective_gain")


@pytest.mark.parametrize(
    "field,value", [("learning_rate", 1e-5), ("steps", 160), ("save_control", True)]
)
def test_final_design_rejects_protocol_drift(tmp_path, field, value):
    path = ROOT / "data/evals/q35-final-capability-20260905.json"
    design = json.loads(path.read_text())
    if field == "learning_rate":
        design["training"][field] = value
    elif field == "steps":
        design["arms"][0]["max_steps"] = value
    else:
        design["arms"][2]["save_checkpoint"] = value
    changed = tmp_path / "design.json"
    changed.write_text(json.dumps(design))
    with pytest.raises(ValueError, match="final capability check"):
        load_design(changed, _sha256(changed))
