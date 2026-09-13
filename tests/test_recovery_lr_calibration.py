from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from jspace_plasticity import synthetic_recovery_sft as sft
from jspace_plasticity.recovery_lr_calibration import (
    NAMES,
    STEPS,
    reduce_calibration,
    select_candidate,
    validation_counts,
)

ROOT = Path(__file__).parents[1]
DESIGN = ROOT / "data/evals/q35-full-lr-calibration-20260905.json"


def rows(j=12, clean=64, phase="initial"):
    return [
        {
            "source_id": str(i),
            "split": "val",
            "dataset": "synthetic_geo",
            "phase": phase,
            "expected_token_id": 1,
            "clean_exact": int(i < clean),
            "clean_predicted_token_id": int(i < clean),
            "jspace_exact": int(i < j),
            "jspace_predicted_token_id": int(i < j),
            "random_exact": 1,
            "random_predicted_token_id": 1,
        }
        for i in range(64)
    ]


def test_lr_design_and_validation_boundaries():
    design = sft.load_design(DESIGN, sft.sha256_path(DESIGN))
    assert [a["learning_rate"] for a in design["arms"]] == [1e-6, 3e-7]
    assert design["training"]["trainable_policy"] == "all_current_decoder_parameters"
    assert validation_counts(rows())["rows"] == 64
    bad = rows()
    bad[0]["split"] = "screen"
    with pytest.raises(ValueError, match="screen or transfer"):
        validation_counts(bad)


def test_selection_retention_and_tie_order():
    candidate = {
        "rows": 64,
        "initial_jspace_correct": 12,
        "jspace_correct": 40,
        "clean_correct": 61,
        "step": 10,
        "learning_rate": 1e-6,
    }
    assert select_candidate([{**candidate, "clean_correct": 60}]) is None
    assert select_candidate([{**candidate, "jspace_correct": 12}]) is None
    assert select_candidate([candidate, {**candidate, "step": 25}]) == candidate
    assert (
        select_candidate([candidate, {**candidate, "learning_rate": 3e-7}])[
            "learning_rate"
        ]
        == 3e-7
    )


def fake_model():
    m = torch.nn.Linear(1, 1)
    m.device = torch.device("cpu")
    return m


def fake_save(model, tokenizer, directory):
    cp = directory / "checkpoint-terminal"
    cp.mkdir()
    (cp / "weights").write_bytes(b"weights")
    manifest = {
        "path": str(cp),
        "files": [
            {
                "path": "weights",
                "bytes": 7,
                "sha256": hashlib.sha256(b"weights").hexdigest(),
            }
        ],
    }
    sft.write_json(directory / "checkpoint-manifest.json", manifest)
    return {
        **manifest,
        "manifest_sha256": sft.sha256_path(directory / "checkpoint-manifest.json"),
    }


def test_milestones_retain_best_and_restore_training_rng(tmp_path, monkeypatch):
    model = fake_model()
    model.train()
    current = {"j": 20, "clean": 64}

    def evaluate(*args, **kwargs):
        model.eval()
        torch.rand(10)
        return rows(**current, phase=kwargs["phase"])

    monkeypatch.setattr(sft, "evaluate_conditions", evaluate)
    monkeypatch.setattr(sft, "_save_checkpoint", fake_save)
    state = torch.get_rng_state().clone()
    previous = []
    for step, j, clean in [(10, 20, 64), (25, 15, 64), (50, 30, 64), (100, 40, 60)]:
        current.update(j=j, clean=clean)
        previous.append(
            sft._calibration_milestone(
                model,
                None,
                None,
                None,
                rows(),
                step=step,
                output_dir=tmp_path / "arm",
                initial_jspace_correct=12,
                learning_rate=1e-6,
                previous=previous,
            )
        )
        assert model.training
        assert torch.equal(state, torch.get_rng_state())
    assert not (tmp_path / "arm/milestones/step-0010/checkpoint-terminal").exists()
    assert (tmp_path / "arm/milestones/step-0010/checkpoint-manifest.json").is_file()
    assert (tmp_path / "arm/milestones/step-0050/checkpoint-terminal/weights").is_file()
    assert previous[-1]["checkpoint"] is None


def test_failed_replacement_preserves_previous_checkpoint(tmp_path, monkeypatch):
    model = fake_model()
    monkeypatch.setattr(sft, "evaluate_conditions", lambda *a, **k: rows(j=20))
    monkeypatch.setattr(sft, "_save_checkpoint", fake_save)
    first = sft._calibration_milestone(
        model,
        None,
        None,
        None,
        rows(),
        step=10,
        output_dir=tmp_path / "arm",
        initial_jspace_correct=12,
        learning_rate=1e-6,
        previous=[],
    )
    monkeypatch.setattr(sft, "evaluate_conditions", lambda *a, **k: rows(j=30))

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(sft, "_save_checkpoint", fail)
    with pytest.raises(OSError, match="disk full"):
        sft._calibration_milestone(
            model,
            None,
            None,
            None,
            rows(),
            step=25,
            output_dir=tmp_path / "arm",
            initial_jspace_correct=12,
            learning_rate=1e-6,
            previous=[first],
        )
    assert (tmp_path / "arm/milestones/step-0010/checkpoint-terminal/weights").exists()
    assert model.training


def test_two_arm_reducer_selects_only_retained_validation_candidate(
    tmp_path, monkeypatch
):
    design = sft.load_design(DESIGN, sft.sha256_path(DESIGN))
    monkeypatch.setattr(sft, "_save_checkpoint", fake_save)
    for arm in design["arms"]:
        directory = tmp_path / "arms" / f"arm-{arm['index']:02d}-{arm['name']}"
        directory.mkdir(parents=True)
        initial = rows()
        (directory / "predictions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in initial)
        )
        previous = []
        for step in STEPS:
            count = 20 + STEPS.index(step)
            monkeypatch.setattr(
                sft,
                "evaluate_conditions",
                lambda *a, j=count, **kw: rows(j=j, phase=kw["phase"]),
            )
            previous.append(
                sft._calibration_milestone(
                    fake_model(),
                    None,
                    None,
                    None,
                    rows(),
                    step=step,
                    output_dir=directory,
                    initial_jspace_correct=12,
                    learning_rate=arm["learning_rate"],
                    previous=previous,
                )
            )
        sft.write_json(
            directory / "result.json",
            {
                "status": "completed",
                "arm": arm,
                "design_sha256": sft.sha256_path(DESIGN),
                "experiment_image": "test",
                "direction_convention": "effective_gain",
                "runtime": {"trainable_parameters": 3570049536},
                "calibration_milestones": previous,
                "predictions_sha256": sft.sha256_path(directory / "predictions.jsonl"),
            },
        )
    summary = reduce_calibration(
        Namespace(
            output_dir=tmp_path,
            design_sha256=sft.sha256_path(DESIGN),
            experiment_image="test",
        ),
        design,
    )
    assert summary["selected"]["arm"] == NAMES[1]
    assert summary["selected"]["step"] == 100
    assert sum(c["checkpoint_retained"] for c in summary["candidates"]) == 2
