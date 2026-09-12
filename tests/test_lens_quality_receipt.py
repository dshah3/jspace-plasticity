from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jspace_plasticity.lens.quality_receipt import (
    CONFIRMATION,
    issue_receipt,
    require_receipt,
)


def _quality(path: Path, lens: Path, *, prompts: int = 1000) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "manual_review_required",
                "scientifically_usable": False,
                "checkpoint": "model",
                "checkpoint_revision": "revision",
                "lens": {
                    "sha256": hashlib.sha256(lens.read_bytes()).hexdigest(),
                    "n_prompts": prompts,
                    "d_model": 2,
                    "layers": [0, 1],
                },
                "objective_clauses": {"geometry": True, "two_hop": True},
                "blocking_failures": [],
            }
        ),
        encoding="utf-8",
    )


def test_quality_receipt_requires_converged_reviewed_lens(tmp_path: Path) -> None:
    lens = tmp_path / "lens.pt"
    lens.write_bytes(b"lens")
    quality = tmp_path / "quality.json"
    _quality(quality, lens)
    convergence = tmp_path / "convergence.csv"
    convergence.write_text(
        "prompt,layer_0_mean_rel_change,layer_1_mean_rel_change\n"
        + "".join(f"{index},0.001,0.0015\n" for index in range(1000)),
        encoding="utf-8",
    )
    output = tmp_path / "receipt.json"
    issued = issue_receipt(
        quality_result=quality,
        convergence=convergence,
        fit_result=None,
        lens=lens,
        output=output,
        reviewer="reviewer",
        confirmation=CONFIRMATION,
        minimum_prompts=1000,
        maximum_relative_change=0.002,
    )
    assert issued["scientifically_usable"]
    loaded = require_receipt(
        output,
        lens=lens,
        checkpoint="model",
        checkpoint_revision="revision",
        minimum_prompts=1000,
    )
    assert loaded["lens"]["n_prompts"] == 1000


@pytest.mark.parametrize(
    ("prompts", "last_change", "match"),
    [(80, 0.001, "requires 1000"), (1000, 0.03, "exceeds 0.002")],
)
def test_quality_receipt_rejects_partial_or_unconverged_lens(
    tmp_path: Path, prompts: int, last_change: float, match: str
) -> None:
    lens = tmp_path / "lens.pt"
    lens.write_bytes(b"lens")
    quality = tmp_path / "quality.json"
    _quality(quality, lens, prompts=prompts)
    convergence = tmp_path / "convergence.csv"
    convergence.write_text(
        "prompt,mean_rel_change\n"
        + "".join(f"{index},{last_change}\n" for index in range(prompts)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=match):
        issue_receipt(
            quality_result=quality,
            convergence=convergence,
            fit_result=None,
            lens=lens,
            output=tmp_path / "receipt.json",
            reviewer="reviewer",
            confirmation=CONFIRMATION,
            minimum_prompts=1000,
            maximum_relative_change=0.002,
        )


def test_quality_receipt_accepts_completed_exact_1000_prompt_fit(
    tmp_path: Path,
) -> None:
    lens = tmp_path / "lens.pt"
    lens.write_bytes(b"lens")
    lens_sha256 = hashlib.sha256(lens.read_bytes()).hexdigest()
    quality = tmp_path / "quality.json"
    _quality(quality, lens)
    fit_result = tmp_path / "fit-result.json"
    fit_result.write_text(
        json.dumps(
            {
                "status": "completed",
                "num_prompts": 1000,
                "fit_config_sha256": "config-hash",
                "artifacts": {"lens": {"sha256": lens_sha256}},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "receipt.json"
    receipt = issue_receipt(
        quality_result=quality,
        convergence=None,
        fit_result=fit_result,
        lens=lens,
        output=output,
        reviewer="reviewer",
        confirmation=CONFIRMATION,
        minimum_prompts=1000,
        maximum_relative_change=0.002,
    )
    assert receipt["fit_evidence"]["kind"] == "exact_averaged_fit"
    require_receipt(
        output,
        lens=lens,
        checkpoint="model",
        checkpoint_revision="revision",
        minimum_prompts=1000,
    )
