from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jspace_plasticity.evals.recovery_figure_suite import (
    EXPECTED_MODELS,
    _geometry_similarity,
    _plot_suite,
    bootstrap_interval,
    load_design,
    rank_category,
    wilson_interval,
)

ROOT = Path(__file__).parents[1]
DESIGN = ROOT / "data/evals/q35-recovery-figure-suite-20260826.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_figure_design_is_evaluation_only() -> None:
    design, source = load_design(DESIGN, _sha256(DESIGN))
    assert design["training"] is False
    assert design["measurement"]["rows"] == 129
    assert [row["model"] for row in design["conditions"]] == [
        "base",
        "primary",
        "high_dose",
    ]
    assert source["evaluation"]["training"] is False


def test_figure_design_hash_and_decision_are_blocking(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_design(DESIGN, "0" * 64)
    mutated = json.loads(DESIGN.read_text(encoding="utf-8"))
    mutated["training"] = True
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation-only"):
        load_design(path, _sha256(path))


def test_bootstrap_is_deterministic_and_contains_point_estimate() -> None:
    values = [1.0, 2.0, 3.0, 8.0, 13.0]
    first = bootstrap_interval(values, resamples=500, seed=7)
    second = bootstrap_interval(values, resamples=500, seed=7)
    assert first == second
    assert first["estimate"] == 3.0
    assert first["low"] <= first["estimate"] <= first["high"]


def test_wilson_interval_handles_observed_recovery_rate() -> None:
    low, high = wilson_interval(114, 129)
    assert low < 114 / 129 < high
    assert 0.80 < low < 0.90
    assert 0.90 < high < 0.95


@pytest.mark.parametrize(
    ("audit", "expected"),
    [
        (
            {
                "intermediate_output_blocked": True,
                "intermediate_seen_top10_any_layer": True,
                "intermediate_rank11_12_without_top10": False,
                "intermediate_best_rank_any_layer": 1,
            },
            "output_protected",
        ),
        (
            {
                "intermediate_output_blocked": False,
                "intermediate_seen_top10_any_layer": True,
                "intermediate_rank11_12_without_top10": False,
                "intermediate_best_rank_any_layer": 6,
            },
            "top_10",
        ),
        (
            {
                "intermediate_output_blocked": False,
                "intermediate_seen_top10_any_layer": False,
                "intermediate_rank11_12_without_top10": True,
                "intermediate_best_rank_any_layer": 11,
            },
            "rank_11_12",
        ),
        (
            {
                "intermediate_output_blocked": False,
                "intermediate_seen_top10_any_layer": False,
                "intermediate_rank11_12_without_top10": False,
                "intermediate_best_rank_any_layer": 30,
            },
            "rank_13_50",
        ),
        (
            {
                "intermediate_output_blocked": False,
                "intermediate_seen_top10_any_layer": False,
                "intermediate_rank11_12_without_top10": False,
                "intermediate_best_rank_any_layer": None,
            },
            "not_seen_top_50",
        ),
    ],
)
def test_rank_categories_are_mutually_exclusive(audit: dict, expected: str) -> None:
    assert rank_category(audit) == expected


def test_geometry_similarity_is_one_for_identical_sketches() -> None:
    rng = np.random.default_rng(3)
    array = rng.normal(size=(2, 32, 8)).astype(np.float16)
    fixture = {
        "layers": np.array([16, 18]),
        "token_ids": np.arange(32),
        "sketches": array,
    }
    rows = _geometry_similarity(
        {"base": fixture, "primary": fixture, "high_dose": fixture}
    )
    assert len(rows) == 4
    assert all(row["linear_cka"] == pytest.approx(1.0, abs=1e-5) for row in rows)
    assert all(
        row["mean_paired_cosine"] == pytest.approx(1.0, abs=1e-5) for row in rows
    )


def test_all_figure_variants_render(tmp_path: Path) -> None:
    layers = [16, 18, 19, 20, 21, 22]
    activation_rows = []
    for model_index, model in enumerate(EXPECTED_MODELS):
        for layer in layers:
            for state, offset in (("clean", 0.0), ("lesion", 0.2)):
                value = 1.0 + model_index * 0.1 + layer * 0.01 + offset
                activation_rows.append(
                    {
                        "model": model,
                        "layer": layer,
                        "state": state,
                        "metric": "kurtosis",
                        "estimate": value,
                        "low": value - 0.05,
                        "high": value + 0.05,
                    }
                )
            for metric, value in (("norm_ratio", 0.8), ("cosine", 0.9)):
                activation_rows.append(
                    {
                        "model": model,
                        "layer": layer,
                        "state": "paired",
                        "metric": metric,
                        "estimate": value,
                        "low": value - 0.02,
                        "high": value + 0.02,
                    }
                )
    rank_rows = [
        {
            "model": model,
            "k": k,
            "control": control,
            "accuracy": 0.2 + model_index * 0.3,
            "low": 0.15 + model_index * 0.3,
            "high": 0.25 + model_index * 0.3,
        }
        for model_index, model in enumerate(EXPECTED_MODELS)
        for k in (10, 12, 16, 20, 32, 50)
        for control in ("jspace", "random")
    ]
    geometry_rows = [
        {
            "model": model,
            "layer": layer,
            "linear_cka": 0.8,
            "mean_paired_cosine": 0.7,
        }
        for model in ("primary", "high_dose")
        for layer in layers
    ]
    categories = (
        "output_protected",
        "top_10",
        "rank_11_12",
        "rank_13_50",
        "not_seen_top_50",
    )
    category_rows = [
        {"model": model, "category": category, "fraction": 0.2}
        for model in EXPECTED_MODELS
        for category in categories
    ]
    figures = _plot_suite(
        tmp_path,
        activation_rows=activation_rows,
        rank_rows=rank_rows,
        geometry_rows=geometry_rows,
        category_rows=category_rows,
        dpi=40,
    )
    assert len(figures) == 9
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures)
    assert (tmp_path / "headline_recovery_figure.pdf").is_file()
