from __future__ import annotations

import pytest

from jspace_plasticity.evals.synthetic_expansion_triage import (
    EXPECTED_WORLD_SIZE,
    paired_metrics,
    summarize,
    workload,
)


def _row(
    source_id: str,
    *,
    clean: int,
    intervention: int,
    cluster: str = "cluster",
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "cluster_id": cluster,
        "relation": "capital",
        "template_id": "capital-t0",
        "token_compatible": True,
        "expected_token_id": 7,
        "clean_predicted_token_id": 7 if clean else 8,
        "clean_exact": float(clean),
        "intervention_exact": float(intervention),
    }


def test_workloads_cover_split_condition_product() -> None:
    observed = [workload(rank) for rank in range(EXPECTED_WORLD_SIZE)]
    assert observed == [
        {"split": "train", "condition": "jspace"},
        {"split": "train", "condition": "matched_random"},
        {"split": "val", "condition": "jspace"},
        {"split": "val", "condition": "matched_random"},
        {"split": "screen", "condition": "jspace"},
        {"split": "screen", "condition": "matched_random"},
    ]
    with pytest.raises(ValueError, match="world_size"):
        workload(0, world_size=5)


def test_summary_filters_to_clean_correct_rows() -> None:
    records = [
        _row("a", clean=1, intervention=1, cluster="x"),
        _row("b", clean=1, intervention=0, cluster="y"),
        _row("c", clean=0, intervention=1, cluster="z"),
    ]
    result = summarize(records)
    assert result["clean_correct_rows"] == 2
    assert result["clean_correct_clusters"] == 2
    assert result["retention"] == 0.5


def test_paired_metrics_use_same_clean_correct_population() -> None:
    jspace = [
        _row("both", clean=1, intervention=1),
        _row("j-only", clean=1, intervention=1),
        _row("r-only", clean=1, intervention=0),
        _row("neither", clean=1, intervention=0),
        _row("clean-wrong", clean=0, intervention=1),
    ]
    random = [
        _row("both", clean=1, intervention=1),
        _row("j-only", clean=1, intervention=0),
        _row("r-only", clean=1, intervention=1),
        _row("neither", clean=1, intervention=0),
        _row("clean-wrong", clean=0, intervention=0),
    ]
    result = paired_metrics(jspace, random)
    assert result["clean_correct_rows"] == 4
    assert result["both_correct"] == 1
    assert result["jspace_only_correct"] == 1
    assert result["random_only_correct"] == 1
    assert result["both_wrong"] == 1
    assert result["random_minus_jspace"] == 0.0
    assert result["mcnemar_exact_two_sided_p"] == 1.0


def test_paired_metrics_reject_clean_path_drift() -> None:
    jspace = [_row("a", clean=1, intervention=0)]
    random = [_row("a", clean=0, intervention=1)]
    with pytest.raises(ValueError, match="clean-path drift"):
        paired_metrics(jspace, random)
