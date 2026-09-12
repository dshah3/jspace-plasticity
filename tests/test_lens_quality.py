from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jspace_plasticity.lens.validate_quality import (
    _block_statistics,
    _load_corpus_partition,
    _post_onset_kurtosis_rise,
    _select_rows,
    _target_best_rank,
    _target_token_ids,
)


class TinyTokenizer:
    def __init__(self) -> None:
        self.mapping = {"cat": [7], " cat": [8], "Cat": [9], " Cat": [10]}
        self.decoded = {7: "cat", 8: " cat", 9: "Cat", 10: " Cat"}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return self.mapping.get(text, [1, 2])

    def decode(self, token_ids: list[int]) -> str:
        return self.decoded[token_ids[0]]


def test_corpus_partition_is_exact_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "".join(json.dumps({"text": f"row-{index}"}) + "\n" for index in range(5)),
        encoding="utf-8",
    )
    assert _load_corpus_partition(path, offset=2, count=2) == [
        {"corpus_index": 2, "text": "row-2"},
        {"corpus_index": 3, "text": "row-3"},
    ]


def test_two_hop_selection_is_seeded_and_order_independent() -> None:
    rows = [{"name": name} for name in ("a", "b", "c", "d")]
    assert _select_rows(rows, count=2, seed=17) == _select_rows(
        list(reversed(rows)), count=2, seed=17
    )


def test_target_token_ids_keep_exact_single_token_variants() -> None:
    assert _target_token_ids(TinyTokenizer(), "cat") == [7, 8, 9, 10]


def test_target_best_rank_scans_all_positions_and_variants() -> None:
    logits = torch.tensor(
        [
            [5.0, 4.0, 3.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
        ]
    )
    assert _target_best_rank(logits, [1, 2]) == (2, 0)


def test_block_statistics_compare_within_and_between() -> None:
    cka = np.full((6, 6), 0.1)
    for start in (0, 2, 4):
        cka[start : start + 2, start : start + 2] = 0.9
    np.fill_diagonal(cka, 1.0)
    metrics = _block_statistics(cka, (2, 4))
    assert metrics["within_block_mean"] == 0.9
    assert metrics["between_block_mean"] == pytest.approx(0.1)
    assert metrics["within_minus_between"] > 0


def test_post_onset_kurtosis_rise_accepts_published_n1000_control() -> None:
    values = [
        2.9797,
        2.4739,
        1.6837,
        1.6188,
        1.4697,
        1.2552,
        1.1670,
        1.0879,
        1.0657,
        1.0884,
        1.0161,
        1.0465,
        1.1982,
        1.0969,
        1.0663,
        1.3494,
        1.2800,
        1.3563,
        1.3549,
        1.6266,
        1.6117,
        1.4703,
        1.4342,
        1.4154,
        1.7146,
        1.8794,
        1.9601,
        1.8190,
        1.8319,
        1.7876,
        1.8458,
    ]
    result = _post_onset_kurtosis_rise(values)
    assert result["passed"] is True
    assert result["relative_rise"] == pytest.approx(0.296, abs=0.002)
    assert result["post_onset_window_indices"] == [10, 13]
    assert result["pre_motor_window_indices"] == [17, 20]


def test_post_onset_kurtosis_rise_rejects_qwen32_pilot_drift() -> None:
    values = [
        1.0253,
        0.9275,
        0.9582,
        1.1337,
        1.4027,
        1.0216,
        0.6470,
        0.6242,
        0.6398,
        0.6836,
        0.6641,
        0.6782,
        0.6115,
        0.6470,
        0.6781,
        0.8196,
        0.6536,
        0.5459,
        0.5861,
        0.6177,
        0.6002,
        0.7339,
        0.9925,
        1.1790,
        0.7785,
    ]
    result = _post_onset_kurtosis_rise(values)
    assert result["passed"] is False
    assert 0.01 < result["relative_rise"] < 0.03


@pytest.mark.parametrize("values", [[], [1.0] * 8, [1.0] * 8 + [np.nan]])
def test_post_onset_kurtosis_rise_rejects_invalid_inputs(values: list[float]) -> None:
    with pytest.raises(ValueError):
        _post_onset_kurtosis_rise(values)
