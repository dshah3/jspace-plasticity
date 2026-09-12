from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

from jspace_plasticity.evals.country_probe_diagnostic import (
    PROBE_LAYERS,
    PROBE_STATES,
    freeze_output_protection,
    linear_probe_scores,
    load_design,
)
from jspace_plasticity.intervention import AblationPlan

ROOT = Path(__file__).parents[1]
DESIGN = ROOT / "data/evals/q35-country-probe-diagnostic-20260829.json"
PROBE_CORPUS = ROOT / "data/closedbook/country-probe-s20260829.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> AblationPlan:
    return AblationPlan(
        directions={16: torch.ones((1, 2, 1, 3))},
        selected_token_ids={16: torch.tensor([[[1], [2]]])},
        blocked_token_ids=torch.tensor([[[4, 5], [6, 7]]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        prompt_length=2,
        clean_next_logits=torch.zeros((1, 11)),
        clean_top_token_ids=torch.zeros((1, 2), dtype=torch.long),
        ranked_token_ids={16: torch.tensor([[[1, 3], [2, 4]]])},
        audit_top_k=2,
    )


def test_freeze_output_protection_replaces_only_exemptions_and_caches() -> None:
    original = _plan()
    frozen_ids = torch.tensor([[[8, 9], [9, 10]]])
    frozen = freeze_output_protection(original, frozen_ids)

    assert torch.equal(frozen.blocked_token_ids, frozen_ids)
    assert frozen.blocked_token_ids.data_ptr() != frozen_ids.data_ptr()
    assert torch.equal(original.blocked_token_ids, torch.tensor([[[4, 5], [6, 7]]]))
    assert frozen.directions is original.directions
    assert frozen.selected_token_ids == {}
    assert frozen.ranked_token_ids is None
    assert frozen.audit_top_k is None
    assert frozen.attention_mask is original.attention_mask
    assert frozen.clean_next_logits is original.clean_next_logits


def test_freeze_output_protection_rejects_misaligned_sequences() -> None:
    with pytest.raises(ValueError, match="differ in shape"):
        freeze_output_protection(_plan(), torch.zeros((1, 3, 2), dtype=torch.long))


def test_city_disjoint_linear_probe_recovers_known_linear_code() -> None:
    rng = np.random.default_rng(7)
    classes = 8
    folds = np.tile(np.arange(4), classes)
    labels = np.repeat(np.arange(classes), 4)
    values = np.eye(classes, dtype=np.float32)[labels]
    values += rng.normal(scale=0.005, size=values.shape).astype(np.float32)

    metrics = linear_probe_scores(values, labels, folds, device=torch.device("cpu"))
    assert metrics["rows"] == 32
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["top5_accuracy"] == pytest.approx(1.0)


def test_frozen_design_is_evaluation_only_and_hash_locked() -> None:
    design = load_design(DESIGN, _sha256(DESIGN))
    assert design["evaluation"]["training"] is False
    assert design["evaluation"]["model_parameters_updated"] is False
    assert design["probe"]["layers"] == list(PROBE_LAYERS)
    assert design["probe"]["states"] == list(PROBE_STATES)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_design(DESIGN, "0" * 64)


def test_probe_corpus_is_balanced_and_city_disjoint() -> None:
    rows = [
        json.loads(line)
        for line in PROBE_CORPUS.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 612
    by_country: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_country[row["cluster_id"]].append(row)
    assert len(by_country) == 51
    for country_rows in by_country.values():
        assert len(country_rows) == 12
        cities_by_fold = {int(row["city_fold"]): row["city"] for row in country_rows}
        assert set(cities_by_fold) == {0, 1, 2, 3}
        assert len(set(cities_by_fold.values())) == 4
        for fold in range(4):
            fold_rows = [row for row in country_rows if row["city_fold"] == fold]
            assert {row["relation"] for row in fold_rows} == {
                "capital",
                "currency",
                "region",
            }
