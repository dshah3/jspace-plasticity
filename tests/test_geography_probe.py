from __future__ import annotations

from pathlib import Path

import pytest

from jspace_plasticity.evals.geography_probe import (
    EXPECTED_RELATIONS,
    capability_verdict,
    load_geography,
)


def test_frozen_geography_confirmation_is_hash_bound_and_balanced() -> None:
    root = Path(__file__).resolve().parents[1]
    rows, manifest = load_geography(
        root / "data/geography/geography-confirmation.json"
    )
    assert manifest["rows"] == 128
    assert len(rows) == 128
    assert {row["relation"] for row in rows} == EXPECTED_RELATIONS
    assert len({row["iso"] for row in rows}) == 32
    france = {row["relation"]: row for row in rows if row["iso"] == "FR"}
    assert france["capital"]["prompt"] == (
        "Fact: The capital of the country containing Marseille is"
    )
    assert france["capital"]["answer"] == "Paris"
    assert france["currency"]["answer"] == "EUR"


def _records(
    *, accuracy_by_relation: dict[str, int], compatible_by_relation: int = 32
) -> list[dict[str, object]]:
    rows = []
    for relation in sorted(EXPECTED_RELATIONS):
        for index in range(32):
            compatible = index < compatible_by_relation
            rows.append(
                {
                    "relation": relation,
                    "token_compatible": compatible,
                    "exact": (
                        float(index < accuracy_by_relation[relation])
                        if compatible
                        else None
                    ),
                }
            )
    return rows


def test_geography_gate_requires_macro_and_each_relation() -> None:
    passed = capability_verdict(
        _records(accuracy_by_relation={relation: 26 for relation in EXPECTED_RELATIONS})
    )
    assert passed["passed"]
    assert passed["accuracy"] == pytest.approx(26 / 32)

    one_weak_relation = {relation: 29 for relation in EXPECTED_RELATIONS}
    one_weak_relation["currency"] = 21
    failed = capability_verdict(_records(accuracy_by_relation=one_weak_relation))
    assert not failed["passed"]
    assert failed["macro_accuracy"] > 0.8
    assert not failed["clauses"]["per_relation_clean_accuracy"]


def test_geography_gate_requires_per_relation_coverage() -> None:
    records = _records(
        accuracy_by_relation={relation: 24 for relation in EXPECTED_RELATIONS},
        compatible_by_relation=23,
    )
    verdict = capability_verdict(records)
    assert not verdict["passed"]
    assert not verdict["clauses"]["per_relation_tokenization_coverage"]
