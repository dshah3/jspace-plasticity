from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jspace_plasticity.lens.reassess_quality import (
    CORRECTED_KURTOSIS_CLAUSE,
    LEGACY_KURTOSIS_CLAUSE,
    reassess,
)


def _write_source(path: Path, values: list[float]) -> str:
    payload = {
        "schema_version": 1,
        "status": "failed",
        "scientifically_usable": False,
        "why_not_yet_usable": "objective gate failed",
        "objective_clauses": {
            "finite_geometry": True,
            LEGACY_KURTOSIS_CLAUSE: False,
            "two_hop": True,
        },
        "blocking_failures": [LEGACY_KURTOSIS_CLAUSE],
        "geometry": {"readout_logit_excess_kurtosis": values},
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reassessment_passes_rising_curve_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source_sha = _write_source(
        source,
        [3.0, 2.5, 1.7, 1.6, 1.0, 1.0, 1.3, 1.5, 1.4, 1.3, 1.2, 1.1],
    )
    before = source.read_bytes()
    output = tmp_path / "reassessed.json"
    result = reassess(
        source_result=source,
        expected_source_sha256=source_sha,
        output=output,
    )
    assert source.read_bytes() == before
    assert result["status"] == "manual_review_required"
    assert result["blocking_failures"] == []
    assert LEGACY_KURTOSIS_CLAUSE not in result["objective_clauses"]
    assert result["objective_clauses"][CORRECTED_KURTOSIS_CLAUSE] is True
    assert result["reassessment"]["source_result_sha256"] == source_sha


def test_reassessment_keeps_other_failures_blocking(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source_sha = _write_source(source, [1.0] * 12)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["objective_clauses"]["two_hop"] = False
    source.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    result = reassess(
        source_result=source,
        expected_source_sha256=source_sha,
        output=tmp_path / "reassessed.json",
    )
    assert result["status"] == "failed"
    assert CORRECTED_KURTOSIS_CLAUSE in result["blocking_failures"]
    assert "two_hop" in result["blocking_failures"]


def test_reassessment_requires_exact_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    _write_source(source, [1.0] * 12)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        reassess(
            source_result=source,
            expected_source_sha256="0" * 64,
            output=tmp_path / "reassessed.json",
        )


def test_reassessment_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source_sha = _write_source(source, [1.0] * 12)
    output = tmp_path / "reassessed.json"
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        reassess(
            source_result=source,
            expected_source_sha256=source_sha,
            output=output,
        )
