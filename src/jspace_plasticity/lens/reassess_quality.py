"""Reassess a completed lens-quality result after a metric correction.

This command never loads a model or lens and never recomputes measurements.  It
binds to an immutable source-result SHA-256, replaces only the superseded
kurtosis clause, and writes a new result without modifying the historical one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from jspace_plasticity.lens.validate_quality import (
    QUALITY_SCHEMA_VERSION,
    _post_onset_kurtosis_rise,
)

LEGACY_KURTOSIS_CLAUSE = "kurtosis_rises_after_first_third"
CORRECTED_KURTOSIS_CLAUSE = (
    "kurtosis_post_onset_relative_rise_at_least_10pct"
)
PAPER_URL = "https://transformer-circuits.pub/2026/workspace/index.html"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reassess(
    *, source_result: Path, expected_source_sha256: str, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reassessment: {output}")
    observed_sha256 = _sha256(source_result)
    if observed_sha256 != expected_source_sha256:
        raise ValueError(
            "quality source SHA-256 mismatch: "
            f"expected {expected_source_sha256}, observed {observed_sha256}"
        )

    source = json.loads(source_result.read_text(encoding="utf-8"))
    if source.get("schema_version") not in {1, QUALITY_SCHEMA_VERSION}:
        raise ValueError("unsupported quality-result schema")
    if source.get("scientifically_usable") is not False:
        raise ValueError("source quality result must not already approve the lens")
    geometry = source.get("geometry", {})
    values = geometry.get("readout_logit_excess_kurtosis")
    if not isinstance(values, list):
        raise ValueError("source result has no layerwise readout kurtosis")
    clauses = source.get("objective_clauses")
    if not isinstance(clauses, dict) or not clauses:
        raise ValueError("source result has no objective clauses")

    corrected = deepcopy(source)
    rise = _post_onset_kurtosis_rise(values)
    corrected_clauses = dict(clauses)
    previous_kurtosis_value = corrected_clauses.pop(LEGACY_KURTOSIS_CLAUSE, None)
    corrected_clauses[CORRECTED_KURTOSIS_CLAUSE] = rise["passed"]
    blocking = sorted(
        name for name, passed in corrected_clauses.items() if not bool(passed)
    )

    corrected["schema_version"] = QUALITY_SCHEMA_VERSION
    corrected["status"] = "failed" if blocking else "manual_review_required"
    corrected["scientifically_usable"] = False
    corrected["why_not_yet_usable"] = (
        "objective gate failed"
        if blocking
        else "qualitative abstract/context readouts and CKA blocks need review"
    )
    corrected["objective_clauses"] = corrected_clauses
    corrected["blocking_failures"] = blocking
    corrected["geometry"]["post_onset_kurtosis_rise"] = rise
    corrected["reassessment"] = {
        "kind": "metric_correction_without_measurement_recomputation",
        "source_result_path": str(source_result),
        "source_result_sha256": observed_sha256,
        "source_schema_version": source.get("schema_version"),
        "source_status": source.get("status"),
        "source_blocking_failures": source.get("blocking_failures", []),
        "superseded_clause": LEGACY_KURTOSIS_CLAUSE,
        "superseded_clause_value": previous_kurtosis_value,
        "corrected_clause": CORRECTED_KURTOSIS_CLAUSE,
        "paper_source": PAPER_URL,
        "reason": (
            "The paper describes a rise beginning around one-third depth. "
            "The superseded gate instead compared the whole middle-third "
            "median against the early-third median, which is not that test."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(corrected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            reassess(
                source_result=args.source_result,
                expected_source_sha256=args.expected_source_sha256,
                output=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
