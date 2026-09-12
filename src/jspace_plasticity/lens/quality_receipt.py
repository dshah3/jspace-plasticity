"""Issue and verify a hash-bound, manually reviewed scientific lens receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONFIRMATION = "I reviewed the qualitative J-lens readouts and approve this lens"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _final_convergence(path: Path) -> tuple[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("convergence CSV has no data rows")
    final: dict[str, float] = {}
    for key, value in rows[-1].items():
        normalized = key.casefold().replace("-", "_").replace(" ", "_")
        if "rel_change" not in normalized and "relative_change" not in normalized:
            continue
        if value not in (None, ""):
            final[key] = float(value)
    if not final:
        raise ValueError("convergence CSV has no relative-change columns")
    return len(rows), final


def issue_receipt(
    *,
    quality_result: Path,
    convergence: Path | None,
    fit_result: Path | None,
    lens: Path,
    output: Path,
    reviewer: str,
    confirmation: str,
    minimum_prompts: int,
    maximum_relative_change: float,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite lens receipt: {output}")
    if confirmation != CONFIRMATION:
        raise ValueError("manual-review confirmation text does not match")
    quality = json.loads(quality_result.read_text(encoding="utf-8"))
    if quality.get("status") != "manual_review_required":
        raise ValueError("quality result did not reach manual_review_required")
    if quality.get("blocking_failures"):
        raise ValueError("quality result contains blocking failures")
    if not all(quality.get("objective_clauses", {}).values()):
        raise ValueError("quality result contains a failed objective clause")
    lens_sha256 = _sha256(lens)
    if quality.get("lens", {}).get("sha256") != lens_sha256:
        raise ValueError("quality result is bound to a different lens SHA-256")
    n_prompts = int(quality.get("lens", {}).get("n_prompts", 0))
    if n_prompts < minimum_prompts:
        raise ValueError(
            f"lens has {n_prompts} fitted prompts; requires {minimum_prompts}"
        )
    if (convergence is None) == (fit_result is None):
        raise ValueError("provide exactly one of convergence or fit_result")
    if convergence is not None:
        convergence_rows, final = _final_convergence(convergence)
        if convergence_rows < minimum_prompts:
            raise ValueError(
                f"convergence CSV has {convergence_rows} rows; "
                f"requires {minimum_prompts}"
            )
        worst = max(abs(value) for value in final.values())
        if worst > maximum_relative_change:
            raise ValueError(
                f"final relative change {worst} exceeds {maximum_relative_change}"
            )
        fit_evidence = {
            "kind": "iterative_convergence",
            "path": str(convergence),
            "sha256": _sha256(convergence),
            "rows": convergence_rows,
            "final_relative_changes": final,
            "worst_final_relative_change": worst,
            "maximum_relative_change": maximum_relative_change,
        }
    else:
        assert fit_result is not None
        fitted = json.loads(fit_result.read_text(encoding="utf-8"))
        if fitted.get("status") != "completed":
            raise ValueError("exact fit result is not completed")
        if int(fitted.get("num_prompts", 0)) != n_prompts:
            raise ValueError("exact fit result prompt count disagrees with quality")
        if fitted.get("artifacts", {}).get("lens", {}).get("sha256") != lens_sha256:
            raise ValueError("exact fit result is bound to a different lens")
        fit_evidence = {
            "kind": "exact_averaged_fit",
            "path": str(fit_result),
            "sha256": _sha256(fit_result),
            "num_prompts": int(fitted["num_prompts"]),
            "fit_config_sha256": fitted.get("fit_config_sha256"),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scientifically_usable": True,
        "decision": "scientific_lens_approved",
        "checkpoint": quality["checkpoint"],
        "checkpoint_revision": quality["checkpoint_revision"],
        "lens": {
            "path": str(lens),
            "sha256": lens_sha256,
            "n_prompts": n_prompts,
            "d_model": quality["lens"]["d_model"],
            "layers": quality["lens"]["layers"],
        },
        "quality_result": {
            "path": str(quality_result),
            "sha256": _sha256(quality_result),
            "status": quality["status"],
        },
        "fit_evidence": fit_evidence,
        "manual_review": {
            "reviewer": reviewer,
            "confirmation": confirmation,
        },
        "issued_at_utc": datetime.now(UTC).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def require_receipt(
    receipt_path: Path,
    *,
    lens: Path,
    checkpoint: str,
    checkpoint_revision: str,
    minimum_prompts: int,
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported scientific lens receipt schema")
    if receipt.get("scientifically_usable") is not True:
        raise ValueError("lens receipt does not approve scientific use")
    if receipt.get("decision") != "scientific_lens_approved":
        raise ValueError("lens receipt has the wrong decision")
    if receipt.get("checkpoint") != checkpoint:
        raise ValueError("lens receipt checkpoint mismatch")
    if receipt.get("checkpoint_revision") != checkpoint_revision:
        raise ValueError("lens receipt checkpoint revision mismatch")
    if receipt.get("lens", {}).get("sha256") != _sha256(lens):
        raise ValueError("lens receipt SHA-256 mismatch")
    if int(receipt.get("lens", {}).get("n_prompts", 0)) < minimum_prompts:
        raise ValueError("lens receipt has too few fitted prompts")
    evidence = receipt.get("fit_evidence", {})
    if evidence.get("kind") == "iterative_convergence":
        if evidence.get("worst_final_relative_change", 1.0) > evidence.get(
            "maximum_relative_change", 0.0
        ):
            raise ValueError("lens receipt convergence requirement failed")
    elif evidence.get("kind") == "exact_averaged_fit":
        if int(evidence.get("num_prompts", 0)) < minimum_prompts:
            raise ValueError("exact fit evidence has too few prompts")
    else:
        raise ValueError("lens receipt has unsupported fit evidence")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-result", required=True, type=Path)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--convergence", type=Path)
    evidence.add_argument("--fit-result", type=Path)
    parser.add_argument("--lens", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--minimum-prompts", type=int, default=1000)
    parser.add_argument("--maximum-relative-change", type=float, default=0.002)
    args = parser.parse_args()
    print(
        json.dumps(
            issue_receipt(
                quality_result=args.quality_result,
                convergence=args.convergence,
                fit_result=args.fit_result,
                lens=args.lens,
                output=args.output,
                reviewer=args.reviewer,
                confirmation=args.confirm,
                minimum_prompts=args.minimum_prompts,
                maximum_relative_change=args.maximum_relative_change,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
