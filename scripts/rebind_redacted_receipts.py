# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Rebind receipt-to-receipt hashes after the publication redaction pass.

Publishing this experiment required replacing the private container registry,
shared-storage mount and home directory with placeholders inside design files,
run summaries and per-arm ``result.json`` receipts. Those files carry SHA-256
bindings to each other, so redacting their bytes invalidated the chain.

This script recomputes *only* the derived metadata bindings:

  * ``design_sha256`` in each run summary and per-arm/-condition result
  * ``SUMMARY.sha256`` sidecars
  * ``result_sha256`` maps inside run summaries
  * ``fit_result_sha256`` for fresh-lens fit stages

It never touches ``predictions_sha256``, and it never touches a
``predictions.jsonl``. Those files were unaffected by redaction and keep their
original hashes, so the per-example predictions remain provably identical to
the bytes produced by the original runs. No numerical value is modified.

Idempotent: rerunning it on an already-rebound tree makes no changes.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict, dry_run: bool) -> None:
    if not dry_run:
        path.write_text(json.dumps(payload, indent=2) + "\n")


def rebind_run(run: Path, design: Path, changes: list[str], dry_run: bool) -> None:
    """Rebind one run directory against its (redacted) design file."""
    design_hash = sha(design)
    summary_path = run / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())

    if summary.get("design_sha256") != design_hash:
        summary["design_sha256"] = design_hash
        changes.append(f"{summary_path.relative_to(ROOT)}: design_sha256")

    # Per-arm / per-condition receipts bind the design too.
    for key in ("arms", "conditions"):
        for child in sorted((run / key).glob("*")) if (run / key).is_dir() else []:
            result_path = child / "result.json"
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text())
            if result.get("design_sha256") != design_hash:
                result["design_sha256"] = design_hash
                write_json(result_path, result, dry_run)
                changes.append(f"{result_path.relative_to(ROOT)}: design_sha256")

    # A fresh-lens condition binds the hash of the 500-prompt fit that
    # produced its lens. Do this after design_sha256 above, since both live in
    # the same result.json and the summary's result_sha256 map below covers it.
    for fit_result in sorted(run.glob("fits/*/stages/0500/result.json")):
        model = fit_result.parents[2].name
        condition_path = run / "conditions" / f"{model}-{model}_fresh" / "result.json"
        if not condition_path.exists():
            continue
        condition = json.loads(condition_path.read_text())
        stage_hash = sha(fit_result)
        if condition.get("lens", {}).get("fit_result_sha256") != stage_hash:
            condition["lens"]["fit_result_sha256"] = stage_hash
            write_json(condition_path, condition, dry_run)
            changes.append(
                f"{condition_path.relative_to(ROOT)}: lens.fit_result_sha256"
            )

    # result_sha256 maps are keyed either by the original absolute result.json
    # path (training runs) or by a bare condition name (fresh-lens runs). Both
    # resolve through the leaf directory name, which is what the audit matches on.
    result_map = summary.get("result_sha256")
    if isinstance(result_map, dict):
        for key in list(result_map):
            leaf = Path(key).parent.name if key.endswith("result.json") else key
            options = (run / "arms" / leaf, run / "conditions" / leaf, run / leaf)
            candidate = next(
                (c for c in options if (c / "result.json").exists()), None
            )
            if candidate is None:
                continue
            actual = sha(candidate / "result.json")
            if result_map[key] != actual:
                result_map[key] = actual
                changes.append(
                    f"{summary_path.relative_to(ROOT)}: result_sha256[{leaf}]"
                )

    write_json(summary_path, summary, dry_run)

    # The sidecar must follow the summary it covers.
    sidecar = run / "SUMMARY.sha256"
    if sidecar.exists():
        summary_hash = sha(summary_path)
        if sidecar.read_text().split()[0] != summary_hash:
            if not dry_run:
                sidecar.write_text(f"{summary_hash}  summary.json\n")
            changes.append(f"{sidecar.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would change without writing",
    )
    args = parser.parse_args()

    evals = ROOT / "data/evals"
    runs = [
        (ROOT / "results/final-training", evals / "q35-final-capability-20260905.json"),
        (ROOT / "results/fresh-lens", evals / "q35-final-fresh-lens-20260905.json"),
    ]

    changes: list[str] = []
    for run, design in runs:
        if run.exists() and design.exists():
            rebind_run(run, design, changes, args.check)

    if not changes:
        print("Receipt bindings already consistent; nothing to rebind.")
        return

    verb = "Would rebind" if args.check else "Rebound"
    print(f"{verb} {len(changes)} binding(s):")
    for change in changes:
        print(f"  {change}")


if __name__ == "__main__":
    main()
