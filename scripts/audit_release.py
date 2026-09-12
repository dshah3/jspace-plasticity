"""Recompute both completed result audits with CPU-only dependencies."""

import argparse
from pathlib import Path

from audit_final_capability import audit as audit_training
from audit_final_fresh_lens import audit as audit_fresh

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, default=Path(".audit-output"))
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
a.output.mkdir(parents=True, exist_ok=True)
audit_training(
    root / "results/final-training",
    root / "data/evals/q35-final-capability-20260905.json",
    root / "data/cohort/eligible_train_rows.jsonl",
    root / "results/calibration",
    a.output / "training-audit.json",
)
audit_fresh(
    root / "results/fresh-lens",
    root / "results/final-training",
    root / "data/evals/q35-final-fresh-lens-20260905.json",
    a.output / "fresh-lens-audit.json",
)
print("Both CPU receipt audits passed.")
