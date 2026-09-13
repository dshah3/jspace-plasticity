# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==2.5.2",
# ]
# ///
"""CPU receipt audit of the final fixed-LR experiment; no model execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np

CONDITIONS = ("clean", "jspace", "random")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_rows(rows):
    keys = [(r["dataset"], r["source_id"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate prediction IDs")
    return dict(zip(keys, rows, strict=True))


def paired_clusters(left, right, field="jspace_exact"):
    """Paired left-minus-right bootstrap, resampling entire country clusters."""
    left, right = index_rows(left), index_rows(right)
    if left.keys() != right.keys():
        raise ValueError("Paired IDs differ")
    groups = {}
    for key, row in left.items():
        assert row["cluster_id"] == right[key]["cluster_id"]
        groups.setdefault(row["cluster_id"], []).append(row[field] - right[key][field])
    groups = dict(sorted(groups.items()))
    sums = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    draws = np.random.default_rng(20260905).integers(
        0, len(groups), (10000, len(groups))
    )
    boot = sums[draws].sum(1) / sizes[draws].sum(1)
    macro = (sums / sizes)[draws].mean(1)
    return dict(
        rows=len(left),
        clusters=len(groups),
        prompt_weighted_difference=float(sums.sum() / sizes.sum()),
        equal_cluster_difference=float((sums / sizes).mean()),
        prompt_weighted_95_percentile=np.quantile(boot, [0.025, 0.975]).tolist(),
        equal_cluster_95_percentile=np.quantile(macro, [0.025, 0.975]).tolist(),
    )


def select(rows, phase, cohort):
    splits = {
        "pooled": {"val", "screen"},
        "val": {"val"},
        "screen": {"screen"},
        "transfer": {"transfer"},
        "train": {"train"},
    }[cohort]
    return [r for r in rows if r["phase"] == phase and r["split"] in splits]


def audit(run, design_path, eligible_path, calibration, output):
    design = json.loads(design_path.read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert (run / "_SUCCESS").exists()
    assert summary["design_sha256"] == sha(design_path)
    assert (run / "SUMMARY.sha256").read_text().split()[0] == sha(run / "summary.json")
    eligible = read_jsonl(eligible_path)
    assert sha(eligible_path) == design["evidence"]["eligible_train_rows"]["sha256"]
    eligible_ids = [r["source_id"] for r in eligible]
    assert len(set(eligible_ids)) == 189
    train_clusters = {r["cluster_id"] for r in eligible}
    arms, raw, reference, inventory = {}, {}, None, None
    for arm in design["arms"]:
        arm_dir = run / "arms" / f"arm-{arm['index']:02d}-{arm['name']}"
        result = json.loads((arm_dir / "result.json").read_text())
        assert result["arm"] == arm and result["status"] == "completed"
        assert result["design_sha256"] == sha(design_path)
        assert result["experiment_image"] == summary["experiment_image"]
        assert result["direction_convention"] == "effective_gain"
        assert sha(arm_dir / "predictions.jsonl") == result["predictions_sha256"]
        matches = [
            h
            for p, h in summary["result_sha256"].items()
            if Path(p).parent.name == arm_dir.name
        ]
        assert matches == [sha(arm_dir / "result.json")]
        rows = read_jsonl(arm_dir / "predictions.jsonl")
        assert len(rows) == 545
        for r in rows:
            for cond in CONDITIONS:
                assert r[cond + "_exact"] == int(
                    r[cond + "_predicted_token_id"] == r["expected_token_id"]
                )
        initial = index_rows([r for r in rows if r["phase"] == "initial"])
        canonical = {
            k: tuple(
                v[f]
                for f in (
                    "expected_token_id",
                    "clean_predicted_token_id",
                    "jspace_predicted_token_id",
                    "random_predicted_token_id",
                )
            )
            for k, v in initial.items()
        }
        if reference is None:
            reference = canonical
        assert canonical == reference
        runtime = result["runtime"]
        assert runtime["trainable_parameters"] == 3570049536
        assert runtime["gradient_checkpointing"] is False
        assert (
            runtime["final_norm_trainable"] is False
            and runtime["lm_head_trainable"] is False
        )
        if inventory is None:
            inventory = runtime["trainable_parameter_inventory"]
        assert runtime["trainable_parameter_inventory"] == inventory
        grads = json.loads((arm_dir / "gradient-inventory.json").read_text())
        assert len(grads["parameters"]) == 424 and all(
            p["has_grad"] for p in grads["parameters"]
        )
        assert grads["foreach"] is False
        with (arm_dir / "metrics.csv").open() as f:
            metrics = list(csv.DictReader(f))
        assert [int(m["step"]) for m in metrics] == list(range(1, 101))
        assert all(
            np.isfinite(float(m["loss"]))
            and np.isfinite(float(m["grad_norm_pre_clip"]))
            for m in metrics
        )
        assert all(
            np.isclose(
                float(m["learning_rate"]),
                1e-6 * min(1, (i + 1) / 5),
                rtol=1e-12,
                atol=0,
            )
            for i, m in enumerate(metrics)
        )
        seen = [s for m in metrics for s in json.loads(m["source_ids"])]
        expected = []
        for epoch in range(3):
            order = eligible_ids.copy()
            random.Random(arm["seed"] + epoch * 100003).shuffle(order)
            expected.extend(order)
        assert seen == expected[:400]
        counts, changes = {}, {}
        for cohort, n in [
            ("val", 64),
            ("screen", 65),
            ("pooled", 129),
            ("transfer", 49),
        ]:
            before, after = (
                select(rows, "initial", cohort),
                select(rows, "terminal", cohort),
            )
            assert len(before) == len(after) == n
            if cohort != "transfer":
                assert not ({r["cluster_id"] for r in after} & train_clusters)
            counts[cohort] = {
                "rows": n,
                "clusters": len({r["cluster_id"] for r in after}),
                "initial": {
                    c: sum(r[c + "_exact"] for r in before) for c in CONDITIONS
                },
                "terminal": {
                    c: sum(r[c + "_exact"] for r in after) for c in CONDITIONS
                },
            }
            changes[cohort] = paired_clusters(after, before)
        train = select(rows, "terminal", "train")
        assert set(r["source_id"] for r in train) == set(eligible_ids)
        counts["train"] = {
            "rows": len(train),
            "terminal": {c: sum(r[c + "_exact"] for r in train) for c in CONDITIONS},
        }
        relations = {}
        for cohort in ("pooled", "screen", "val"):
            relations[cohort] = {}
            for relation in sorted(
                {r["relation"] for r in select(rows, "terminal", cohort)}
            ):
                subset = [
                    r
                    for r in select(rows, "terminal", cohort)
                    if r["relation"] == relation
                ]
                relations[cohort][relation] = {
                    "rows": len(subset),
                    **{c: sum(r[c + "_exact"] for r in subset) for c in CONDITIONS},
                }
        arms[arm["name"]] = {
            "counts": counts,
            "paired_cluster_changes": changes,
            "relations": relations,
            "runtime": {
                k: v for k, v in runtime.items() if k != "trainable_parameter_inventory"
            },
            "gradient_tensors": 424,
            "missing_gradients": 0,
            "training_order_matches": True,
            "memory": result["memory"],
            "result_sha256": sha(arm_dir / "result.json"),
        }
        raw[arm["name"]] = rows
    primary = design["evaluation"]["primary_arm"]
    differences = {}
    for arm in design["arms"][1:]:
        name = arm["name"]
        differences[name] = {
            c: paired_clusters(
                select(raw[primary], "terminal", c), select(raw[name], "terminal", c)
            )
            for c in ("val", "screen", "pooled", "transfer")
        }
    cal_path = (
        calibration / "arms/arm-00-j-full-lr1e-6/milestones/step-0100/predictions.jsonl"
    )
    cal = index_rows(read_jsonl(cal_path))
    final = index_rows(select(raw[primary], "terminal", "val"))
    assert cal.keys() == final.keys()
    comparison = {
        c: [
            k[1]
            for k in cal
            if cal[k][c + "_predicted_token_id"] != final[k][c + "_predicted_token_id"]
        ]
        for c in CONDITIONS
    }
    out = {
        "status": "receipt_audit_passed",
        "summary_sha256": sha(run / "summary.json"),
        "checked_prediction_rows": sum(len(v) for v in raw.values()),
        "arms": arms,
        "primary_minus_other_arms": differences,
        "calibration_vs_rerun_validation_token_mismatches": comparison,
        "calibration_prediction_sha256": sha(cal_path),
        "bootstrap_seed": 20260905,
        "bootstrap_replicates": 10000,
        "limitations": [
            "Saved-token consistency is not independent GPU re-execution.",
            "Exploratory country bootstrap; prior screening and LR selection remain.",
            "Transfer clusters are saved task IDs, not independent-country claims.",
            "Published base lens only; no fresh-lens or orthogonal-span final results.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")
    for name, a in arms.items():
        print(name, json.dumps(a["counts"]))
    print(
        "calibration rerun mismatch counts", {c: len(v) for c, v in comparison.items()}
    )
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--eligible", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.run, a.design, a.eligible, a.calibration, a.output)
