"""CPU-only audit and probe refit; never loads model weights or submits jobs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from jspace_plasticity.evals.country_probe_diagnostic import (
    PROBE_STATES,
    _linear_probe_fold,
)
from jspace_plasticity.synthetic_recovery_sft import read_jsonl, select_training_rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, output):
    source = read_jsonl(Path("data/closedbook/closedbook-geo-s20260825.jsonl"))
    eligible = read_jsonl(
        root / "q35-synthetic-expansion-triage-dp6-r1/eligible_train_rows.jsonl"
    )
    arms = []
    for arm in sorted((root / "q35-synthetic-recovery-dp8-r1/arms").iterdir()):
        result = json.loads((arm / "result.json").read_text())
        rows = read_jsonl(arm / "predictions.jsonl")
        assert sha(arm / "predictions.jsonl") == result["predictions_sha256"]
        for row in rows:
            for condition in ("clean", "jspace", "random"):
                assert row[condition + "_exact"] == int(
                    row[condition + "_predicted_token_id"] == row["expected_token_id"]
                )
        config = result["arm"]
        selected, _ = select_training_rows(source, eligible, config["data_fraction"])
        with (arm / "metrics.csv").open() as f:
            seen = [
                s for row in csv.DictReader(f) for s in json.loads(row["source_ids"])
            ]
        expected = []
        epoch = 0
        while len(expected) < len(seen):
            order = selected.copy()
            random.Random(config["seed"] + epoch * 100_003).shuffle(order)
            expected.extend(row["source_id"] for row in order)
            epoch += 1
        mismatches = [
            i for i, (a, b) in enumerate(zip(seen, expected, strict=False)) if a != b
        ]
        initial = {
            r["source_id"]: r
            for r in rows
            if r["phase"] == "initial"
            and r["dataset"] == "synthetic_geo"
            and r["split"] in ("val", "screen")
        }
        terminal = {
            r["source_id"]: r
            for r in rows
            if r["phase"] == "terminal"
            and r["dataset"] == "synthetic_geo"
            and r["split"] in ("val", "screen")
        }
        assert initial.keys() == terminal.keys()
        clusters = sorted({r["cluster_id"] for r in initial.values()})
        # Resample whole countries; retain within-country dependence and report
        # both prompt-weighted and equal-country estimands.
        sums, sizes = [], []
        for cluster in clusters:
            ids = [i for i, r in initial.items() if r["cluster_id"] == cluster]
            sums.append(
                sum(
                    terminal[i]["jspace_exact"] - initial[i]["jspace_exact"]
                    for i in ids
                )
            )
            sizes.append(len(ids))
        sums, sizes = np.asarray(sums), np.asarray(sizes)
        draws = np.random.default_rng(20260905).integers(
            0, len(clusters), (10000, len(clusters))
        )
        boot = sums[draws].sum(1) / sizes[draws].sum(1)
        macro = (sums / sizes)[draws].mean(1)
        arms.append(
            {
                "arm": arm.name,
                "result_sha256": sha(arm / "result.json"),
                "runtime": result["runtime"],
                "memory": result["memory"],
                "logged_examples": len(seen),
                "order_first_mismatch": mismatches[0] if mismatches else None,
                "initial_correct": sum(r["jspace_exact"] for r in initial.values()),
                "terminal_correct": sum(r["jspace_exact"] for r in terminal.values()),
                "rows": len(initial),
                "countries": len(clusters),
                "prompt_weighted_gain": float(sums.sum() / sizes.sum()),
                "country_bootstrap_95_percentile": np.quantile(
                    boot, [0.025, 0.975]
                ).tolist(),
                "equal_country_gain": float((sums / sizes).mean()),
                "equal_country_95_percentile": np.quantile(
                    macro, [0.025, 0.975]
                ).tolist(),
            }
        )
    joined = [
        {
            **r,
            "expected_token_id": next(
                e["expected_token_id"]
                for e in eligible
                if e["source_id"] == r["source_id"]
            ),
        }
        for r in source
        if r["source_id"] in {e["source_id"] for e in eligible}
    ]
    (output / "joined_eligible_train.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in joined)
    )
    (output / "behavior_and_order.json").write_text(
        json.dumps(
            {
                "arms": arms,
                "bootstrap_seed": 20260905,
                "bootstrap_replicates": 10000,
                "claim_boundary": (
                    "Exploratory screen+validation cohort; bootstrap does "
                    "not undo selection bias."
                ),
            },
            indent=2,
        )
        + "\n"
    )


def refit(root, probe_dir, output):
    torch.set_num_threads(4)
    payloads, provenance = [], []
    for index in range(4):
        worker = f"worker-{index:02d}"
        path = probe_dir / "workers" / worker / "probe_activations.npz"
        result = json.loads(
            (
                root / "q35-country-probe-dp4-r1/workers" / worker / "result.json"
            ).read_text()
        )
        assert sha(path) == result["probe_activations_sha256"]
        payloads.append(np.load(path))
        provenance.append({"worker": worker, "activation_sha256": sha(path)})
    source_ids = np.concatenate([p["source_ids"] for p in payloads])
    order = np.argsort(source_ids)
    source_ids = source_ids[order]
    assert len(set(source_ids)) == len(source_ids) == 612
    clusters = np.concatenate([p["cluster_ids"] for p in payloads])[order]
    folds = np.concatenate([p["city_folds"] for p in payloads])[order]
    classes = sorted(set(clusters.tolist()))
    labels = np.asarray([classes.index(c) for c in clusters])
    layers = payloads[0]["layers"].tolist()
    assert all(p["layers"].tolist() == layers for p in payloads)
    summaries = []
    with (output / "probe_fold_predictions.jsonl").open("w") as f:
        for state in PROBE_STATES:
            values = np.concatenate([p[state] for p in payloads])[order]
            for li, layer in enumerate(layers):
                correct, top5_correct = 0, 0
                for fold in range(4):
                    gold, scores = _linear_probe_fold(
                        values[:, li], labels, folds, fold, device=torch.device("cpu")
                    )
                    ids = source_ids[folds == fold]
                    predicted = scores.argmax(1)
                    top5 = np.argpartition(scores, -5, axis=1)[:, -5:]
                    correct += int((gold == predicted).sum())
                    top5_correct += int((top5 == gold[:, None]).any(1).sum())
                    for sid, g, p, t5 in zip(ids, gold, predicted, top5, strict=True):
                        f.write(
                            json.dumps(
                                {
                                    "state": state,
                                    "layer": layer,
                                    "fold": fold,
                                    "source_id": str(sid),
                                    "gold_class": int(g),
                                    "predicted_class": int(p),
                                    "top5_classes": [int(t) for t in t5],
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                summaries.append(
                    {
                        "state": state,
                        "layer": layer,
                        "rows": 612,
                        "accuracy": correct / 612,
                        "top5_accuracy": top5_correct / 612,
                    }
                )
    historical_path = root / "q35-country-probe-dp4-r1/linear_country_probe.csv"
    with historical_path.open() as f:
        historical = {(r["state"], int(r["layer"])): r for r in csv.DictReader(f)}
    differences = [
        {
            **r,
            "historical_accuracy": float(
                historical[r["state"], r["layer"]]["accuracy"]
            ),
        }
        for r in summaries
        if abs(r["accuracy"] - float(historical[r["state"], r["layer"]]["accuracy"]))
        > 1e-10
    ]
    (output / "probe_refit.json").write_text(
        json.dumps(
            {
                "device": "cpu",
                "torch": torch.__version__,
                "classes": classes,
                "layers": layers,
                "states": PROBE_STATES,
                "inputs": provenance,
                "summaries": summaries,
                "accuracy_differences": differences,
                "predictions_sha256": sha(output / "probe_fold_predictions.jsonl"),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Refit {len(summaries)} probe accuracies; "
        f"{len(differences)} differ from historical summaries."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit(args.receipts, args.output_dir)
    if args.probe_dir:
        refit(args.receipts, args.probe_dir, args.output_dir)


if __name__ == "__main__":
    main()
