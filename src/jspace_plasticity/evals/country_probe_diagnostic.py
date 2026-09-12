"""Evaluate output-protection evasion and linear country decodability.

Collection compares the clean base and recovered checkpoint under their own
fresh J-lenses.  For the recovered model it records both the ordinary dynamic
output-protection set and a protection set frozen from the clean base model.
Reduction fits fixed-regularization linear ridge probes with leave-one-city-fold
out evaluation; model weights are never updated.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jspace_plasticity.config import ModelConfig
from jspace_plasticity.intervention import (
    AblationPlan,
    JSpaceAblator,
    _hidden_from_output,
)
from jspace_plasticity.lens import LensMatrices
from jspace_plasticity.lens.fit_exact_dp import validate_model_manifest
from jspace_plasticity.modeling import load_model_and_tokenizer
from jspace_plasticity.synthetic_recovery_sft import (
    MODEL_REVISION,
    _encode,
    intervention_config,
    paired_difference,
    read_jsonl,
    sha256_path,
    write_json,
)

SCHEMA_VERSION = 1
WORKERS = 4
PROBE_LAYERS = (8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31)
PROBE_STATES = (
    "base_clean",
    "base_lesion",
    "primary_clean",
    "primary_current_protection",
    "primary_frozen_base_protection",
)
RIDGE_ALPHA = 1e-3


def _require_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256_path(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return observed


def load_design(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require_sha(path, expected_sha256, "country-probe design")
    design = json.loads(path.read_text(encoding="utf-8"))
    if design.get("decision") != "country_probe_and_frozen_protection_authorized":
        raise ValueError("design does not authorize this diagnostic")
    if design["checkpoint"]["revision"] != MODEL_REVISION:
        raise ValueError("model revision drifted")
    if design["probe"]["layers"] != list(PROBE_LAYERS):
        raise ValueError("probe layers drifted")
    if design["probe"]["states"] != list(PROBE_STATES):
        raise ValueError("probe states drifted")
    if design["probe"]["ridge_alpha"] != RIDGE_ALPHA:
        raise ValueError("probe regularization drifted")
    if design["evaluation"]["training"] is not False:
        raise ValueError("diagnostic must not update model weights")
    return design


def freeze_output_protection(
    plan: AblationPlan, base_blocked_token_ids: torch.Tensor
) -> AblationPlan:
    """Clone a plan while replacing only its output-token exemption set."""
    if plan.blocked_token_ids.shape != base_blocked_token_ids.shape:
        raise ValueError("base and recovered protection tensors differ in shape")
    return replace(
        plan,
        directions=plan.directions,
        selected_token_ids={},
        blocked_token_ids=base_blocked_token_ids.detach().clone(),
        ranked_token_ids=None,
        audit_top_k=None,
    )


@contextmanager
def _capture_layers(
    resolved: Any, layers: Sequence[int]
) -> Iterator[dict[int, torch.Tensor]]:
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            del module, inputs
            captured[layer] = _hidden_from_output(output).detach()

        return hook

    try:
        for layer in layers:
            handles.append(
                resolved.layers[layer].register_forward_hook(make_hook(layer))
            )
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def _forward_capture(
    model: Any,
    resolved: Any,
    encoded: dict[str, torch.Tensor],
    layers: Sequence[int],
    *,
    ablator: JSpaceAblator | None = None,
    plan: AblationPlan | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    if (ablator is None) != (plan is None):
        raise ValueError("ablator and plan must be supplied together")
    if ablator is None:
        with _capture_layers(resolved, layers) as captured:
            logits = model(**encoded, use_cache=False).logits[0, -1].float()
    else:
        # Lesion hooks register first, so capture observes post-lesion states.
        with ablator.apply(plan):
            with _capture_layers(resolved, layers) as captured:
                logits = model(**encoded, use_cache=False).logits[0, -1].float()
    missing = sorted(set(layers) - set(captured))
    if missing:
        raise RuntimeError(f"forward did not capture layers {missing}")
    values = torch.stack([captured[layer][0, -1].float().cpu() for layer in layers])
    return logits, values.numpy().astype(np.float16)


def _load_lens(
    design: dict[str, Any], key: str, *, expected_prompts: int
) -> LensMatrices:
    receipt = design["lenses"][key]
    path = Path(receipt["path"])
    _require_sha(path, receipt["sha256"], f"{key} lens")
    lens = LensMatrices.load(path)
    if lens.n_prompts != expected_prompts:
        raise ValueError(f"{key} lens has {lens.n_prompts} prompts")
    return lens


def _ablator(resolved: Any, lens: LensMatrices, path: str) -> JSpaceAblator:
    return JSpaceAblator(
        resolved,
        lens,
        intervention_config("jspace", Path(path)),
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )


def _model(
    path: str, revision: str, manifest: dict[str, str] | None
) -> tuple[Any, Any, Any]:
    if manifest is not None:
        validate_model_manifest(
            Path(manifest["path"]),
            expected_sha256=manifest["sha256"],
            model_dir=Path(path),
        )
    model, tokenizer, resolved = load_model_and_tokenizer(
        ModelConfig(
            name_or_path=path,
            revision=revision,
            dtype="float32",
            attn_implementation="sdpa",
            gradient_checkpointing=False,
            freeze_output_head=True,
        ),
        torch.device("cuda"),
    )
    model.eval()
    return model, tokenizer, resolved


def _semantic_token_ids(tokenizer: Any, text: str) -> set[int]:
    values = set()
    for rendered in (text, f" {text}"):
        values.update(
            int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)
        )
    return values


def _protection_audit(blocked: torch.Tensor, candidate_ids: set[int]) -> dict[str, Any]:
    final = {int(value) for value in blocked[0, -1].tolist()}
    anywhere = {int(value) for value in blocked.flatten().tolist()}
    return {
        "intermediate_protected_final": bool(final & candidate_ids),
        "intermediate_protected_any_position": bool(anywhere & candidate_ids),
        "protected_token_ids_final": sorted(final & candidate_ids),
        "protected_token_ids_any_position": sorted(anywhere & candidate_ids),
    }


def _load_heldout(design: dict[str, Any]) -> list[dict[str, Any]]:
    source = Path(design["cohorts"]["task_source"]["path"])
    _require_sha(source, design["cohorts"]["task_source"]["sha256"], "task source")
    by_id = {row["source_id"]: row for row in read_jsonl(source)}
    predictions = Path(design["cohorts"]["heldout_receipt"]["path"])
    _require_sha(
        predictions,
        design["cohorts"]["heldout_receipt"]["sha256"],
        "held-out receipt",
    )
    rows = []
    for receipt in read_jsonl(predictions):
        if (
            receipt["phase"] == "initial"
            and receipt["dataset"] == "synthetic_geo"
            and receipt["split"] in {"val", "screen"}
        ):
            source_row = by_id[receipt["source_id"]]
            rows.append(
                {
                    **source_row,
                    "expected_token_id": int(receipt["expected_token_id"]),
                }
            )
    rows.sort(key=lambda row: row["source_id"])
    if len(rows) != 129:
        raise ValueError(f"expected 129 held-out rows, got {len(rows)}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


@torch.inference_mode()
def collect(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    design = load_design(args.design, args.design_sha256)
    if not 0 <= args.worker_index < WORKERS:
        raise ValueError("worker index outside frozen worker count")
    if os.environ.get("EXPERIMENT_IMAGE") != args.experiment_image:
        raise ValueError("runtime image does not match immutable CLI image")
    checkpoint = design["checkpoint"]
    base, base_tokenizer, base_resolved = _model(
        checkpoint["base_path"], checkpoint["revision"], None
    )
    primary, primary_tokenizer, primary_resolved = _model(
        checkpoint["primary_path"],
        checkpoint["revision"],
        checkpoint["primary_manifest"],
    )
    if base_tokenizer.encode("A test") != primary_tokenizer.encode("A test"):
        raise ValueError("base and recovered tokenizers differ")

    base_lens = _load_lens(design, "base_fresh", expected_prompts=500)
    base_ablator = _ablator(
        base_resolved, base_lens, design["lenses"]["base_fresh"]["path"]
    )
    del base_lens
    primary_lens = _load_lens(design, "primary_fresh", expected_prompts=500)
    primary_ablator = _ablator(
        primary_resolved, primary_lens, design["lenses"]["primary_fresh"]["path"]
    )
    del primary_lens
    published_lens = _load_lens(design, "published", expected_prompts=1000)
    primary_published = _ablator(
        primary_resolved, published_lens, design["lenses"]["published"]["path"]
    )
    del published_lens

    probe_path = Path(design["cohorts"]["probe"]["path"])
    _require_sha(probe_path, design["cohorts"]["probe"]["sha256"], "probe corpus")
    probe_manifest = probe_path.with_suffix(probe_path.suffix + ".manifest.json")
    _require_sha(
        probe_manifest,
        design["cohorts"]["probe"]["manifest_sha256"],
        "probe manifest",
    )
    probe_rows = read_jsonl(probe_path)[args.worker_index :: WORKERS]
    activations: dict[str, list[np.ndarray]] = {state: [] for state in PROBE_STATES}
    probe_receipts = []
    for row in probe_rows:
        encoded = _encode(base_tokenizer, row["prompt"], base.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            base_plan = base_ablator.build_plan(
                base, encoded["input_ids"], encoded["attention_mask"]
            )
            _, values = _forward_capture(base, base_resolved, encoded, PROBE_LAYERS)
            activations["base_clean"].append(values)
            _, values = _forward_capture(
                base,
                base_resolved,
                encoded,
                PROBE_LAYERS,
                ablator=base_ablator,
                plan=base_plan,
            )
            activations["base_lesion"].append(values)
            primary_plan = primary_ablator.build_plan(
                primary, encoded["input_ids"], encoded["attention_mask"]
            )
            _, values = _forward_capture(
                primary, primary_resolved, encoded, PROBE_LAYERS
            )
            activations["primary_clean"].append(values)
            _, values = _forward_capture(
                primary,
                primary_resolved,
                encoded,
                PROBE_LAYERS,
                ablator=primary_ablator,
                plan=primary_plan,
            )
            activations["primary_current_protection"].append(values)
            frozen = freeze_output_protection(primary_plan, base_plan.blocked_token_ids)
            _, values = _forward_capture(
                primary,
                primary_resolved,
                encoded,
                PROBE_LAYERS,
                ablator=primary_ablator,
                plan=frozen,
            )
            activations["primary_frozen_base_protection"].append(values)
        probe_receipts.append(
            {
                "source_id": row["source_id"],
                "cluster_id": row["cluster_id"],
                "city_fold": int(row["city_fold"]),
                "relation": row["relation"],
            }
        )

    heldout = _load_heldout(design)[args.worker_index :: WORKERS]
    output_rows = []
    for row in heldout:
        encoded = _encode(base_tokenizer, row["prompt"], base.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            base_plan = base_ablator.build_plan(
                base, encoded["input_ids"], encoded["attention_mask"]
            )
            own_plan = primary_ablator.build_plan(
                primary, encoded["input_ids"], encoded["attention_mask"]
            )
            own_clean = own_plan.clean_next_logits[0].float()
            with primary_ablator.apply(own_plan):
                own_logits = primary(**encoded, use_cache=False).logits[0, -1].float()
            own_frozen = freeze_output_protection(own_plan, base_plan.blocked_token_ids)
            with primary_ablator.apply(own_frozen):
                own_frozen_logits = (
                    primary(**encoded, use_cache=False).logits[0, -1].float()
                )
            published_plan = primary_published.build_plan(
                primary, encoded["input_ids"], encoded["attention_mask"]
            )
            with primary_published.apply(published_plan):
                published_logits = (
                    primary(**encoded, use_cache=False).logits[0, -1].float()
                )
            published_frozen = freeze_output_protection(
                published_plan, base_plan.blocked_token_ids
            )
            with primary_published.apply(published_frozen):
                published_frozen_logits = (
                    primary(**encoded, use_cache=False).logits[0, -1].float()
                )
        target = int(row["expected_token_id"])
        candidates = _semantic_token_ids(base_tokenizer, row["intermediate"])
        output_rows.append(
            {
                "source_id": row["source_id"],
                "relation": row["relation"],
                "cluster_id": row["cluster_id"],
                "expected_token_id": target,
                "clean_exact": int(int(own_clean.argmax()) == target),
                "own_fresh_current_exact": int(int(own_logits.argmax()) == target),
                "own_fresh_frozen_exact": int(
                    int(own_frozen_logits.argmax()) == target
                ),
                "published_current_exact": int(
                    int(published_logits.argmax()) == target
                ),
                "published_frozen_exact": int(
                    int(published_frozen_logits.argmax()) == target
                ),
                "base_protection": _protection_audit(
                    base_plan.blocked_token_ids, candidates
                ),
                "recovered_protection": _protection_audit(
                    own_plan.blocked_token_ids, candidates
                ),
            }
        )

    args.output_dir.mkdir(parents=True)
    activation_path = args.output_dir / "probe_activations.npz"
    payload: dict[str, np.ndarray] = {
        "layers": np.asarray(PROBE_LAYERS, dtype=np.int64),
        "source_ids": np.asarray([row["source_id"] for row in probe_receipts]),
        "cluster_ids": np.asarray([row["cluster_id"] for row in probe_receipts]),
        "city_folds": np.asarray([row["city_fold"] for row in probe_receipts]),
        "relations": np.asarray([row["relation"] for row in probe_receipts]),
    }
    for state, values in activations.items():
        payload[state] = np.stack(values)
    np.savez_compressed(activation_path, **payload)
    output_path = args.output_dir / "output_protection.jsonl"
    _write_jsonl(output_path, output_rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "worker_index": args.worker_index,
        "probe_rows": len(probe_rows),
        "output_rows": len(output_rows),
        "probe_activations_sha256": sha256_path(activation_path),
        "output_protection_sha256": sha256_path(output_path),
    }
    write_json(args.output_dir / "result.json", result)
    return result


def _linear_probe_fold(
    x: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    test_fold: int,
    *,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = folds != test_fold
    test = ~train
    x_train = torch.from_numpy(x[train]).float().to(device)
    x_test = torch.from_numpy(x[test]).float().to(device)
    mean = x_train.mean(dim=0, keepdim=True)
    scale = x_train.std(dim=0, keepdim=True).clamp_min(1e-4)
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    x_train = torch.nn.functional.normalize(x_train, dim=1)
    x_test = torch.nn.functional.normalize(x_test, dim=1)
    x_train = torch.cat((x_train, torch.ones_like(x_train[:, :1])), dim=1)
    x_test = torch.cat((x_test, torch.ones_like(x_test[:, :1])), dim=1)
    n_classes = int(labels.max()) + 1
    y = torch.nn.functional.one_hot(
        torch.from_numpy(labels[train]).long().to(device), n_classes
    ).float()
    gram = x_train @ x_train.T
    gram.diagonal().add_(RIDGE_ALPHA)
    weights = x_train.T @ torch.linalg.solve(gram, y)
    scores = x_test @ weights
    return labels[test], scores.cpu().numpy()


def linear_probe_scores(
    x: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    device: torch.device | None = None,
) -> dict[str, float]:
    gold = []
    scores = []
    for fold in range(4):
        fold_gold, fold_scores = _linear_probe_fold(
            x, labels, folds, fold, device=device
        )
        gold.append(fold_gold)
        scores.append(fold_scores)
    gold_array = np.concatenate(gold)
    score_array = np.concatenate(scores)
    predicted = score_array.argmax(axis=1)
    top5 = np.argpartition(score_array, -5, axis=1)[:, -5:]
    return {
        "rows": int(len(gold_array)),
        "accuracy": float(np.mean(predicted == gold_array)),
        "top5_accuracy": float(np.mean(np.any(top5 == gold_array[:, None], axis=1))),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_probe(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    labels = {
        "base_clean": "Base clean",
        "base_lesion": "Base lesion",
        "primary_clean": "Recovered clean",
        "primary_current_protection": "Recovered lesion, current protection",
        "primary_frozen_base_protection": "Recovered lesion, frozen protection",
    }
    for state in PROBE_STATES:
        selected = [row for row in rows if row["state"] == state]
        axis.plot(
            [row["layer"] for row in selected],
            [row["accuracy"] for row in selected],
            marker="o",
            label=labels[state],
        )
    axis.axvspan(16, 22, color="#d62728", alpha=0.08, label="Lesion band")
    axis.axhline(1 / 51, color="black", linestyle="--", linewidth=1, label="Chance")
    axis.set(
        xlabel="Decoder block",
        ylabel="City-disjoint country probe accuracy",
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def reduce_outputs(args: argparse.Namespace) -> dict[str, Any]:
    design = load_design(args.design, args.design_sha256)
    worker_root = args.output_dir / "workers"
    payloads = []
    output_rows = []
    for index in range(WORKERS):
        worker = worker_root / f"worker-{index:02d}"
        result = json.loads((worker / "result.json").read_text(encoding="utf-8"))
        _require_sha(
            worker / "probe_activations.npz",
            result["probe_activations_sha256"],
            f"worker {index} probe activations",
        )
        _require_sha(
            worker / "output_protection.jsonl",
            result["output_protection_sha256"],
            f"worker {index} output protection",
        )
        payloads.append(np.load(worker / "probe_activations.npz"))
        output_rows.extend(read_jsonl(worker / "output_protection.jsonl"))

    source_ids = np.concatenate([payload["source_ids"] for payload in payloads])
    order = np.argsort(source_ids)
    source_ids = source_ids[order]
    if len(source_ids) != 612 or len(set(source_ids.tolist())) != 612:
        raise ValueError("probe rows are incomplete or duplicated")
    cluster_ids = np.concatenate([payload["cluster_ids"] for payload in payloads])[
        order
    ]
    folds = np.concatenate([payload["city_folds"] for payload in payloads])[order]
    classes = {
        value: index for index, value in enumerate(sorted(set(cluster_ids.tolist())))
    }
    labels = np.asarray([classes[value] for value in cluster_ids], dtype=np.int64)
    layers = payloads[0]["layers"].tolist()
    probe_rows = []
    for state in PROBE_STATES:
        values = np.concatenate([payload[state] for payload in payloads], axis=0)[order]
        for layer_index, layer in enumerate(layers):
            metrics = linear_probe_scores(values[:, layer_index], labels, folds)
            probe_rows.append({"state": state, "layer": int(layer), **metrics})
    probe_path = args.output_dir / "linear_country_probe.csv"
    _write_csv(probe_path, probe_rows)
    probe_figure = args.output_dir / "linear_country_probe.png"
    _plot_probe(probe_rows, probe_figure)

    output_rows.sort(key=lambda row: row["source_id"])
    if len(output_rows) != 129 or len({row["source_id"] for row in output_rows}) != 129:
        raise ValueError("output-protection rows are incomplete or duplicated")
    output_path = args.output_dir / "output_protection_predictions.jsonl"
    _write_jsonl(output_path, output_rows)
    conditions = (
        "clean_exact",
        "own_fresh_current_exact",
        "own_fresh_frozen_exact",
        "published_current_exact",
        "published_frozen_exact",
    )
    accuracy = {
        condition: sum(int(row[condition]) for row in output_rows) / len(output_rows)
        for condition in conditions
    }
    by_relation: dict[str, dict[str, float]] = {}
    for relation in sorted({row["relation"] for row in output_rows}):
        selected = [row for row in output_rows if row["relation"] == relation]
        by_relation[relation] = {
            "rows": len(selected),
            **{
                condition: sum(int(row[condition]) for row in selected) / len(selected)
                for condition in conditions
            },
        }
    protection = {}
    for key in ("base_protection", "recovered_protection"):
        protection[key] = {
            field: sum(bool(row[key][field]) for row in output_rows) / len(output_rows)
            for field in (
                "intermediate_protected_final",
                "intermediate_protected_any_position",
            )
        }

    def compare(left_field: str, right_field: str) -> dict[str, Any]:
        left = [
            {"source_id": row["source_id"], "exact": row[left_field]}
            for row in output_rows
        ]
        right = [
            {"source_id": row["source_id"], "exact": row[right_field]}
            for row in output_rows
        ]
        return paired_difference(left, right, field="exact")

    paired = {
        "own_fresh_frozen_minus_current": compare(
            "own_fresh_frozen_exact", "own_fresh_current_exact"
        ),
        "published_frozen_minus_current": compare(
            "published_frozen_exact", "published_current_exact"
        ),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "training": False,
        "probe": {
            "classes": len(classes),
            "rows": len(source_ids),
            "outer_split": "leave-one-city-fold-out",
            "ridge_alpha": RIDGE_ALPHA,
            "chance_accuracy": 1 / len(classes),
            "csv_sha256": sha256_path(probe_path),
            "figure_sha256": sha256_path(probe_figure),
        },
        "output_protection": {
            "rows": len(output_rows),
            "accuracy": accuracy,
            "by_relation": by_relation,
            "intermediate_protection_fraction": protection,
            "paired": paired,
            "predictions_sha256": sha256_path(output_path),
        },
        "interpretation": design["interpretation"],
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--design-sha256", required=True)
    parser.add_argument("--experiment-image", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--reduce-only", action="store_true")
    args = parser.parse_args()
    if args.reduce_only == (args.worker_index is not None):
        parser.error("provide exactly one of --worker-index or --reduce-only")
    return args


def main() -> None:
    args = parse_args()
    result = reduce_outputs(args) if args.reduce_only else collect(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
