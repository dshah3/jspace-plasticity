"""Build activation-conditioned figures for the Qwen3.5-4B recovery result.

This evaluator never trains.  Three GPU workers measure the clean base, primary
recovered, and high-dose recovered checkpoints on the same frozen 129-prompt
synthetic cohort.  Each checkpoint is crossed with its own exact 500-prompt
fresh lens, both with and without the original online-current k=10 lesion.

The reducer combines those new activation measurements with the already frozen
rank-cutoff sweep.  Every source JSON and prediction file is hash checked before
it contributes a plotted number.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from jspace_plasticity.config import ModelConfig
from jspace_plasticity.evals.recovery_rank_cutoff_sweep import (
    EXPECTED_K_VALUES,
    _intervention_config,
    _lens_receipt,
    _load_rows,
    _model_receipt,
)
from jspace_plasticity.evals.recovery_rank_cutoff_sweep import (
    load_design as load_rank_design,
)
from jspace_plasticity.intervention import JSpaceAblator, _hidden_from_output
from jspace_plasticity.lens import LensMatrices
from jspace_plasticity.lens.geometry import excess_kurtosis, linear_cka
from jspace_plasticity.modeling import load_model_and_tokenizer
from jspace_plasticity.synthetic_recovery_sft import (
    LESION_LAYERS,
    _encode,
    read_jsonl,
    sha256_path,
    write_json,
)

SCHEMA_VERSION = 1
EXPECTED_MODELS = ("base", "primary", "high_dose")
MODEL_LABELS = {
    "base": "Base",
    "primary": "Recovered (primary)",
    "high_dose": "Recovered (high dose)",
}
MODEL_COLORS = {
    "base": "#4C78A8",
    "primary": "#F58518",
    "high_dose": "#54A24B",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256_path(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return observed


def _expected_conditions() -> list[dict[str, Any]]:
    lenses = {
        "base": "base_fresh",
        "primary": "primary_fresh",
        "high_dose": "high_fresh",
    }
    return [
        {
            "index": index,
            "name": model.replace("_dose", ""),
            "model": model,
            "lens": lenses[model],
        }
        for index, model in enumerate(EXPECTED_MODELS)
    ]


def load_design(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_sha(path, expected_sha256, "figure-suite design")
    design = json.loads(path.read_text(encoding="utf-8"))
    if design.get("decision") != "posthoc_recovery_figure_suite_authorized":
        raise ValueError("design does not authorize the recovery figure suite")
    if design.get("conditions") != _expected_conditions():
        raise ValueError("figure-suite conditions drifted")
    measurement = design.get("measurement", {})
    if measurement != {
        "cohort": "synthetic_geo",
        "rows": 129,
        "layers": LESION_LAYERS,
        "lesion_k": 10,
        "activation_position": "final_prompt_token",
        "readout": "unembedding(final_norm(hidden @ J_layer.T))",
        "statistic": "full_vocabulary_logit_excess_kurtosis",
        "aggregation": "prompt_median_with_paired_bootstrap_95pct_ci",
        "residual_diagnostics": ["lesion_to_clean_norm_ratio", "clean_lesion_cosine"],
    }:
        raise ValueError("activation measurement definition drifted")
    geometry = design.get("geometry", {})
    if geometry != {
        "token_count": 4096,
        "sketch_dimension": 256,
        "seed": 20260826,
        "metrics": ["linear_cka", "mean_paired_cosine"],
        "warning": "random-projection estimates; descriptive, not causal",
    }:
        raise ValueError("fresh-lens geometry definition drifted")
    if design.get("bootstrap") != {"resamples": 2000, "seed": 20260826}:
        raise ValueError("bootstrap definition drifted")
    if design.get("training") is not False:
        raise ValueError("figure suite must remain evaluation-only")

    source = design["source_rank_design"]
    rank_path = Path(source["path"])
    if not rank_path.exists() and str(rank_path).startswith("/opt/experiment/"):
        # Unit tests run from the checkout, while the immutable job uses the
        # exact in-image /opt/experiment path recorded in the design.
        rank_path = path.parents[2] / rank_path.relative_to("/opt/experiment")
    rank_design = load_rank_design(rank_path, source["sha256"])
    if rank_design["intervention"]["layers"] != LESION_LAYERS:
        raise ValueError("source rank design layer set drifted")
    return design, rank_design


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
def _forward_with_capture(
    model: Any,
    resolved: Any,
    encoded: dict[str, torch.Tensor],
    layers: Sequence[int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    with _capture_layers(resolved, layers) as captured:
        logits = model(**encoded, use_cache=False).logits[0, -1].float()
    missing = sorted(set(layers) - set(captured))
    if missing:
        raise RuntimeError(f"forward did not capture layers {missing}")
    return logits, captured


@torch.inference_mode()
def _activation_metrics(
    resolved: Any,
    jacobians: dict[int, torch.Tensor],
    clean: dict[int, torch.Tensor],
    lesioned: dict[int, torch.Tensor],
    layers: Sequence[int],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for layer in layers:
        clean_hidden = clean[layer][0, -1].float()
        lesion_hidden = lesioned[layer][0, -1].float()
        jacobian = jacobians[layer]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            clean_transport = clean_hidden.to(torch.bfloat16) @ jacobian.T
            lesion_transport = lesion_hidden.to(torch.bfloat16) @ jacobian.T
            clean_readout = resolved.lm_head(resolved.final_norm(clean_transport))
            lesion_readout = resolved.lm_head(resolved.final_norm(lesion_transport))
        clean_norm = clean_hidden.norm().clamp_min(1e-12)
        clean_kurtosis = excess_kurtosis(clean_readout)
        lesion_kurtosis = excess_kurtosis(lesion_readout)
        result[str(layer)] = {
            "clean_kurtosis": clean_kurtosis,
            "lesion_kurtosis": lesion_kurtosis,
            "lesion_minus_clean_kurtosis": lesion_kurtosis - clean_kurtosis,
            "clean_residual_norm": float(clean_norm),
            "lesion_residual_norm": float(lesion_hidden.norm()),
            "lesion_to_clean_norm_ratio": float(lesion_hidden.norm() / clean_norm),
            "clean_lesion_cosine": float(
                F.cosine_similarity(clean_hidden, lesion_hidden, dim=0)
            ),
        }
        del jacobian, clean_transport, lesion_transport, clean_readout, lesion_readout
    return result


def _frequent_token_ids(tokenizer: Any, count: int) -> torch.Tensor:
    blocked = set(tokenizer.all_special_ids)
    selected = [
        token_id for token_id in range(len(tokenizer)) if token_id not in blocked
    ]
    if len(selected) < count:
        raise ValueError(
            f"only {len(selected)} non-special tokens for {count} requested"
        )
    return torch.tensor(selected[:count], dtype=torch.long)


@torch.inference_mode()
def _dictionary_sketch(
    resolved: Any,
    tokenizer: Any,
    lens: LensMatrices,
    *,
    layers: Sequence[int],
    token_count: int,
    sketch_dimension: int,
    seed: int,
) -> dict[str, Any]:
    token_ids = _frequent_token_ids(tokenizer, token_count)
    device_ids = token_ids.to(resolved.lm_head.weight.device)
    rows = resolved.lm_head.weight.detach().index_select(0, device_ids).float()
    # Historical dictionary geometry (legacy_weight), not corrected readout gain.
    norm_weight = getattr(resolved.final_norm, "weight", None)
    if norm_weight is not None:
        rows = rows * norm_weight.detach().float()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = (
        torch.randn(
            lens.d_model,
            sketch_dimension,
            generator=generator,
            dtype=torch.float32,
        )
        .div_(math.sqrt(sketch_dimension))
        .to(rows.device)
    )
    sketches = []
    for layer in layers:
        jacobian = lens.jacobians[layer].to(rows.device)
        directions = F.normalize(rows @ jacobian, dim=-1, eps=1e-8)
        sketches.append((directions @ projection).to(torch.float16).cpu())
        del jacobian, directions
    stacked = torch.stack(sketches)
    return {
        "layers": np.asarray(layers, dtype=np.int64),
        "token_ids": token_ids.numpy(),
        "sketches": stacked.numpy(),
        "token_ids_sha256": _sha256_bytes(token_ids.numpy().tobytes()),
    }


def _rank_source_predictions(
    design: dict[str, Any], model_key: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = design["source_rank_result"]
    summary_path = Path(source["path"])
    _require_sha(summary_path, source["sha256"], "source rank-cutoff summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    condition_name = {
        "base": "base-k10",
        "primary": "primary-k10",
        "high_dose": "high-k10",
    }[model_key]
    matches = [
        (Path(path), digest)
        for path, digest in summary["result_sha256"].items()
        if Path(path).parent.name.endswith(f"-{condition_name}")
    ]
    if len(matches) != 1:
        raise ValueError(f"could not resolve source condition {condition_name}")
    result_path, result_sha = matches[0]
    _require_sha(result_path, result_sha, f"source {condition_name} result")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    predictions_path = result_path.parent / "predictions.jsonl"
    _require_sha(
        predictions_path,
        result["predictions_sha256"],
        f"source {condition_name} predictions",
    )
    rows = [
        row for row in read_jsonl(predictions_path) if row["dataset"] == "synthetic_geo"
    ]
    return rows, {
        "condition": condition_name,
        "result_path": str(result_path),
        "result_sha256": result_sha,
        "predictions_path": str(predictions_path),
        "predictions_sha256": result["predictions_sha256"],
    }


def _verify_prediction_parity(
    observed: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> dict[str, Any]:
    expected_by_id = {row["source_id"]: row for row in expected}
    mismatches = []
    compared = 0
    for row in observed:
        prior = expected_by_id[row["source_id"]]
        for field in ("clean_predicted_token_id", "jspace_predicted_token_id"):
            compared += 1
            if int(row[field]) != int(prior[field]):
                mismatches.append(
                    {
                        "source_id": row["source_id"],
                        "field": field,
                        "expected": int(prior[field]),
                        "observed": int(row[field]),
                    }
                )
    return {
        "compared_values": compared,
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def run_condition(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite condition: {args.output_dir}")
    design, rank_design = load_design(args.design, args.design_sha256)
    if not 0 <= args.condition_index < len(design["conditions"]):
        raise ValueError("condition index is outside frozen figure design")
    condition = design["conditions"][args.condition_index]
    if condition["index"] != args.condition_index:
        raise ValueError("condition ordering drifted")
    if os.environ.get("EXPERIMENT_IMAGE") != args.experiment_image:
        raise ValueError("runtime image does not match immutable CLI image")

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    model_receipt = _model_receipt(rank_design, condition["model"])
    model, tokenizer, resolved = load_model_and_tokenizer(
        ModelConfig(
            name_or_path=model_receipt["path"],
            revision=model_receipt["revision"],
            dtype="float32",
            attn_implementation="sdpa",
            gradient_checkpointing=False,
            freeze_output_head=True,
        ),
        device,
    )
    synthetic, _ = _load_rows(rank_design, tokenizer)
    if len(synthetic) != design["measurement"]["rows"]:
        raise ValueError("frozen synthetic cohort size drifted")
    lens_receipt = _lens_receipt(rank_design, condition["lens"])
    lens = LensMatrices.load(lens_receipt["path"])
    jspace = JSpaceAblator(
        resolved,
        lens,
        _intervention_config("jspace", Path(lens_receipt["path"]), 10),
        device=device,
        dtype=torch.bfloat16,
    )

    records = []
    model.eval()
    for row in synthetic:
        # Use the exact encoding helper from the frozen recovery/rank evaluators;
        # parity below makes any future prompt-path drift blocking.
        encoded = _encode(tokenizer, row["prompt"], device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            plan = jspace.build_plan(
                model, encoded["input_ids"], encoded["attention_mask"]
            )
            clean_logits, clean_activations = _forward_with_capture(
                model, resolved, encoded, LESION_LAYERS
            )
            with jspace.apply(plan):
                lesion_logits, lesion_activations = _forward_with_capture(
                    model, resolved, encoded, LESION_LAYERS
                )
        if int(plan.clean_next_logits[0].argmax()) != int(clean_logits.argmax()):
            raise RuntimeError("clean argmax changed between plan and capture forward")
        expected_token_id = int(row["expected_token_id"])
        records.append(
            {
                "dataset": "synthetic_geo",
                "source_id": row["source_id"],
                "expected_token_id": expected_token_id,
                "clean_predicted_token_id": int(clean_logits.argmax()),
                "jspace_predicted_token_id": int(lesion_logits.argmax()),
                "clean_exact": int(clean_logits.argmax() == expected_token_id),
                "jspace_exact": int(lesion_logits.argmax() == expected_token_id),
                "layers": _activation_metrics(
                    resolved,
                    jspace.jacobians,
                    clean_activations,
                    lesion_activations,
                    LESION_LAYERS,
                ),
            }
        )

    prior_rows, prior_receipt = _rank_source_predictions(design, condition["model"])
    parity = _verify_prediction_parity(records, prior_rows)
    if not parity["passed"]:
        raise RuntimeError(
            f"activation capture changed frozen k=10 predictions: {parity}"
        )

    geometry = design["geometry"]
    sketch = _dictionary_sketch(
        resolved,
        tokenizer,
        lens,
        layers=LESION_LAYERS,
        token_count=geometry["token_count"],
        sketch_dimension=geometry["sketch_dimension"],
        seed=geometry["seed"],
    )
    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "activation_metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    sketch_path = args.output_dir / "dictionary_sketch.npz"
    np.savez_compressed(
        sketch_path,
        layers=sketch["layers"],
        token_ids=sketch["token_ids"],
        sketches=sketch["sketches"],
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "protocol": "q35_recovery_figure_suite",
        "training": False,
        "condition": condition,
        "experiment_image": args.experiment_image,
        "design_sha256": args.design_sha256,
        "model_receipt": model_receipt,
        "lens_receipt": lens_receipt,
        "rows": len(records),
        "clean_accuracy": sum(row["clean_exact"] for row in records) / len(records),
        "jspace_accuracy": sum(row["jspace_exact"] for row in records) / len(records),
        "source_parity": parity,
        "source_receipt": prior_receipt,
        "activation_metrics_sha256": sha256_path(metrics_path),
        "dictionary_sketch_sha256": sha256_path(sketch_path),
        "dictionary_token_ids_sha256": sketch["token_ids_sha256"],
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def bootstrap_interval(
    values: Sequence[float], *, resamples: int, seed: int, statistic: str = "median"
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a finite 1-D sample of size >=2")
    if statistic not in {"median", "mean"}:
        raise ValueError("bootstrap statistic must be median or mean")
    reducer = np.median if statistic == "median" else np.mean
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(resamples, len(array)))
    estimates = reducer(array[draws], axis=1)
    return {
        "estimate": float(reducer(array)),
        "low": float(np.quantile(estimates, 0.025)),
        "high": float(np.quantile(estimates, 0.975)),
    }


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if not 0 <= successes <= total or total < 1:
        raise ValueError("invalid binomial counts")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return center - radius, center + radius


def rank_category(audit: dict[str, Any]) -> str:
    if audit["intermediate_output_blocked"]:
        return "output_protected"
    if audit["intermediate_seen_top10_any_layer"]:
        return "top_10"
    if audit["intermediate_rank11_12_without_top10"]:
        return "rank_11_12"
    best = audit["intermediate_best_rank_any_layer"]
    if best is not None and best <= 50:
        return "rank_13_50"
    return "not_seen_top_50"


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_rank_summary(design: dict[str, Any]) -> dict[str, Any]:
    receipt = design["source_rank_result"]
    path = Path(receipt["path"])
    _require_sha(path, receipt["sha256"], "source rank-cutoff summary")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("training") is not False:
        raise ValueError("source rank-cutoff result is not a completed evaluation")
    return summary


def _geometry_similarity(
    sketches: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    base = sketches["base"]
    rows = []
    for model in ("primary", "high_dose"):
        other = sketches[model]
        if not np.array_equal(base["layers"], other["layers"]):
            raise ValueError("dictionary sketch layers differ")
        if not np.array_equal(base["token_ids"], other["token_ids"]):
            raise ValueError("dictionary sketch token IDs differ")
        for index, layer in enumerate(base["layers"].tolist()):
            left = torch.from_numpy(base["sketches"][index]).float()
            right = torch.from_numpy(other["sketches"][index]).float()
            paired_cosine = F.cosine_similarity(left, right, dim=-1)
            rows.append(
                {
                    "model": model,
                    "layer": int(layer),
                    "linear_cka": linear_cka(left, right),
                    "mean_paired_cosine": float(paired_cosine.mean()),
                }
            )
    return rows


def _plot_suite(
    output_dir: Path,
    *,
    activation_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    dpi: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )

    def activation_panel(axis: Any, metric: str, title: str, ylabel: str) -> None:
        for model in EXPECTED_MODELS:
            for state, linestyle in (("clean", "-"), ("lesion", "--")):
                selected = [
                    row
                    for row in activation_rows
                    if row["model"] == model
                    and row["state"] == state
                    and row["metric"] == metric
                ]
                axis.plot(
                    [row["layer"] for row in selected],
                    [row["estimate"] for row in selected],
                    color=MODEL_COLORS[model],
                    linestyle=linestyle,
                    marker="o",
                    linewidth=1.8,
                    label=f"{MODEL_LABELS[model]} · {state}",
                )
                axis.fill_between(
                    [row["layer"] for row in selected],
                    [row["low"] for row in selected],
                    [row["high"] for row in selected],
                    color=MODEL_COLORS[model],
                    alpha=0.08,
                )
        axis.set(title=title, xlabel="Decoder layer", ylabel=ylabel)
        axis.grid(alpha=0.2)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    activation_panel(
        axis,
        "kurtosis",
        "J-lens readout concentration without and during the lesion",
        "Median full-vocabulary excess kurtosis",
    )
    axis.legend(ncol=2, fontsize=8, frameon=False)
    figure.tight_layout()
    kurtosis_path = output_dir / "activation_kurtosis.png"
    figure.savefig(kurtosis_path, dpi=dpi)
    figure.savefig(output_dir / "activation_kurtosis.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for model in EXPECTED_MODELS:
        selected = [
            row
            for row in activation_rows
            if row["model"] == model and row["metric"] == "norm_ratio"
        ]
        axes[0].plot(
            [row["layer"] for row in selected],
            [row["estimate"] for row in selected],
            marker="o",
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        axes[0].fill_between(
            [row["layer"] for row in selected],
            [row["low"] for row in selected],
            [row["high"] for row in selected],
            color=MODEL_COLORS[model],
            alpha=0.1,
        )
        selected = [
            row
            for row in activation_rows
            if row["model"] == model and row["metric"] == "cosine"
        ]
        axes[1].plot(
            [row["layer"] for row in selected],
            [row["estimate"] for row in selected],
            marker="o",
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    axes[0].set(
        title="Residual norm retained",
        xlabel="Decoder layer",
        ylabel="Lesion / clean norm",
    )
    axes[1].set(
        title="Clean–lesioned residual alignment",
        xlabel="Decoder layer",
        ylabel="Cosine similarity",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    perturbation_path = output_dir / "activation_perturbation.png"
    figure.savefig(perturbation_path, dpi=dpi)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for model in EXPECTED_MODELS:
        for control, linestyle, alpha in (("jspace", "-", 1.0), ("random", ":", 0.65)):
            selected = [
                row
                for row in rank_rows
                if row["model"] == model and row["control"] == control
            ]
            axis.plot(
                [row["k"] for row in selected],
                [row["accuracy"] for row in selected],
                color=MODEL_COLORS[model],
                linestyle=linestyle,
                marker="o",
                alpha=alpha,
                linewidth=2,
                label=f"{MODEL_LABELS[model]} · {control}",
            )
            axis.fill_between(
                [row["k"] for row in selected],
                [row["low"] for row in selected],
                [row["high"] for row in selected],
                color=MODEL_COLORS[model],
                alpha=0.07,
            )
    axis.set(
        title="Recovered capability survives wider model-specific lesions",
        xlabel="Active J-directions removed per layer (k)",
        ylabel="Held-out synthetic multihop accuracy",
        ylim=(0, 1.02),
        xticks=list(EXPECTED_K_VALUES),
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=8, frameon=False)
    figure.tight_layout()
    rank_path = output_dir / "accuracy_vs_lesion_width.png"
    figure.savefig(rank_path, dpi=dpi)
    figure.savefig(output_dir / "accuracy_vs_lesion_width.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 5))
    for model in ("primary", "high_dose"):
        selected = [row for row in geometry_rows if row["model"] == model]
        axis.plot(
            [row["layer"] for row in selected],
            [row["linear_cka"] for row in selected],
            marker="o",
            color=MODEL_COLORS[model],
            label=f"{MODEL_LABELS[model]} vs base · CKA",
        )
        axis.plot(
            [row["layer"] for row in selected],
            [row["mean_paired_cosine"] for row in selected],
            marker="s",
            linestyle="--",
            color=MODEL_COLORS[model],
            alpha=0.8,
            label=f"{MODEL_LABELS[model]} vs base · paired cosine",
        )
    axis.set(
        title="Fresh-lens geometry after recovery",
        xlabel="Decoder layer",
        ylabel="Similarity to base fresh lens",
        ylim=(-0.05, 1.05),
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    geometry_path = output_dir / "fresh_lens_similarity.png"
    figure.savefig(geometry_path, dpi=dpi)
    plt.close(figure)

    categories = [
        "output_protected",
        "top_10",
        "rank_11_12",
        "rank_13_50",
        "not_seen_top_50",
    ]
    category_labels = [
        "Output-protected",
        "Top 10",
        "Ranks 11–12",
        "Ranks 13–50",
        "Not in top 50",
    ]
    category_colors = ["#B279A2", "#E45756", "#F2CF5B", "#72B7B2", "#BAB0AC"]
    figure, axis = plt.subplots(figsize=(8.5, 5))
    bottoms = np.zeros(len(EXPECTED_MODELS))
    for category, label, color in zip(
        categories, category_labels, category_colors, strict=True
    ):
        values = [
            next(
                row["fraction"]
                for row in category_rows
                if row["model"] == model and row["category"] == category
            )
            for model in EXPECTED_MODELS
        ]
        axis.bar(
            range(len(EXPECTED_MODELS)),
            values,
            bottom=bottoms,
            label=label,
            color=color,
        )
        bottoms += np.asarray(values)
    axis.set(
        title="Where the known intermediate appears under the k=10 trajectory",
        ylabel="Fraction of held-out prompts",
        xticks=range(len(EXPECTED_MODELS)),
        xticklabels=[MODEL_LABELS[model] for model in EXPECTED_MODELS],
        ylim=(0, 1),
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout()
    category_path = output_dir / "intermediate_rank_distribution.png"
    figure.savefig(category_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    activation_panel(
        axes[0, 0],
        "kurtosis",
        "A  Activation-readout kurtosis",
        "Median excess kurtosis",
    )
    for model in EXPECTED_MODELS:
        selected = [
            row
            for row in rank_rows
            if row["model"] == model and row["control"] == "jspace"
        ]
        axes[0, 1].plot(
            [row["k"] for row in selected],
            [row["accuracy"] for row in selected],
            marker="o",
            linewidth=2,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        selected = [
            row
            for row in activation_rows
            if row["model"] == model and row["metric"] == "norm_ratio"
        ]
        axes[1, 0].plot(
            [row["layer"] for row in selected],
            [row["estimate"] for row in selected],
            marker="o",
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    axes[0, 1].set(
        title="B  Accuracy under wider J-space lesions",
        xlabel="Directions removed (k)",
        ylabel="Held-out accuracy",
        ylim=(0, 1.02),
        xticks=list(EXPECTED_K_VALUES),
    )
    axes[1, 0].set(
        title="C  Lesion severity",
        xlabel="Decoder layer",
        ylabel="Lesion / clean residual norm",
    )
    for model in ("primary", "high_dose"):
        selected = [row for row in geometry_rows if row["model"] == model]
        axes[1, 1].plot(
            [row["layer"] for row in selected],
            [row["linear_cka"] for row in selected],
            marker="o",
            color=MODEL_COLORS[model],
            label=f"{MODEL_LABELS[model]} vs base",
        )
    axes[1, 1].set(
        title="D  Model-specific fresh-lens geometry",
        xlabel="Decoder layer",
        ylabel="Linear CKA",
        ylim=(-0.05, 1.05),
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, frameon=False)
    figure.suptitle(
        "Functional recovery under a persistent, freshly refit J-space lesion",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    headline_path = output_dir / "headline_recovery_figure.png"
    figure.savefig(headline_path, dpi=dpi)
    figure.savefig(output_dir / "headline_recovery_figure.pdf")
    plt.close(figure)
    return [
        kurtosis_path,
        output_dir / "activation_kurtosis.pdf",
        perturbation_path,
        rank_path,
        output_dir / "accuracy_vs_lesion_width.pdf",
        geometry_path,
        category_path,
        headline_path,
        output_dir / "headline_recovery_figure.pdf",
    ]


def reduce_results(args: argparse.Namespace) -> dict[str, Any]:
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("refusing to overwrite figure-suite summary")
    design, _ = load_design(args.design, args.design_sha256)
    bootstrap = design["bootstrap"]
    condition_results = {}
    metric_records: dict[str, list[dict[str, Any]]] = {}
    sketches = {}
    result_hashes = {}
    for condition in design["conditions"]:
        condition_dir = (
            args.output_dir
            / "conditions"
            / f"condition-{condition['index']:02d}-{condition['name']}"
        )
        result_path = condition_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed" or result.get("condition") != condition:
            raise ValueError(f"condition result drift: {condition['name']}")
        if result.get("experiment_image") != args.experiment_image:
            raise ValueError(f"condition image drift: {condition['name']}")
        metrics_path = condition_dir / "activation_metrics.jsonl"
        sketch_path = condition_dir / "dictionary_sketch.npz"
        _require_sha(
            metrics_path,
            result["activation_metrics_sha256"],
            f"{condition['name']} metrics",
        )
        _require_sha(
            sketch_path,
            result["dictionary_sketch_sha256"],
            f"{condition['name']} sketch",
        )
        if not result["source_parity"]["passed"]:
            raise ValueError(f"source parity failed: {condition['name']}")
        condition_results[condition["model"]] = result
        metric_records[condition["model"]] = read_jsonl(metrics_path)
        with np.load(sketch_path) as payload:
            sketches[condition["model"]] = {
                key: payload[key].copy() for key in payload.files
            }
        result_hashes[str(result_path)] = sha256_path(result_path)

    activation_rows = []
    for model, records in metric_records.items():
        for layer in LESION_LAYERS:
            for state, field in (
                ("clean", "clean_kurtosis"),
                ("lesion", "lesion_kurtosis"),
            ):
                interval = bootstrap_interval(
                    [row["layers"][str(layer)][field] for row in records],
                    resamples=bootstrap["resamples"],
                    # Identical resample indices preserve the clean/lesion
                    # prompt pairing when the two intervals are compared.
                    seed=bootstrap["seed"] + layer,
                )
                activation_rows.append(
                    {
                        "model": model,
                        "layer": layer,
                        "state": state,
                        "metric": "kurtosis",
                        **interval,
                    }
                )
            for metric, field in (
                ("norm_ratio", "lesion_to_clean_norm_ratio"),
                ("cosine", "clean_lesion_cosine"),
                ("kurtosis_delta", "lesion_minus_clean_kurtosis"),
            ):
                interval = bootstrap_interval(
                    [row["layers"][str(layer)][field] for row in records],
                    resamples=bootstrap["resamples"],
                    seed=bootstrap["seed"]
                    + layer
                    + {"norm_ratio": 2000, "cosine": 3000, "kurtosis_delta": 4000}[
                        metric
                    ],
                )
                activation_rows.append(
                    {
                        "model": model,
                        "layer": layer,
                        "state": "paired",
                        "metric": metric,
                        **interval,
                    }
                )

    rank_summary = _load_rank_summary(design)
    rank_rows = []
    rank_labels = {"base": "base", "primary": "primary", "high_dose": "high"}
    for model in EXPECTED_MODELS:
        for k in EXPECTED_K_VALUES:
            metrics = rank_summary["condition_metrics"][f"{rank_labels[model]}-k{k}"][
                "synthetic_geo"
            ]
            for control, field in (
                ("jspace", "jspace_accuracy"),
                ("random", "random_accuracy"),
            ):
                accuracy = float(metrics[field])
                successes = round(accuracy * 129)
                low, high = wilson_interval(successes, 129)
                rank_rows.append(
                    {
                        "model": model,
                        "k": k,
                        "control": control,
                        "successes": successes,
                        "rows": 129,
                        "accuracy": accuracy,
                        "low": low,
                        "high": high,
                    }
                )

    category_rows = []
    for model in EXPECTED_MODELS:
        predictions, _ = _rank_source_predictions(design, model)
        counts = {
            category: 0
            for category in (
                "output_protected",
                "top_10",
                "rank_11_12",
                "rank_13_50",
                "not_seen_top_50",
            )
        }
        for row in predictions:
            counts[rank_category(row["rank_audit"])] += 1
        if sum(counts.values()) != 129:
            raise ValueError("rank-category cohort size drifted")
        category_rows.extend(
            {
                "model": model,
                "category": category,
                "count": count,
                "rows": 129,
                "fraction": count / 129,
            }
            for category, count in counts.items()
        )

    geometry_rows = _geometry_similarity(sketches)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "activation_metrics_summary.csv", activation_rows)
    _write_csv(args.output_dir / "accuracy_vs_k.csv", rank_rows)
    _write_csv(args.output_dir / "fresh_lens_similarity.csv", geometry_rows)
    _write_csv(args.output_dir / "intermediate_rank_distribution.csv", category_rows)
    figures = _plot_suite(
        args.output_dir,
        activation_rows=activation_rows,
        rank_rows=rank_rows,
        geometry_rows=geometry_rows,
        category_rows=category_rows,
        dpi=int(design["figures"]["dpi"]),
    )
    artifact_hashes = {
        str(path): sha256_path(path)
        for path in [
            args.output_dir / "activation_metrics_summary.csv",
            args.output_dir / "accuracy_vs_k.csv",
            args.output_dir / "fresh_lens_similarity.csv",
            args.output_dir / "intermediate_rank_distribution.csv",
            *figures,
        ]
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "protocol": "q35_recovery_figure_suite",
        "training": False,
        "experiment_image": args.experiment_image,
        "design_sha256": args.design_sha256,
        "source_rank_result": design["source_rank_result"],
        "condition_result_sha256": result_hashes,
        "condition_accuracy": {
            model: {
                "clean": result["clean_accuracy"],
                "jspace_k10": result["jspace_accuracy"],
            }
            for model, result in condition_results.items()
        },
        "activation_metrics": activation_rows,
        "rank_cutoff_metrics": rank_rows,
        "fresh_lens_similarity": geometry_rows,
        "intermediate_rank_distribution": category_rows,
        "artifacts_sha256": artifact_hashes,
        "interpretation_guardrail": design["interpretation_guardrail"],
        "further_training_authorized": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--design-sha256", required=True)
    parser.add_argument("--experiment-image", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition-index", type=int, default=0)
    parser.add_argument("--reduce-only", action="store_true")
    args = parser.parse_args()
    if args.reduce_only:
        reduce_results(args)
    else:
        run_condition(args)


if __name__ == "__main__":
    main()
