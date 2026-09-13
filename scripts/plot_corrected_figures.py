#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==2.5.2",
# ]
# ///
"""CPU reduction of corrected measurements. No legacy numerical inputs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

MODELS = ("base", "primary", "replicate")
COLORS = ("#4C78A8", "#F58518", "#54A24B")
LABELS = ("Base", "Corrected primary", "Corrected replicate")


def cluster_interval(values, countries, resamples=2000, seed=20260908):
    """Prompt-weighted median; sample whole countries with replacement."""
    values = np.asarray(values, dtype=float)
    countries = np.asarray(countries)
    if not np.isfinite(values).all() or len(values) != len(countries):
        raise ValueError("Invalid bootstrap inputs")
    groups = [np.flatnonzero(countries == c) for c in np.unique(countries)]
    if len(groups) < 2:
        raise ValueError("At least two countries required")
    rng = np.random.default_rng(seed)
    draws = [
        np.median(
            values[
                np.concatenate(
                    [groups[j] for j in rng.integers(len(groups), size=len(groups))]
                )
            ]
        )
        for _ in range(resamples)
    ]
    return float(np.median(values)), *np.percentile(draws, [2.5, 97.5]).tolist()


def cka_matrix(states):
    """states: prompt x layer x feature; exact centered Gram linear CKA."""
    x = np.asarray(states, dtype=np.float64).transpose(1, 0, 2)
    x = x - x.mean(axis=1, keepdims=True)
    grams = x @ x.transpose(0, 2, 1)
    flat = grams.reshape(len(grams), -1)
    norm = np.linalg.norm(flat, axis=1)
    if np.any(norm == 0):
        raise ValueError("CKA undefined for constant activations")
    return (flat @ flat.T) / np.outer(norm, norm)


def csv_write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reduce(root, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads((root / "design.json").read_text())
    pre = json.loads((root / "preflight.json").read_text())
    for model in MODELS:
        gate = json.loads((root / f"measure-{model}.json").read_text())
        if not gate["passed"] or gate["design_sha256"] != pre["design_sha256"]:
            raise ValueError("Incomplete/mismatched measurement run")
    summaries, raw, matrices = [], [], {}
    reference_ids = pre["prompt_ids"]
    countries_ref = None
    for model in MODELS:
        for key in ("published", model + "_fresh"):
            name = model + "-" + key
            rows = json.loads((root / f"metrics-{name}.json").read_text())
            if [r["source_id"] for r in rows] != reference_ids:
                raise ValueError("Metric prompt ordering drift")
            countries = [str(r["cluster_id"]) for r in rows]
            if countries_ref is None:
                countries_ref = countries
            if countries != countries_ref:
                raise ValueError("Country grouping drift")
            mode = "published" if key == "published" else "own_fresh"
            for layer in spec["measurement"]["layers"]:
                for metric in (
                    "clean_kurtosis",
                    "lesion_kurtosis",
                    "lesion_minus_clean_kurtosis",
                    "lesion_to_clean_norm_ratio",
                    "clean_lesion_cosine",
                ):
                    vals = [r["layers"][str(layer)][metric] for r in rows]
                    estimate, low, high = cluster_interval(
                        vals,
                        countries,
                        spec["measurement"]["bootstrap_resamples"],
                        spec["measurement"]["bootstrap_seed"],
                    )
                    summaries.append(
                        dict(
                            model=model,
                            lens=mode,
                            layer=layer,
                            metric=metric,
                            estimate=estimate,
                            low=low,
                            high=high,
                        )
                    )
                for row in rows:
                    raw.append(
                        dict(
                            model=model,
                            lens=mode,
                            layer=layer,
                            source_id=row["source_id"],
                            country_id=row["cluster_id"],
                            **row["layers"][str(layer)],
                        )
                    )
            with np.load(root / f"activations-{name}.npz", allow_pickle=False) as a:
                if (
                    a["prompt_ids"].tolist() != reference_ids
                    or a["country_ids"].tolist() != countries
                ):
                    raise ValueError("Activation ordering drift")
                if key == "published":
                    if a["layers"].tolist() != list(range(a["clean"].shape[1])):
                        raise ValueError("All ordered decoder layers required")
                    for state in ("clean", "lesioned"):
                        matrices[model + "_" + state] = cka_matrix(a[state])
    csv_write(output / "corrected_measurements.csv", raw)
    csv_write(output / "corrected_country_bootstrap.csv", summaries)
    np.savez_compressed(output / "corrected_prompt_space_cka.npz", **matrices)
    captions = {}

    def save(fig, name, caption):
        fig.tight_layout()
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"{name}.{extension}", dpi=220, bbox_inches="tight")
        plt.close(fig)
        captions[name] = caption

    def curve(ax, model, mode, metric, color, label, style="-"):
        rows = [
            r
            for r in summaries
            if r["model"] == model and r["lens"] == mode and r["metric"] == metric
        ]
        x = [r["layer"] for r in rows]
        ax.plot(
            x,
            [r["estimate"] for r in rows],
            style,
            marker="o",
            color=color,
            label=label,
        )
        ax.fill_between(
            x,
            [r["low"] for r in rows],
            [r["high"] for r in rows],
            color=color,
            alpha=0.10,
        )
        ax.set_xticks(spec["measurement"]["layers"])
        ax.grid(alpha=0.2)
        ax.set_xlabel("Decoder block (zero-based)")

    for mode in ("published", "own_fresh"):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for model, color, label in zip(MODELS, COLORS, LABELS, strict=True):
            for condition, style in (("clean", "-"), ("lesion", "--")):
                curve(
                    ax,
                    model,
                    mode,
                    condition + "_kurtosis",
                    color,
                    label + " — " + condition,
                    style,
                )
        ax.set_ylabel("Median full-vocabulary excess kurtosis")
        ax.set_title(
            "J-lens readout concentration: "
            + (
                "published base lens"
                if mode == "published"
                else "each model’s own 500-prompt lens"
            )
        )
        ax.legend(ncol=2, frameon=False, fontsize=8)
        save(
            fig,
            "corrected_activation_kurtosis_" + mode,
            f"Final-prompt J-lens logit excess kurtosis across 129 fixed prompts; {mode} lens. "  # noqa: E501
            "Solid clean, dashed corrected online-current sequential k=10 lesion. Medians and pointwise 95% "  # noqa: E501
            "country-cluster bootstrap intervals (2,000 resamples; prompt-weighted median). "  # noqa: E501
            "This describes readout concentration, not a compensating circuit. Model-specific norm/head are used with the stated lens matrices.",  # noqa: E501
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, metric, label in zip(
            axes,
            ("lesion_to_clean_norm_ratio", "clean_lesion_cosine"),
            ("Lesioned / clean residual norm", "Clean–lesioned cosine similarity"),
            strict=True,
        ):
            for model, color, title in zip(MODELS, COLORS, LABELS, strict=True):
                curve(ax, model, mode, metric, color, title)
            ax.set_ylabel(label)
            ax.axhline(1, color="gray", lw=0.7, ls=":")
        axes[0].legend(frameon=False, fontsize=8)
        save(
            fig,
            "corrected_lesion_severity_" + mode,
            f"Paired final-prompt residual norm ratio and cosine under {mode} lens for the same 129 prompts. "  # noqa: E501
            "Medians with pointwise country-cluster 95% intervals. States are captured after lesion hooks; "  # noqa: E501
            "differences include preceding lesions. Comparable severity can constrain simple intervention shrinkage, "  # noqa: E501
            "but does not establish exact span erasure.",
        )
    base, damaged = matrices["base_clean"], matrices["base_lesioned"]
    deltas = [damaged - base] + [
        matrices[m + "_lesioned"] - damaged for m in MODELS[1:]
    ]
    limit = max(float(np.abs(x).max()) for x in deltas) or 1e-12
    for model in MODELS[1:]:
        recovered = matrices[model + "_lesioned"]
        fig, axes = plt.subplots(1, 5, figsize=(19, 4.1))
        fig.suptitle("Prompt-space activation CKA · published base lens", fontsize=13)
        panels = (base, damaged, recovered, damaged - base, recovered - damaged)
        titles = (
            "Base clean",
            "Base lesion",
            f"Corrected {model} lesion",
            "Lesion effect",
            "Recovery effect",
        )
        for i, (ax, matrix, title) in enumerate(zip(axes, panels, titles, strict=True)):
            im = ax.imshow(
                matrix,
                origin="lower",
                cmap="viridis" if i < 3 else "RdBu_r",
                vmin=0 if i < 3 else -limit,
                vmax=1 if i < 3 else limit,
            )
            ax.set(title=title, xlabel="Decoder block", ylabel="Decoder block")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        save(
            fig,
            "corrected_prompt_space_cka_" + model,
            f"Prompt-space activation CKA for corrected {model}, using the published lens intervention. "  # noqa: E501
            "Each square compares decoder layers within one condition using centered linear CKA on the ordered "  # noqa: E501
            "129 final-position residuals. Differences subtract the displayed condition matrices; they are not "  # noqa: E501
            "cross-model CKA. Both seed figures share color scales. Global organization does not prove causal rerouting.",  # noqa: E501
        )
    (output / "CAPTIONS.json").write_text(json.dumps(captions, indent=2) + "\n")
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.iterdir()
        if p.is_file()
    }
    (output / "receipt.json").write_text(
        json.dumps(
            {
                "input_hashes": hashes,
                "design": spec,
                "preflight": pre,
                "plot_source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    reduce(a.input, a.output)
