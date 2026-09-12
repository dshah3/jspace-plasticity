"""Measure J-lens geometry and propose a three-block workspace band."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jspace_plasticity.config import ModelConfig
from jspace_plasticity.lens import LensMatrices
from jspace_plasticity.lens.geometry import excess_kurtosis, linear_cka
from jspace_plasticity.modeling import ResolvedModel, load_model_and_tokenizer
from jspace_plasticity.readout import readout_rows


def _dictionary_token_ids(tokenizer: object, count: int, *, seed: int) -> torch.Tensor:
    special = set(getattr(tokenizer, "all_special_ids", []))
    vocabulary_size = len(tokenizer)  # type: ignore[arg-type]
    candidates = np.asarray(
        [token_id for token_id in range(vocabulary_size) if token_id not in special],
        dtype=np.int64,
    )
    if count > len(candidates):
        raise ValueError(
            f"tokenizer exposes {len(candidates)} non-special tokens; requested {count}"
        )
    # Match Elie Bakouch's open-model analysis: a frozen RandomState(0) sample
    # over the vocabulary, rather than a hand-curated semantic subset.
    selected = np.random.RandomState(seed).choice(candidates, size=count, replace=False)
    return torch.from_numpy(selected)


def _three_blocks(cka: np.ndarray, minimum_size: int = 2) -> tuple[int, int]:
    n_layers = cka.shape[0]
    if n_layers < 3 * minimum_size:
        raise ValueError("not enough fitted layers for a three-block segmentation")
    best_score = float("-inf")
    best = (minimum_size, n_layers - minimum_size)
    for first in range(minimum_size, n_layers - 2 * minimum_size + 1):
        for second in range(first + minimum_size, n_layers - minimum_size + 1):
            labels = np.zeros(n_layers, dtype=np.int64)
            labels[first:second] = 1
            labels[second:] = 2
            upper = np.triu_indices(n_layers, k=1)
            same = labels[upper[0]] == labels[upper[1]]
            within = float(cka[upper][same].mean())
            between = float(cka[upper][~same].mean())
            score = within - between
            if score > best_score:
                best_score = score
                best = (first, second)
    return best


@torch.no_grad()
def _geometry(
    model: ResolvedModel,
    tokenizer: object,
    lens: LensMatrices,
    *,
    token_count: int,
    token_seed: int,
    sketch_dimension: int,
    device: torch.device,
) -> tuple[list[int], np.ndarray, list[float], list[float], list[int]]:
    token_ids = _dictionary_token_ids(tokenizer, token_count, seed=token_seed)
    lm_head = model.lm_head
    final_norm = model.final_norm
    rows = lm_head.weight.detach().index_select(0, token_ids.to(lm_head.weight.device))
    rows = readout_rows(rows, final_norm)
    rows = rows.float().to(device)

    layers = list(lens.layers)
    generator = torch.Generator(device="cpu").manual_seed(token_seed + 71_923)
    sketch = (
        torch.randn(
            rows.shape[1], sketch_dimension, generator=generator, dtype=torch.float32
        )
        / sketch_dimension**0.5
    )
    sketches: list[torch.Tensor] = []
    participation_ratios: list[float] = []
    kurtoses: list[float] = []
    for layer in layers:
        vectors = rows @ lens.jacobians[layer].to(device)
        vectors -= vectors.mean(dim=0, keepdim=True)
        kurtoses.append(excess_kurtosis(vectors))
        compact = vectors.cpu() @ sketch
        covariance = compact.T @ compact
        participation_ratios.append(
            float(
                compact.square()
                .sum()
                .square()
                .div(covariance.square().sum().clamp_min(1e-12))
                .item()
            )
        )
        sketches.append(compact)
        del vectors, compact, covariance
    cka = np.eye(len(layers), dtype=np.float64)
    for left in range(len(layers)):
        for right in range(left + 1, len(layers)):
            value = linear_cka(sketches[left], sketches[right])
            cka[left, right] = cka[right, left] = value
    return layers, cka, participation_ratios, kurtoses, token_ids.tolist()


def _plot(
    path: Path,
    layers: list[int],
    cka: np.ndarray,
    participation_ratios: list[float],
    boundaries: tuple[int, int],
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (heatmap, ratio) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": (1.2, 1)}
    )
    image = heatmap.imshow(cka, vmin=0.0, vmax=1.0, cmap="viridis")
    heatmap.set(title="J-lens dictionary CKA", xlabel="layer", ylabel="layer")
    tick_positions = np.arange(len(layers))
    stride = max(1, len(layers) // 8)
    heatmap.set_xticks(tick_positions[::stride], np.asarray(layers)[::stride])
    heatmap.set_yticks(tick_positions[::stride], np.asarray(layers)[::stride])
    for boundary in boundaries:
        heatmap.axvline(boundary - 0.5, color="white", linestyle="--", linewidth=1)
        heatmap.axhline(boundary - 0.5, color="white", linestyle="--", linewidth=1)
    figure.colorbar(image, ax=heatmap, fraction=0.046)

    ratio.plot(layers, participation_ratios, marker="o", linewidth=1.5)
    for boundary in boundaries:
        ratio.axvline(layers[boundary], color="black", linestyle="--", linewidth=1)
    ratio.set(
        title="Dictionary participation ratio",
        xlabel="layer",
        ylabel="effective directions",
    )
    ratio.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _plot_kurtosis(
    path: Path, layers: list[int], kurtoses: list[float], dpi: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(layers, kurtoses, marker="o", linewidth=1.5)
    axis.set(
        title="J-lens token-vector excess kurtosis",
        xlabel="layer",
        ylabel="excess kurtosis",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lens", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--token-seed", type=int, default=0)
    parser.add_argument("--cka-sketch-dimension", type=int, default=256)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, resolved = load_model_and_tokenizer(
        ModelConfig(
            name_or_path=args.checkpoint,
            dtype=args.dtype,
            gradient_checkpointing=False,
        ),
        device,
    )
    del model
    lens = LensMatrices.load(args.lens)
    layers, cka, ratios, kurtoses, token_ids = _geometry(
        resolved,
        tokenizer,
        lens,
        token_count=args.tokens,
        token_seed=args.token_seed,
        sketch_dimension=args.cka_sketch_dimension,
        device=device,
    )
    boundary_positions = _three_blocks(cka)
    workspace_layers = layers[boundary_positions[0] : boundary_positions[1]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": args.checkpoint,
        "lens": str(args.lens),
        "layers": layers,
        "cka": cka.tolist(),
        "participation_ratio": ratios,
        "excess_kurtosis": kurtoses,
        "cka_method": "feature-space linear CKA after seeded Gaussian sketch",
        "cka_sketch_dimension": args.cka_sketch_dimension,
        "token_ids": token_ids,
        "block_start_positions": list(boundary_positions),
        "proposed_workspace_layers": workspace_layers,
        "warning": (
            "The segmentation is a proposal, not authorization to train. Confirm the "
            "band with readouts and a clean-versus-lesion ablation sweep."
        ),
    }
    (args.output_dir / "lens_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _plot(
        args.output_dir / "lens_geometry.png",
        layers,
        cka,
        ratios,
        boundary_positions,
        args.dpi,
    )
    _plot_kurtosis(args.output_dir / "lens_kurtosis.png", layers, kurtoses, args.dpi)


if __name__ == "__main__":
    main()
