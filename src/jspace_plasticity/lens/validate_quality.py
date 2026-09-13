"""Run the blocking Section 2.2 quality checks for a fitted J-lens.

This module evaluates an existing lens.  It never fits or updates Jacobians.
The output deliberately remains ``manual_review_required`` until a researcher
reviews the qualitative mid-layer readouts and the CKA block structure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from jspace_plasticity.analyze_lens import _plot, _three_blocks
from jspace_plasticity.evals.two_hop_probe import load_probe_swap
from jspace_plasticity.lens import LensMatrices
from jspace_plasticity.lens.geometry import excess_kurtosis, linear_cka
from jspace_plasticity.modeling import auto_model_class_for_config, resolve_model
from jspace_plasticity.readout import readout_rows

QUALITY_SCHEMA_VERSION = 2
KURTOSIS_WINDOW_LAYERS = 3
MIN_POST_ONSET_KURTOSIS_RELATIVE_RISE = 0.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(path: Path, expected: str) -> str:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, observed {observed}"
        )
    return observed


def _load_corpus_partition(
    path: Path, *, offset: int, count: int
) -> list[dict[str, Any]]:
    if offset < 0 or count < 1:
        raise ValueError("corpus offset must be nonnegative and count must be positive")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if index < offset:
            continue
        if len(rows) == count:
            break
        payload = json.loads(line)
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"invalid corpus text at row {index}")
        rows.append({"corpus_index": index, "text": text})
    if len(rows) != count:
        raise ValueError(
            f"corpus has only {len(rows)} rows at offset {offset}; expected {count}"
        )
    return rows


def _frequent_token_ids(tokenizer: Any, count: int) -> torch.Tensor:
    """Use Qwen's merge-rank-ordered vocabulary as a frozen frequency proxy."""

    blocked = set(tokenizer.all_special_ids)
    selected = [
        token_id for token_id in range(len(tokenizer)) if token_id not in blocked
    ]
    if len(selected) < count:
        raise ValueError(
            f"tokenizer exposes {len(selected)} non-special tokens; requested {count}"
        )
    return torch.tensor(selected[:count], dtype=torch.long)


def _select_rows(
    rows: Sequence[dict[str, str]], *, count: int, seed: int
) -> list[dict[str, str]]:
    if count < 1 or count > len(rows):
        raise ValueError(f"cannot select {count} rows from {len(rows)}")

    def key(row: dict[str, str]) -> str:
        return hashlib.sha256(f"{seed}:{row['name']}".encode()).hexdigest()

    return sorted(rows, key=key)[:count]


def _target_token_ids(tokenizer: Any, text: str) -> list[int]:
    """Return exact one-token renderings of a semantic target.

    Both word-initial and whitespace-prefixed variants matter for BPE
    tokenizers.  Multi-token renderings are not silently approximated.
    """

    variants = {
        text,
        text.lower(),
        text.title(),
        f" {text}",
        f" {text.lower()}",
        f" {text.title()}",
    }
    target = text.strip().casefold()
    token_ids: set[int] = set()
    for variant in variants:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if len(encoded) != 1:
            continue
        token_id = int(encoded[0])
        if tokenizer.decode([token_id]).strip().casefold() == target:
            token_ids.add(token_id)
    return sorted(token_ids)


def _top_tokens(
    logits: torch.Tensor, tokenizer: Any, *, count: int
) -> list[dict[str, Any]]:
    values, indices = logits.topk(count)
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "logit": float(value),
        }
        for rank, (token_id, value) in enumerate(
            zip(indices.tolist(), values.tolist(), strict=True), 1
        )
    ]


def _target_best_rank(
    logits: torch.Tensor, target_ids: Sequence[int]
) -> tuple[int, int]:
    """Return (best rank, position) over a [position, vocabulary] matrix."""

    if logits.ndim != 2 or not target_ids:
        raise ValueError("rank calculation needs 2-D logits and target token IDs")
    target = logits[:, list(target_ids)].amax(dim=-1)
    ranks = (logits > target.unsqueeze(-1)).sum(dim=-1) + 1
    best_position = int(ranks.argmin().item())
    return int(ranks[best_position].item()), best_position


def _block_statistics(cka: np.ndarray, boundaries: tuple[int, int]) -> dict[str, float]:
    labels = np.zeros(cka.shape[0], dtype=np.int64)
    labels[boundaries[0] : boundaries[1]] = 1
    labels[boundaries[1] :] = 2
    upper = np.triu_indices(cka.shape[0], k=1)
    same = labels[upper[0]] == labels[upper[1]]
    within = float(cka[upper][same].mean())
    between = float(cka[upper][~same].mean())
    return {
        "within_block_mean": within,
        "between_block_mean": between,
        "within_minus_between": within - between,
    }


def _post_onset_kurtosis_rise(
    values: Sequence[float],
    *,
    window_layers: int = KURTOSIS_WINDOW_LAYERS,
    minimum_relative_rise: float = MIN_POST_ONSET_KURTOSIS_RELATIVE_RISE,
) -> dict[str, Any]:
    """Measure the paper's within-workspace kurtosis rise.

    Anthropic describes excess kurtosis as beginning to increase roughly one
    third of the way through model depth.  Comparing the *entire* middle-third
    median against the early-third median is not equivalent: large sensory-layer
    values can make that comparison fail even when the post-onset curve rises.

    We instead compare robust local medians immediately after one-third depth
    and immediately before two-thirds depth.  A ten-percent relative increase
    is the preregistered diagnostic floor; it rejects the historical Qwen3-32B
    pilot's two-percent drift while accepting the published n=1000 control's
    approximately thirty-percent rise.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 9:
        raise ValueError("kurtosis rise requires at least nine fitted layers")
    if not np.isfinite(array).all():
        raise ValueError("kurtosis rise requires finite values")
    if window_layers < 1:
        raise ValueError("kurtosis window must contain at least one layer")
    if minimum_relative_rise <= 0:
        raise ValueError("minimum relative kurtosis rise must be positive")

    first_third = len(array) // 3
    second_third = 2 * len(array) // 3
    available = second_third - first_third
    window = min(window_layers, max(1, available // 2))
    onset_slice = array[first_third : first_third + window]
    pre_motor_slice = array[second_third - window : second_third]
    onset_median = float(np.median(onset_slice))
    pre_motor_median = float(np.median(pre_motor_slice))
    absolute_rise = pre_motor_median - onset_median
    relative_rise = absolute_rise / max(abs(onset_median), 1e-12)
    return {
        "passed": bool(relative_rise >= minimum_relative_rise),
        "first_third_index": first_third,
        "second_third_index": second_third,
        "window_layers": window,
        "post_onset_window_indices": [first_third, first_third + window],
        "pre_motor_window_indices": [second_third - window, second_third],
        "post_onset_median": onset_median,
        "pre_motor_median": pre_motor_median,
        "absolute_rise": absolute_rise,
        "relative_rise": relative_rise,
        "minimum_relative_rise": minimum_relative_rise,
        "definition": (
            "relative increase from the median of the first three sampled "
            "layers after one-third depth to the median of the final three "
            "sampled layers before two-thirds depth"
        ),
    }


@torch.inference_mode()
def _geometry(
    *,
    model: Any,
    tokenizer: Any,
    lens: LensMatrices,
    token_count: int,
    sketch_dimension: int,
    seed: int,
) -> tuple[list[int], np.ndarray, list[float], list[int]]:
    resolved = resolve_model(model)
    token_ids = _frequent_token_ids(tokenizer, token_count)
    rows = resolved.lm_head.weight.detach().index_select(
        0, token_ids.to(resolved.lm_head.weight.device)
    )
    rows = readout_rows(rows, resolved.final_norm)
    rows = rows.float()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_projection = (
        torch.randn(
            lens.d_model,
            sketch_dimension,
            generator=generator,
            dtype=torch.float32,
        )
        .div_(sketch_dimension**0.5)
        .to(rows.device)
    )
    sketches: list[torch.Tensor] = []
    participation: list[float] = []
    layers = list(lens.layers)
    for layer in layers:
        jacobian = lens.jacobians[layer].to(rows.device)
        vectors = F.normalize(rows @ jacobian, dim=-1)
        vectors -= vectors.mean(dim=0, keepdim=True)
        compact = vectors @ random_projection
        covariance = compact.T @ compact
        participation.append(
            float(
                compact.square()
                .sum()
                .square()
                .div(covariance.square().sum().clamp_min(1e-12))
                .item()
            )
        )
        sketches.append(compact)
        del jacobian, vectors, compact, covariance

    cka = np.eye(len(layers), dtype=np.float64)
    for left in range(len(layers)):
        for right in range(left + 1, len(layers)):
            value = linear_cka(sketches[left], sketches[right])
            cka[left, right] = cka[right, left] = value
    return layers, cka, participation, token_ids.tolist()


def _plot_readout_kurtosis(
    path: Path, layers: Sequence[int], values: Sequence[float], dpi: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(layers, values, marker="o", linewidth=1.5)
    axis.set(
        title="J-lens activation-readout logit excess kurtosis",
        xlabel="layer",
        ylabel="median excess kurtosis across held-out activations",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _record_qualitative_readouts(
    *,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    prompts: Sequence[dict[str, Any]],
    layers: Sequence[int],
    output: Path,
    max_length: int,
) -> tuple[dict[str, float], list[float]]:
    from jlens import ActivationRecorder

    final_layer = lens_model.n_layers - 1
    last_fitted = max(layers)
    top1_matches = 0
    top10_overlaps: list[float] = []
    readout_kurtoses: dict[int, list[float]] = {layer: [] for layer in layers}
    with output.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            input_ids = lens_model.encode(prompt["text"], max_length=max_length)
            record_at = sorted(set(layers) | {final_layer})
            with ActivationRecorder(lens_model.layers, at=record_at) as recorder:
                lens_model.forward(input_ids)
                activations = {
                    layer: recorder.activations[layer][0, -1].detach().float()
                    for layer in record_at
                }
            model_logits = lens_model.unembed(activations[final_layer]).float().cpu()
            model_top = _top_tokens(model_logits, tokenizer, count=10)
            layer_rows: dict[str, Any] = {}
            for layer in layers:
                transported = lens.transport(activations[layer], layer)
                logits = lens_model.unembed(transported).float().cpu()
                readout_kurtoses[layer].append(excess_kurtosis(logits))
                layer_rows[str(layer)] = _top_tokens(logits, tokenizer, count=10)
                del transported, logits
            last_ids = {row["token_id"] for row in layer_rows[str(last_fitted)]}
            model_ids = {row["token_id"] for row in model_top}
            top10_overlaps.append(len(last_ids & model_ids) / 10)
            top1_matches += int(
                layer_rows[str(last_fitted)][0]["token_id"] == model_top[0]["token_id"]
            )
            payload = {
                **prompt,
                "input_tokens": int(input_ids.shape[-1]),
                "model_top10": model_top,
                "lens_top10_by_layer": layer_rows,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return (
        {
            "last_layer_top1_agreement": top1_matches / len(prompts),
            "last_layer_mean_top10_overlap": float(np.mean(top10_overlaps)),
        },
        [float(np.median(readout_kurtoses[layer])) for layer in layers],
    )


def _two_hop_sanity(
    *,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    workspace_layers: Sequence[int],
    max_length: int,
    top_k: int,
) -> list[dict[str, Any]]:
    from jlens import ActivationRecorder

    records: list[dict[str, Any]] = []
    for row in rows:
        target_ids = _target_token_ids(tokenizer, row["intermediate"])
        visible = row["intermediate"].casefold() in row["prompt"].casefold()
        record: dict[str, Any] = {
            "name": row["name"],
            "category": row["category"],
            "prompt": row["prompt"],
            "intermediate": row["intermediate"],
            "target_token_ids": json.dumps(target_ids),
            "intermediate_visible_in_prompt": visible,
            "token_compatible": bool(target_ids),
            "hit": False,
            "best_rank": None,
            "best_layer": None,
            "best_position": None,
            "best_position_token": "",
        }
        if not target_ids or visible:
            records.append(record)
            continue

        input_ids = lens_model.encode(row["prompt"], max_length=max_length)
        with ActivationRecorder(lens_model.layers, at=workspace_layers) as recorder:
            lens_model.forward(input_ids)
            activations = {
                layer: recorder.activations[layer][0].detach().float()
                for layer in workspace_layers
            }
        best: tuple[int, int, int] | None = None
        for layer in workspace_layers:
            transported = lens.transport(activations[layer], layer)
            logits = lens_model.unembed(transported).float().cpu()
            rank, position = _target_best_rank(logits, target_ids)
            candidate = (rank, layer, position)
            if best is None or candidate < best:
                best = candidate
            del transported, logits
        assert best is not None
        best_rank, best_layer, best_position = best
        prompt_token_ids = input_ids[0].detach().cpu().tolist()
        record.update(
            hit=best_rank <= top_k,
            best_rank=best_rank,
            best_layer=best_layer,
            best_position=best_position,
            best_position_token=tokenizer.decode([prompt_token_ids[best_position]]),
        )
        records.append(record)
    return records


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fields = sorted({field for row in materialized for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from jlens import JacobianLens, from_hf
    from transformers import AutoConfig, AutoTokenizer

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    lens_sha256 = _require_sha256(args.lens, args.expected_lens_sha256)
    corpus_sha256 = _require_sha256(args.corpus, args.expected_corpus_sha256)
    lens_matrices = LensMatrices.load(args.lens)
    if (
        args.expected_lens_prompts is not None
        and lens_matrices.n_prompts != args.expected_lens_prompts
    ):
        raise ValueError(
            f"lens contains {lens_matrices.n_prompts} prompts; "
            f"expected {args.expected_lens_prompts}"
        )

    qualitative_prompts = _load_corpus_partition(
        args.corpus,
        offset=args.heldout_offset,
        count=args.qualitative_prompts,
    )
    two_hop_rows, two_hop_manifest = load_probe_swap(args.two_hop_data)
    selected_two_hop = _select_rows(
        two_hop_rows, count=args.two_hop_prompts, seed=args.two_hop_seed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, revision=args.revision)
    hf_config = AutoConfig.from_pretrained(args.checkpoint, revision=args.revision)
    auto_model = auto_model_class_for_config(hf_config)
    model = auto_model.from_pretrained(
        args.checkpoint,
        revision=args.revision,
        config=hf_config,
        dtype="bfloat16",
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    lens_model = from_hf(model, tokenizer, compile=False, force_bos=True)
    if lens_matrices.d_model != lens_model.d_model:
        raise ValueError(
            f"lens d_model={lens_matrices.d_model}, model d_model={lens_model.d_model}"
        )
    official_lens = JacobianLens(
        jacobians=lens_matrices.jacobians,
        n_prompts=lens_matrices.n_prompts,
        d_model=lens_matrices.d_model,
    )

    args.output_dir.mkdir(parents=True)
    layers, cka, ratios, token_ids = _geometry(
        model=model,
        tokenizer=tokenizer,
        lens=lens_matrices,
        token_count=args.vocabulary_size,
        sketch_dimension=args.cka_sketch_dimension,
        seed=args.geometry_seed,
    )
    boundaries = _three_blocks(cka)
    workspace_layers = layers[boundaries[0] : boundaries[1]]
    if not workspace_layers:
        raise RuntimeError("three-block segmentation produced an empty workspace")
    block_stats = _block_statistics(cka, boundaries)
    _plot(
        args.output_dir / "lens_geometry.png",
        layers,
        cka,
        ratios,
        boundaries,
        args.dpi,
    )
    np.savetxt(args.output_dir / "cka.csv", cka, delimiter=",")
    (args.output_dir / "dictionary_token_ids.json").write_text(
        json.dumps(token_ids) + "\n", encoding="utf-8"
    )

    qualitative_metrics, readout_kurtoses = _record_qualitative_readouts(
        lens_model=lens_model,
        lens=official_lens,
        tokenizer=tokenizer,
        prompts=qualitative_prompts,
        layers=layers,
        output=args.output_dir / "qualitative_readouts.jsonl",
        max_length=args.max_length,
    )
    _plot_readout_kurtosis(
        args.output_dir / "lens_kurtosis.png",
        layers,
        readout_kurtoses,
        args.dpi,
    )
    first_third = max(1, len(layers) // 3)
    second_third = max(first_third + 1, 2 * len(layers) // 3)
    early_kurtosis = float(np.median(readout_kurtoses[:first_third]))
    workspace_kurtosis = float(np.median(readout_kurtoses[first_third:second_third]))
    late_kurtosis = float(np.median(readout_kurtoses[second_third:]))
    kurtosis_rise = _post_onset_kurtosis_rise(readout_kurtoses)
    two_hop_records = _two_hop_sanity(
        lens_model=lens_model,
        lens=official_lens,
        tokenizer=tokenizer,
        rows=selected_two_hop,
        workspace_layers=workspace_layers,
        max_length=args.max_length,
        top_k=args.two_hop_top_k,
    )
    _write_csv(args.output_dir / "two_hop_sanity.csv", two_hop_records)
    two_hop_hits = sum(bool(row["hit"]) for row in two_hop_records)
    two_hop_rate = two_hop_hits / len(two_hop_records)
    compatible = sum(bool(row["token_compatible"]) for row in two_hop_records)

    objective_clauses = {
        "lens_shape_and_provenance": True,
        "finite_geometry": bool(
            np.isfinite(cka).all()
            and np.isfinite(ratios).all()
            and np.isfinite(readout_kurtoses).all()
        ),
        "three_block_separation_positive": block_stats["within_minus_between"] > 0,
        "kurtosis_post_onset_relative_rise_at_least_10pct": kurtosis_rise["passed"],
        "two_hop_intermediate_top25_at_least_60pct": (
            two_hop_rate >= args.min_two_hop_rate
        ),
    }
    blocking_failures = [
        name for name, passed in objective_clauses.items() if not passed
    ]
    result = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "status": "failed" if blocking_failures else "manual_review_required",
        "scientifically_usable": False,
        "why_not_yet_usable": (
            "objective gate failed"
            if blocking_failures
            else "qualitative abstract/context readouts and CKA blocks need review"
        ),
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.revision,
        "lens": {
            "path": str(args.lens),
            "sha256": lens_sha256,
            "n_prompts": lens_matrices.n_prompts,
            "d_model": lens_matrices.d_model,
            "layers": layers,
        },
        "corpus": {
            "path": str(args.corpus),
            "sha256": corpus_sha256,
            "qualitative_offset": args.heldout_offset,
            "qualitative_count": len(qualitative_prompts),
        },
        "dictionary": {
            "definition": (
                "first non-special tokenizer IDs (merge-rank frequency proxy)"
            ),
            "size": len(token_ids),
            "token_id_sha256": hashlib.sha256(
                np.asarray(token_ids, dtype=np.int64).tobytes()
            ).hexdigest(),
        },
        "geometry": {
            "cka_method": "linear CKA after seeded Gaussian feature sketch",
            "cka_sketch_dimension": args.cka_sketch_dimension,
            "block_boundary_positions": list(boundaries),
            "blocks": {
                "sensory": layers[: boundaries[0]],
                "workspace": workspace_layers,
                "motor": layers[boundaries[1] :],
            },
            **block_stats,
            "participation_ratio": ratios,
            "readout_logit_excess_kurtosis": readout_kurtoses,
            "readout_kurtosis_definition": (
                "median across held-out prompts of the excess kurtosis across "
                "full-vocabulary logits for one final-position activation per layer"
            ),
            "early_kurtosis_median": early_kurtosis,
            "workspace_kurtosis_median": workspace_kurtosis,
            "late_kurtosis_median": late_kurtosis,
            "post_onset_kurtosis_rise": kurtosis_rise,
        },
        "qualitative": {
            "manual_review_required": True,
            "review_question": (
                "Do mid-layer top-10s express abstract/context tokens, while "
                "late layers converge toward the model's next-token readout?"
            ),
            **qualitative_metrics,
        },
        "two_hop": {
            "dataset": two_hop_manifest,
            "selection_seed": args.two_hop_seed,
            "rows": len(two_hop_records),
            "token_compatible_rows": compatible,
            "hits": two_hop_hits,
            "hit_rate": two_hop_rate,
            "minimum_hit_rate": args.min_two_hop_rate,
            "workspace_layers": workspace_layers,
            "definition": (
                "unspoken intermediate has rank <= 25 at any prompt position "
                "in an automatically proposed workspace layer"
            ),
        },
        "objective_clauses": objective_clauses,
        "blocking_failures": blocking_failures,
        "files": {
            "cka": "cka.csv",
            "geometry_plot": "lens_geometry.png",
            "kurtosis_plot": "lens_kurtosis.png",
            "qualitative_readouts": "qualitative_readouts.jsonl",
            "two_hop_sanity": "two_hop_sanity.csv",
        },
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-32B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--lens", required=True, type=Path)
    parser.add_argument("--expected-lens-sha256", required=True)
    parser.add_argument("--expected-lens-prompts", type=int)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--expected-corpus-sha256", required=True)
    parser.add_argument("--heldout-offset", type=int, default=2000)
    parser.add_argument("--qualitative-prompts", type=int, default=20)
    parser.add_argument("--two-hop-data", required=True, type=Path)
    parser.add_argument("--two-hop-prompts", type=int, default=20)
    parser.add_argument("--two-hop-seed", type=int, default=20260823)
    parser.add_argument("--two-hop-top-k", type=int, default=25)
    parser.add_argument("--min-two-hop-rate", type=float, default=0.60)
    parser.add_argument("--vocabulary-size", type=int, default=32_000)
    parser.add_argument("--cka-sketch-dimension", type=int, default=256)
    parser.add_argument("--geometry-seed", type=int, default=20260823)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
