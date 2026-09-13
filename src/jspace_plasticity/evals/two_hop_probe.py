"""Clean raw next-token capability gate on Anthropic's two-hop prompts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from jspace_plasticity.modeling import auto_model_class_for_config

ANTHROPIC_DATA_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
QWEN3_8B_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_probe_swap(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source_revision"] != ANTHROPIC_DATA_REVISION:
        raise ValueError("unexpected Anthropic dataset revision")
    if manifest["sha256"] != _sha256(path):
        raise ValueError("two-hop dataset SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["items"]
    if len(rows) != manifest["rows"] or len(rows) != 90:
        raise ValueError("two-hop dataset row count mismatch")
    return rows, manifest


def answer_continuation(prompt: str, answer: str) -> str:
    """Return the exact text whose first token is scored after ``prompt``."""

    if not prompt or not answer:
        raise ValueError("prompt and answer must be nonempty")
    return answer if prompt[-1].isspace() else f" {answer}"


def expected_token_id(
    tokenizer: Any, prompt: str, answer: str
) -> tuple[int | None, str, str | None]:
    continuation = answer_continuation(prompt, answer)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    joint_ids = tokenizer.encode(prompt + continuation, add_special_tokens=False)
    if joint_ids[: len(prompt_ids)] != prompt_ids:
        return None, continuation, "prompt_boundary_retokenized"
    token_ids = joint_ids[len(prompt_ids) :]
    if len(token_ids) != 1:
        return None, continuation, f"answer_token_count={len(token_ids)}"
    decoded = tokenizer.decode(token_ids)
    if decoded.strip() != answer:
        return None, continuation, f"decoded_answer={decoded!r}"
    return int(token_ids[0]), continuation, None


@torch.inference_mode()
def evaluate(
    model: Any,
    tokenizer: Any,
    source_rows: list[dict[str, str]],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records: list[dict[str, Any]] = []
    compatible: list[tuple[int, dict[str, str], int, str]] = []
    for index, row in enumerate(source_rows):
        token_id, continuation, reason = expected_token_id(
            tokenizer, row["prompt"], row["answer"]
        )
        record = {
            "example_index": index,
            **row,
            "answer_continuation": continuation,
            "token_compatible": token_id is not None,
            "incompatible_reason": reason or "",
            "expected_token_id": token_id,
            "predicted_token_id": None,
            "predicted_token": "",
            "top5_tokens": "",
            "exact": None,
        }
        records.append(record)
        if token_id is not None:
            compatible.append((index, row, token_id, continuation))

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("batched evaluation requires a PAD or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    for start in range(0, len(compatible), batch_size):
        batch = compatible[start : start + batch_size]
        encoded = tokenizer(
            [row["prompt"] for _, row, _, _ in batch],
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        logits = model(**encoded, use_cache=False).logits[:, -1].float()
        top_ids = logits.topk(5, dim=-1).indices.cpu().tolist()
        for (index, _, token_id, _), candidates in zip(
            batch, top_ids, strict=True
        ):
            predicted = int(candidates[0])
            records[index].update(
                predicted_token_id=predicted,
                predicted_token=tokenizer.decode([predicted]),
                top5_tokens=json.dumps(
                    [tokenizer.decode([candidate]) for candidate in candidates],
                    ensure_ascii=False,
                ),
                exact=float(predicted == token_id),
            )
    return records


def capability_verdict(
    records: list[dict[str, Any]],
    *,
    min_accuracy: float = 0.8,
    min_coverage: float = 0.8,
    min_correct: int = 0,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")
    compatible = [row for row in records if row["token_compatible"]]
    coverage = len(compatible) / len(records)
    accuracy = (
        sum(float(row["exact"]) for row in compatible) / len(compatible)
        if compatible
        else 0.0
    )
    correct = sum(float(row["exact"]) for row in compatible)
    clauses = {
        "tokenization_coverage": coverage >= min_coverage,
        "clean_accuracy": accuracy >= min_accuracy,
        "clean_correct_count": correct >= min_correct,
    }
    return {
        "passed": all(clauses.values()),
        "clauses": clauses,
        "rows": len(records),
        "compatible_rows": len(compatible),
        "coverage": coverage,
        "accuracy": accuracy,
        "correct_rows": int(correct),
        "min_coverage": min_coverage,
        "min_accuracy": min_accuracy,
        "min_correct": min_correct,
    }


def _category_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["category"]].append(row)
    result = {}
    for category, rows in sorted(grouped.items()):
        compatible = [row for row in rows if row["token_compatible"]]
        result[category] = {
            "rows": float(len(rows)),
            "coverage": len(compatible) / len(rows),
            "accuracy": (
                sum(float(row["exact"]) for row in compatible) / len(compatible)
                if compatible
                else 0.0
            ),
        }
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, verdict: dict[str, Any], *, checkpoint: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("token coverage", "clean accuracy")
    values = (verdict["coverage"], verdict["accuracy"])
    thresholds = (verdict["min_coverage"], verdict["min_accuracy"])
    figure, axis = plt.subplots(figsize=(6, 4))
    positions = range(len(labels))
    axis.bar(positions, values, color=("#4c78a8", "#f58518"))
    for position, threshold in zip(positions, thresholds, strict=True):
        axis.hlines(
            threshold,
            position - 0.42,
            position + 0.42,
            colors="black",
            linestyles="--",
        )
    axis.set_xticks(list(positions), labels)
    axis.set_ylim(0, 1.03)
    axis.set_ylabel("fraction")
    axis.set_title(f"{checkpoint} paper two-hop capability gate")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    source_rows, data_manifest = load_probe_swap(args.data)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, revision=args.revision
    )
    hf_config = AutoConfig.from_pretrained(
        args.checkpoint, revision=args.revision
    )
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
    records = evaluate(
        model, tokenizer, source_rows, batch_size=args.batch_size
    )
    verdict = capability_verdict(
        records,
        min_accuracy=args.min_accuracy,
        min_coverage=args.min_coverage,
        min_correct=args.min_correct,
    )
    payload = {
        "schema_version": 1,
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.revision,
        "protocol": "raw_full_vocab_greedy_next_token",
        "batch_size": args.batch_size,
        "dataset": data_manifest,
        "prompt_sha256": hashlib.sha256(
            "\n\0\n".join(row["prompt"] for row in source_rows).encode()
        ).hexdigest(),
        "capability_gate": verdict,
        "category_metrics": _category_metrics(records),
    }
    args.output_dir.mkdir(parents=True)
    _write_csv(args.output_dir / "predictions.csv", records)
    (args.output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(
        args.output_dir / "capability_gate.png",
        verdict,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default=QWEN3_8B_REVISION)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-accuracy", type=float, default=0.8)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--min-correct", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 0 <= args.min_accuracy <= 1 or not 0 <= args.min_coverage <= 1:
        parser.error("capability fractions must lie in [0, 1]")
    if args.min_correct < 0:
        parser.error("--min-correct must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
