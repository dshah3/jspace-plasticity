#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Freeze Qwen-tokenized WikiText chunks for independent lens fits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_SPLIT = "train"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
MODEL_ID = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(output: Path, *, chunks: int, tokens_per_chunk: int) -> dict[str, object]:
    if output.exists() or output.with_suffix(output.suffix + ".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite frozen corpus: {output}")
    if chunks < 1 or tokens_per_chunk < 2:
        raise ValueError("chunks must be positive and tokens_per_chunk must be >= 2")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    source = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
        revision=DATASET_REVISION,
    )
    token_buffer: list[int] = []
    rows: list[dict[str, object]] = []
    source_rows = 0
    for source_row, item in enumerate(source):
        source_rows = source_row + 1
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        token_buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        while len(token_buffer) >= tokens_per_chunk and len(rows) < chunks:
            token_ids = token_buffer[:tokens_per_chunk]
            del token_buffer[:tokens_per_chunk]
            rendered = tokenizer.decode(token_ids, skip_special_tokens=True)
            roundtrip = tokenizer.encode(rendered, add_special_tokens=False)
            rows.append(
                {
                    "chunk_index": len(rows),
                    "source_row_exclusive_end": source_row + 1,
                    "source_token_count": tokens_per_chunk,
                    "roundtrip_token_count": len(roundtrip),
                    "text": rendered,
                }
            )
        if len(rows) == chunks:
            break
    if len(rows) != chunks:
        raise RuntimeError(f"source yielded only {len(rows)} of {chunks} chunks")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    counts = [int(row["roundtrip_token_count"]) for row in rows]
    digest = _sha256(output)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "sha256": digest,
        "rows": len(rows),
        "tokens_per_source_chunk": tokens_per_chunk,
        "roundtrip_token_count_min": min(counts),
        "roundtrip_token_count_max": max(counts),
        "source_rows_consumed": source_rows,
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "revision": DATASET_REVISION,
            "license": "CC-BY-SA-3.0",
        },
        "tokenizer": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "partitions": {
            "lens_A": {"offset": 0, "rows": 1000},
            "lens_B": {"offset": 1000, "rows": 500},
            "lens_C": {"offset": 1500, "rows": 500},
            "heldout_calibration": {"offset": 2000, "rows": chunks - 2000},
        },
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/lens/wikitext-qwen3-32b-128tok.jsonl"),
    )
    parser.add_argument("--chunks", type=int, default=2304)
    parser.add_argument("--tokens-per-chunk", type=int, default=128)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(
                args.output,
                chunks=args.chunks,
                tokens_per_chunk=args.tokens_per_chunk,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
