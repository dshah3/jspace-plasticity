#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Vendor Anthropic's revision-pinned Jacobian-lens multihop evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

SOURCE_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
SOURCE_PATH = "data/evaluations/lens-eval-multihop.json"
SOURCE_URL = (
    "https://raw.githubusercontent.com/anthropics/jacobian-lens/"
    f"{SOURCE_REVISION}/{SOURCE_PATH}"
)
SOURCE_SHA256 = "50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416"
SOURCE_ROWS = 93
REQUIRED_FIELDS = {"name", "prompt", "target", "intermediates"}


def validate_payload(raw: bytes) -> list[dict[str, Any]]:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {SOURCE_SHA256}, got {digest}"
        )
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {"items"}:
        raise ValueError("multihop evaluation root must contain only items")
    items = payload["items"]
    if not isinstance(items, list) or len(items) != SOURCE_ROWS:
        raise ValueError(
            f"multihop evaluation must contain exactly {SOURCE_ROWS} items"
        )
    names: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != REQUIRED_FIELDS:
            raise ValueError(f"invalid fields at item {index}")
        for field in ("name", "prompt", "target"):
            if not isinstance(item[field], str) or not item[field]:
                raise ValueError(f"item {index} has an invalid {field}")
        intermediates = item["intermediates"]
        if not isinstance(intermediates, list) or not intermediates:
            raise ValueError(f"item {index} has no intermediates")
        if any(not isinstance(value, str) or not value for value in intermediates):
            raise ValueError(f"item {index} has an invalid intermediate")
        if item["name"] in names:
            raise ValueError(f"duplicate item name: {item['name']}")
        names.add(item["name"])
    return items


def vendor(output: Path, *, source: Path | None = None) -> dict[str, Any]:
    if source is None:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:  # noqa: S310
            raw = response.read()
    else:
        raw = source.read_bytes()
    items = validate_payload(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != raw:
        raise FileExistsError(f"refusing to overwrite different data: {output}")
    output.write_bytes(raw)
    manifest = {
        "schema_version": 1,
        "dataset": "anthropics/jacobian-lens lens-eval-multihop",
        "source_repository": "https://github.com/anthropics/jacobian-lens",
        "source_revision": SOURCE_REVISION,
        "source_path": SOURCE_PATH,
        "source_url": SOURCE_URL,
        "license": "Apache-2.0",
        "rows": len(items),
        "sha256": SOURCE_SHA256,
        "official_metric_note": (
            "The upstream README defines this as a lens-readout evaluation. "
            "This repository additionally scores target as the raw next-token "
            "continuation for the ablation positive control."
        ),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evals/anthropic-lens-eval-multihop.json"),
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Optional local byte-for-byte source; the pinned SHA is still enforced.",
    )
    args = parser.parse_args()
    print(json.dumps(vendor(args.output, source=args.source), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
