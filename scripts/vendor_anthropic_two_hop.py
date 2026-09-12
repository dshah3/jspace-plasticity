#!/usr/bin/env python3
"""Vendor Anthropic's revision-pinned two-hop probe-swap evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

SOURCE_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
SOURCE_PATH = "data/experiments/probe-swap.json"
SOURCE_URL = (
    "https://raw.githubusercontent.com/anthropics/jacobian-lens/"
    f"{SOURCE_REVISION}/{SOURCE_PATH}"
)
SOURCE_SHA256 = "a0edd27ca23f7b4d0fbe90448c2ddcc7457a3d812121bf024ed12a032ff86796"
REQUIRED_FIELDS = {
    "name",
    "category",
    "prompt",
    "intermediate",
    "answer",
    "swap_to",
    "swap_answer",
}


def validate_payload(raw: bytes) -> list[dict[str, Any]]:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {SOURCE_SHA256}, got {digest}"
        )
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {"items"}:
        raise ValueError("probe-swap root must contain only items")
    items = payload["items"]
    if not isinstance(items, list) or len(items) != 90:
        raise ValueError("probe-swap must contain exactly 90 items")
    names: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != REQUIRED_FIELDS:
            raise ValueError(f"invalid fields at item {index}")
        if any(not isinstance(value, str) or not value for value in item.values()):
            raise ValueError(f"item {index} contains a nonempty-string violation")
        if item["name"] in names:
            raise ValueError(f"duplicate item name: {item['name']}")
        names.add(item["name"])
    return items


def vendor(output: Path) -> dict[str, Any]:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:  # noqa: S310
        raw = response.read()
    items = validate_payload(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != raw:
        raise FileExistsError(f"refusing to overwrite different data: {output}")
    output.write_bytes(raw)
    manifest = {
        "schema_version": 1,
        "dataset": "anthropics/jacobian-lens probe-swap",
        "source_repository": "https://github.com/anthropics/jacobian-lens",
        "source_revision": SOURCE_REVISION,
        "source_path": SOURCE_PATH,
        "source_url": SOURCE_URL,
        "license": "Apache-2.0",
        "rows": len(items),
        "sha256": SOURCE_SHA256,
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
        default=Path("data/evals/anthropic-probe-swap.json"),
    )
    args = parser.parse_args()
    print(json.dumps(vendor(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
