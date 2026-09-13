#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Build city-disjoint country-probe rows from the frozen GeoNames snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jspace_plasticity.tasks.closedbook_geo import (
    FEATURE_CODES,
    PROMPT_TEMPLATES,
    SOURCE_SHA256,
    _city_rows,
    sha256_path,
    validate_sources,
)

SCHEMA_VERSION = 1
CITIES_PER_COUNTRY = 4
RELATIONS = ("capital", "currency", "region")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def build_rows(
    source_dir: Path, task_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_sources(source_dir)
    task_rows = _read_jsonl(task_path)
    heldout: dict[str, dict[str, Any]] = {}
    for row in task_rows:
        if row["split"] in {"val", "screen"}:
            heldout.setdefault(row["iso"], row)
    if len(heldout) != 52:
        raise ValueError(f"expected 52 held-out countries, got {len(heldout)}")

    city_rows = _city_rows(source_dir / "cities15000.zip")
    ascii_counts = Counter(row[2].casefold() for row in city_rows if row[2].isascii())
    by_country: dict[str, list[list[str]]] = defaultdict(list)
    for row in city_rows:
        name = row[2]
        if (
            name.isascii()
            and ascii_counts[name.casefold()] == 1
            and row[7] in FEATURE_CODES
            and int(row[14] or 0) >= 15_000
        ):
            by_country[row[8]].append(row)

    rows: list[dict[str, Any]] = []
    excluded = []
    for iso, task in sorted(heldout.items()):
        candidates = [
            row
            for row in by_country[iso]
            if int(row[0]) != int(task["city_geonameid"])
            and row[2].casefold() != str(task["capital"]).casefold()
        ]
        candidates.sort(key=lambda row: (-int(row[14] or 0), row[2], int(row[0])))
        if len(candidates) < CITIES_PER_COUNTRY:
            excluded.append(
                {
                    "iso": iso,
                    "country": task["country"],
                    "eligible_cities": len(candidates),
                }
            )
            continue
        chosen = candidates[:CITIES_PER_COUNTRY]
        chosen.sort(
            key=lambda row: hashlib.sha256(f"20260829:{iso}:{row[0]}".encode()).digest()
        )
        for city_fold, city in enumerate(chosen):
            for relation in RELATIONS:
                template = PROMPT_TEMPLATES[relation][0]
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "source_id": f"{iso.casefold()}-probe-c{city_fold}-{relation}",
                        "cluster_id": iso,
                        "country": task["country"],
                        "city": city[2],
                        "city_geonameid": int(city[0]),
                        "city_population": int(city[14] or 0),
                        "city_fold": city_fold,
                        "relation": relation,
                        "prompt": template.format(city=city[2]),
                    }
                )
    if len(excluded) != 1 or excluded[0]["iso"] != "GQ":
        raise ValueError(f"unexpected probe-country exclusions: {excluded}")
    expected_rows = 51 * CITIES_PER_COUNTRY * len(RELATIONS)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} probe rows, got {len(rows)}")
    rows.sort(key=lambda row: row["source_id"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "city-disjoint supervised decoding of the unspoken country",
        "rows": len(rows),
        "countries": 51,
        "cities_per_country": CITIES_PER_COUNTRY,
        "relations": list(RELATIONS),
        "folds": list(range(CITIES_PER_COUNTRY)),
        "heldout_country_splits": ["val", "screen"],
        "excluded_countries": excluded,
        "task_sha256": sha256_path(task_path),
        "source_sha256": SOURCE_SHA256,
        "split_rule": (
            "outer leave-one-city-fold-out; no city appears in probe train and test"
        ),
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, manifest = build_rows(args.source_dir, args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest["sha256"] = sha256_path(args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
