#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Build the frozen GeoNames geography-composition confirmation set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SOURCE_FILES = {
    "cities15000.zip": (
        "d5c5cdab8f5bc46cf13a93a64a92d0cdfe48235fe82fac208be8bbbf550e5185"
    ),
    "countryInfo.txt": (
        "93bafc525813f22e4711ff9ed6d626343094ce48c26388dc7c49189b3d7d5512"
    ),
    "iso-languagecodes.txt": (
        "cb0d34f492775deec8ec5713da6efa4463dad99b5e7ba2172bd094cfdcb76571"
    ),
    "readme.txt": (
        "b1957379b6c1242c700c98ac9a8aa0a09f56c3c0a50ee72175527005f48ef2c5"
    ),
}
SOURCE_URL_ROOT = "https://download.geonames.org/export/dump"
SNAPSHOT_DATE = "2026-08-21"
MIN_CITY_POPULATION = 500_000
FEATURE_CODES = frozenset({"PPL", "PPLA", "PPLA2"})
CONTINENT_NAMES = {
    "AF": "Africa",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "Americas",
    "SA": "Americas",
    "OC": "Oceania",
}
CONTINENT_QUOTAS = {"AF": 7, "AS": 7, "EU": 7, "NA": 3, "SA": 7, "OC": 1}
RELATIONS = {
    "country": (
        "Fact: The country containing {city} is",
        "country",
    ),
    "capital": (
        "Fact: The capital of the country containing {city} is",
        "capital",
    ),
    "currency": (
        "Fact: The three-letter currency code of the country containing {city} is",
        "currency_code",
    ),
    "region": (
        "Fact: The broad world region containing the country that contains {city} is",
        "continent_name",
    ),
}

COUNTRY_FIELDS = (
    "iso",
    "iso3",
    "iso_numeric",
    "fips",
    "country",
    "capital",
    "area",
    "population",
    "continent",
    "tld",
    "currency_code",
    "currency_name",
    "phone",
    "postal_code_format",
    "postal_code_regex",
    "languages",
    "geonameid",
    "neighbours",
    "equivalent_fips_code",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_sources(source_dir: Path) -> None:
    for name, expected in SOURCE_FILES.items():
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"GeoNames source hash mismatch for {name}: {actual} != {expected}"
            )


def _country_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        lines = (line for line in handle if not line.startswith("#"))
        rows = csv.DictReader(lines, delimiter="\t", fieldnames=COUNTRY_FIELDS)
        return {row["iso"]: row for row in rows}


def _city_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != ["cities15000.txt"]:
            raise ValueError(f"unexpected cities archive members: {names}")
        with archive.open(names[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            return list(csv.reader(text, delimiter="\t"))


def select_entities(source_dir: Path) -> list[dict[str, Any]]:
    """Apply the frozen, model-independent selection rule."""

    countries = _country_rows(source_dir / "countryInfo.txt")
    city_rows = _city_rows(source_dir / "cities15000.zip")
    ascii_name_counts = Counter(
        row[2].casefold() for row in city_rows if row[2].isascii()
    )
    by_country: dict[str, list[list[str]]] = defaultdict(list)
    for row in city_rows:
        ascii_name = row[2]
        population = int(row[14] or 0)
        if (
            ascii_name.isascii()
            and ascii_name_counts[ascii_name.casefold()] == 1
            and population >= MIN_CITY_POPULATION
            and row[7] in FEATURE_CODES
        ):
            by_country[row[8]].append(row)

    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for iso, country in countries.items():
        if iso not in by_country or country["continent"] not in CONTINENT_NAMES:
            continue
        # Qwen token compatibility is checked only at evaluation time. These
        # source-level constraints merely avoid guaranteed multi-word answers.
        if not (
            country["country"].isascii()
            and country["country"].isalpha()
            and country["capital"].isascii()
            and country["capital"].isalpha()
            and len(country["currency_code"]) == 3
            and country["currency_code"].isalpha()
        ):
            continue
        non_capitals = [
            row
            for row in by_country[iso]
            if row[2].casefold() != country["capital"].casefold()
        ]
        if not non_capitals:
            continue
        city = max(non_capitals, key=lambda row: (int(row[14] or 0), row[2]))
        eligible[country["continent"]].append(
            {
                "iso": iso,
                "iso3": country["iso3"],
                "country": country["country"],
                "capital": country["capital"],
                "currency_code": country["currency_code"].upper(),
                "continent_code": country["continent"],
                "continent_name": CONTINENT_NAMES[country["continent"]],
                "country_population": int(country["population"] or 0),
                "city": city[2],
                "city_geonameid": int(city[0]),
                "city_population": int(city[14] or 0),
                "city_feature_code": city[7],
            }
        )

    selected: list[dict[str, Any]] = []
    for continent, quota in CONTINENT_QUOTAS.items():
        candidates = sorted(
            eligible[continent],
            key=lambda row: (-row["country_population"], row["iso"]),
        )
        if len(candidates) < quota:
            raise ValueError(
                f"not enough eligible {continent} countries: "
                f"{len(candidates)} < {quota}"
            )
        selected.extend(candidates[:quota])
    selected.sort(key=lambda row: row["iso"])
    if len(selected) != 32 or len({row["iso"] for row in selected}) != 32:
        raise AssertionError("selection must contain 32 distinct countries")
    return selected


def build_payload(source_dir: Path) -> dict[str, Any]:
    validate_sources(source_dir)
    entities = select_entities(source_dir)
    items = []
    for entity in entities:
        for relation, (template, answer_field) in RELATIONS.items():
            items.append(
                {
                    "name": f"{entity['iso'].lower()}-{relation}",
                    "relation": relation,
                    "prompt": template.format(city=entity["city"]),
                    "answer": entity[answer_field],
                    **entity,
                }
            )
    return {
        "schema_version": 1,
        "dataset": "GeoNames geography-composition confirmation",
        "split": "clean_confirmation_eval_only",
        "selection": {
            "model_independent": True,
            "min_city_population": MIN_CITY_POPULATION,
            "city_feature_codes": sorted(FEATURE_CODES),
            "city_ascii_name_globally_unique": True,
            "exclude_capital_as_city": True,
            "single_ascii_word_country_and_capital": True,
            "rank_within_country": "descending city population, then ASCII city name",
            "rank_within_continent": "descending country population, then ISO code",
            "continent_quotas": CONTINENT_QUOTAS,
        },
        "relations": sorted(RELATIONS),
        "entities": entities,
        "items": items,
    }


def write_dataset(
    source_dir: Path, output: Path, *, replace: bool = False
) -> dict[str, Any]:
    payload = build_payload(source_dir)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if output.exists() and output.read_bytes() != encoded and not replace:
        raise FileExistsError(f"refusing to overwrite different dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    manifest = {
        "schema_version": 1,
        "dataset": payload["dataset"],
        "snapshot_date": SNAPSHOT_DATE,
        "source": "GeoNames Gazetteer extract",
        "source_url_root": SOURCE_URL_ROOT,
        "source_license": "CC-BY-4.0",
        "source_files": {
            name: {"sha256": digest, "url": f"{SOURCE_URL_ROOT}/{name}"}
            for name, digest in SOURCE_FILES.items()
        },
        "entities": len(payload["entities"]),
        "rows": len(payload["items"]),
        "relations": payload["relations"],
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=Path("data/geography/source")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/geography/geography-confirmation.json"),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace a different derived file after an explicitly reviewed change",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            write_dataset(args.source_dir, args.output, replace=args.replace), indent=2
        )
    )


if __name__ == "__main__":
    main()
