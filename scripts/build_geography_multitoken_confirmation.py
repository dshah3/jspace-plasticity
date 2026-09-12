#!/usr/bin/env python3
"""Build the fresh held-out multi-token geography confirmation set."""

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

from build_geography_confirmation import COUNTRY_FIELDS, FEATURE_CODES, validate_sources

PRIOR_DATA_SHA256 = (
    "84452211a8e50e10f2b2d427aa6fa47df271956e90b356bd63e498b215488731"
)
MIN_CITY_POPULATION = 250_000
CONTINENT_QUOTAS = {"AF": 8, "AS": 9, "EU": 8, "NA": 4, "SA": 2, "OC": 1}
RELATIONS = {
    "capital": (
        "Question: What is the capital of the country containing {city}?\nAnswer:",
        "capital",
    ),
    "currency": (
        "Question: What is the three-letter currency code of the country "
        "containing {city}?\nAnswer:",
        "currency_code",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _country_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        lines = (line for line in handle if not line.startswith("#"))
        rows = csv.DictReader(lines, delimiter="\t", fieldnames=COUNTRY_FIELDS)
        return {row["iso"]: row for row in rows}


def _city_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["cities15000.txt"]:
            raise ValueError("unexpected cities15000 archive members")
        with archive.open("cities15000.txt") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            return list(csv.reader(text, delimiter="\t"))


def _load_prior_isos(path: Path) -> set[str]:
    if _sha256(path) != PRIOR_DATA_SHA256:
        raise ValueError("prior geography dataset SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    isos = {row["iso"] for row in payload["entities"]}
    if len(isos) != 32:
        raise ValueError("prior geography dataset must contain 32 entities")
    return isos


def select_entities(source_dir: Path, prior_data: Path) -> list[dict[str, Any]]:
    """Select fresh countries and cities without consulting model behavior."""

    validate_sources(source_dir)
    excluded_isos = _load_prior_isos(prior_data)
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
        continent = country["continent"]
        if iso in excluded_isos or continent not in CONTINENT_QUOTAS:
            continue
        if iso not in by_country:
            continue
        if not (
            country["capital"]
            and country["capital"].isascii()
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
        eligible[continent].append(
            {
                "iso": iso,
                "iso3": country["iso3"],
                "country": country["country"],
                "capital": country["capital"],
                "currency_code": country["currency_code"].upper(),
                "continent_code": continent,
                "country_population": int(country["population"] or 0),
                "city": city[2],
                "city_geonameid": int(city[0]),
                "city_population": int(city[14] or 0),
                "city_feature_code": city[7],
            }
        )

    selected = []
    for continent, quota in CONTINENT_QUOTAS.items():
        candidates = sorted(
            eligible[continent],
            key=lambda row: (-row["country_population"], row["iso"]),
        )
        if len(candidates) < quota:
            raise ValueError(
                f"not enough fresh {continent} countries: "
                f"{len(candidates)} < {quota}"
            )
        selected.extend(candidates[:quota])
    selected.sort(key=lambda row: row["iso"])
    if len(selected) != 32 or {row["iso"] for row in selected} & excluded_isos:
        raise AssertionError("fresh selection contract failed")
    return selected


def build_payload(source_dir: Path, prior_data: Path) -> dict[str, Any]:
    entities = select_entities(source_dir, prior_data)
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
        "dataset": "GeoNames geography multi-token confirmation",
        "split": "fresh_clean_confirmation_eval_only",
        "prior_dataset_sha256": PRIOR_DATA_SHA256,
        "selection": {
            "model_independent": True,
            "exclude_all_prior_countries": True,
            "min_city_population": MIN_CITY_POPULATION,
            "city_feature_codes": sorted(FEATURE_CODES),
            "city_ascii_name_globally_unique": True,
            "exclude_capital_as_city": True,
            "rank_within_country": "descending city population, then ASCII city name",
            "rank_within_continent": "descending country population, then ISO code",
            "continent_quotas": CONTINENT_QUOTAS,
        },
        "relations": sorted(RELATIONS),
        "entities": entities,
        "items": items,
    }


def write_dataset(source_dir: Path, prior_data: Path, output: Path) -> dict[str, Any]:
    payload = build_payload(source_dir, prior_data)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if output.exists() and output.read_bytes() != encoded:
        raise FileExistsError(f"refusing to overwrite different dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    manifest = {
        "schema_version": 1,
        "dataset": payload["dataset"],
        "source": "GeoNames Gazetteer extract, snapshot 2026-08-21",
        "source_license": "CC-BY-4.0",
        "prior_dataset_sha256": PRIOR_DATA_SHA256,
        "entities": len(payload["entities"]),
        "rows": len(payload["items"]),
        "relations": payload["relations"],
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=Path("data/geography/source")
    )
    parser.add_argument(
        "--prior-data",
        type=Path,
        default=Path("data/geography/geography-confirmation.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/geography/geography-multitoken-confirmation.json"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            write_dataset(args.source_dir, args.prior_data, args.output), indent=2
        )
    )


if __name__ == "__main__":
    main()
