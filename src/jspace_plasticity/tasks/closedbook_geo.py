"""Frozen closed-book factual chains derived from the GeoNames snapshot.

Every prompt identifies a city, requiring the model to recover its country as
an unspoken intermediate before reporting that country's capital, currency
code, or continent.  No supporting facts or chain of thought are supplied.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SELECTION_SEED = 20_260_825
ENTITY_SPLITS = {"train": 56, "val": 26, "screen": 26}
MIN_CITY_POPULATION = 150_000
FEATURE_CODES = frozenset({"PPL", "PPLA", "PPLA2"})
CONTINENT_NAMES = {
    "AF": "Africa",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "Americas",
    "SA": "Americas",
    "OC": "Oceania",
}
SOURCE_SHA256 = {
    "cities15000.zip": (
        "d5c5cdab8f5bc46cf13a93a64a92d0cdfe48235fe82fac208be8bbbf550e5185"
    ),
    "countryInfo.txt": (
        "93bafc525813f22e4711ff9ed6d626343094ce48c26388dc7c49189b3d7d5512"
    ),
}
EXPECTED_EXCLUSION_SHA256 = {
    "anthropic-probe-swap.json": (
        "a0edd27ca23f7b4d0fbe90448c2ddcc7457a3d812121bf024ed12a032ff86796"
    ),
    "anthropic-lens-eval-multihop.json": (
        "50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416"
    ),
    "geography-confirmation.json": (
        "84452211a8e50e10f2b2d427aa6fa47df271956e90b356bd63e498b215488731"
    ),
    "geography-multitoken-confirmation.json": (
        "697f8b89ad0050e3072fb242f4d8c143bf28d2932c8564c76c0bb42ad7fc813e"
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

PROMPT_TEMPLATES = {
    "capital": (
        "Fact: The capital of the country containing {city} is",
        "Fact: The city that is the national capital of the country where "
        "{city} is located is",
        "Fact: {city} is in a country whose capital is",
        "Fact: The seat of government of the nation containing {city} is",
    ),
    "currency": (
        "Fact: The three-letter currency code used in the country containing {city} is",
        "Fact: The ISO currency code of the nation where {city} is located is",
        "Fact: {city} is in a country whose currency code is",
        "Fact: The currency abbreviation for the country that contains {city} is",
    ),
    "region": (
        "Fact: The broad world region containing the country where {city} is "
        "located is",
        "Fact: The country containing {city} is in the broad world region called",
        "Fact: {city} is in a nation located in the broad world region called",
        "Fact: The broad world region of the nation in which {city} lies is",
    ),
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def validate_sources(source_dir: Path) -> None:
    for name, expected in SOURCE_SHA256.items():
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_path(path) != expected:
            raise ValueError(f"GeoNames source SHA-256 mismatch: {name}")


def _country_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        lines = (line for line in handle if not line.startswith("#"))
        rows = csv.DictReader(lines, delimiter="\t", fieldnames=COUNTRY_FIELDS)
        return {str(row["iso"]): dict(row) for row in rows}


def _city_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["cities15000.txt"]:
            raise ValueError("unexpected cities15000 archive members")
        with archive.open("cities15000.txt") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            return list(csv.reader(text, delimiter="\t"))


def exclusion_evidence(
    paths: list[Path],
) -> tuple[set[str], set[int], dict[str, str]]:
    """Return forbidden semantic values and prior clue IDs with exact hashes."""

    semantic_values: set[str] = set()
    city_ids: set[int] = set()
    hashes: dict[str, str] = {}
    for path in paths:
        expected = EXPECTED_EXCLUSION_SHA256.get(path.name)
        observed = sha256_path(path)
        if expected is None or observed != expected:
            raise ValueError(f"unexpected exclusion artifact: {path}")
        hashes[path.name] = observed
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path.name.startswith("anthropic-"):
            for row in payload.get("items", []):
                for field in (
                    "answer",
                    "intermediate",
                    "swap_to",
                    "swap_answer",
                    "target",
                ):
                    value = row.get(field)
                    if isinstance(value, str) and value.strip():
                        semantic_values.add(normalize_text(value))
                for value in row.get("intermediates", []):
                    if isinstance(value, str) and value.strip():
                        semantic_values.add(normalize_text(value))
        else:
            for entity in payload.get("entities", []):
                city_id = entity.get("city_geonameid")
                if isinstance(city_id, int):
                    city_ids.add(city_id)
    return semantic_values, city_ids, hashes


def select_entities(
    source_dir: Path,
    exclusion_paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    validate_sources(source_dir)
    forbidden_values, forbidden_city_ids, exclusion_hashes = exclusion_evidence(
        exclusion_paths
    )
    countries = _country_rows(source_dir / "countryInfo.txt")
    city_rows = _city_rows(source_dir / "cities15000.zip")
    ascii_counts = Counter(row[2].casefold() for row in city_rows if row[2].isascii())
    by_country: dict[str, list[list[str]]] = defaultdict(list)
    for row in city_rows:
        name = row[2]
        population = int(row[14] or 0)
        if (
            name.isascii()
            and ascii_counts[name.casefold()] == 1
            and population >= MIN_CITY_POPULATION
            and row[7] in FEATURE_CODES
            and int(row[0]) not in forbidden_city_ids
        ):
            by_country[row[8]].append(row)

    eligible: list[dict[str, Any]] = []
    for iso, country in countries.items():
        if iso not in by_country or country["continent"] not in CONTINENT_NAMES:
            continue
        if not country["capital"] or not country["currency_code"]:
            continue
        city_candidates = [
            row
            for row in by_country[iso]
            if row[2].casefold() != country["capital"].casefold()
        ]
        if not city_candidates:
            continue
        city = max(city_candidates, key=lambda row: (int(row[14] or 0), row[2]))
        semantic_fields = (country["country"], country["capital"], city[2])
        if any(normalize_text(value) in forbidden_values for value in semantic_fields):
            continue
        eligible.append(
            {
                "iso": iso,
                "country": country["country"],
                "capital": country["capital"],
                "currency_code": country["currency_code"].upper(),
                "continent_code": country["continent"],
                "continent": CONTINENT_NAMES[country["continent"]],
                "country_population": int(country["population"] or 0),
                "city": city[2],
                "city_geonameid": int(city[0]),
                "city_population": int(city[14] or 0),
            }
        )
    required = sum(ENTITY_SPLITS.values())
    eligible.sort(key=lambda row: (-row["city_population"], row["iso"]))
    if len(eligible) < required:
        raise ValueError(f"only {len(eligible)} eligible entities; need {required}")
    selected = eligible[:required]
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"{SELECTION_SEED}:{row['iso']}:{row['city_geonameid']}".encode()
        ).digest()
    )
    return selected, exclusion_hashes


def build_rows(
    source_dir: Path,
    exclusion_paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entities, exclusion_hashes = select_entities(source_dir, exclusion_paths)
    split_by_iso: dict[str, str] = {}
    offset = 0
    for split, count in ENTITY_SPLITS.items():
        for entity in entities[offset : offset + count]:
            split_by_iso[entity["iso"]] = split
        offset += count
    answer_fields = {
        "capital": "capital",
        "currency": "currency_code",
        "region": "continent",
    }
    rows: list[dict[str, Any]] = []
    for entity in entities:
        split = split_by_iso[entity["iso"]]
        for relation, templates in PROMPT_TEMPLATES.items():
            for template_index, template in enumerate(templates):
                answer = entity[answer_fields[relation]]
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "source_id": (
                            f"{entity['iso'].lower()}-{relation}-t{template_index}"
                        ),
                        "split": split,
                        "cluster_id": entity["iso"],
                        "relation": relation,
                        "template_id": f"{relation}-t{template_index}",
                        "prompt": template.format(city=entity["city"]),
                        "completion": f" {answer}",
                        "answer": answer,
                        "intermediate": entity["country"],
                        **entity,
                    }
                )
    rows.sort(key=lambda row: (row["split"], row["source_id"]))
    split_rows = Counter(row["split"] for row in rows)
    split_entities = Counter(split_by_iso.values())
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "closed_book_raw_greedy_next_token_no_cot",
        "selection_seed": SELECTION_SEED,
        "selection_model_independent": True,
        "source_sha256": SOURCE_SHA256,
        "exclusion_sha256": exclusion_hashes,
        "entity_splits": dict(ENTITY_SPLITS),
        "observed_split_entities": dict(split_entities),
        "split_rows": dict(split_rows),
        "relations": sorted(PROMPT_TEMPLATES),
        "templates_per_relation": {
            relation: len(templates) for relation, templates in PROMPT_TEMPLATES.items()
        },
        "rows": len(rows),
        "train_split": "train",
        "evaluation_only_splits": ["screen", "val"],
    }
    return rows, metadata


def validate_manifest(
    manifest: Mapping[str, Any], *, expected_data_sha256: str
) -> None:
    if manifest.get("sha256") != expected_data_sha256:
        raise ValueError("closed-book manifest is bound to another dataset")
    if manifest.get("protocol") != "closed_book_raw_greedy_next_token_no_cot":
        raise ValueError("unexpected closed-book factual protocol")
    if manifest.get("entity_splits") != ENTITY_SPLITS:
        raise ValueError("unexpected closed-book entity split sizes")
    if manifest.get("source_sha256") != SOURCE_SHA256:
        raise ValueError("unexpected closed-book source provenance")
    if set(manifest.get("exclusion_sha256", {}).values()) != set(
        EXPECTED_EXCLUSION_SHA256.values()
    ):
        raise ValueError("unexpected closed-book exclusion provenance")


def load_split(
    data_path: Path,
    *,
    expected_sha256: str,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sha256_path(data_path) != expected_sha256:
        raise ValueError("closed-book factual dataset SHA-256 mismatch")
    manifest_path = data_path.with_suffix(data_path.suffix + ".manifest.json")
    portable_manifest = data_path.with_suffix(".manifest.json")
    if manifest_path.exists() and portable_manifest.exists():
        if json.loads(manifest_path.read_text()) != json.loads(
            portable_manifest.read_text()
        ):
            raise ValueError("conflicting closed-book dataset manifests")
    elif not manifest_path.exists() and portable_manifest.exists():
        manifest_path = portable_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, expected_data_sha256=expected_sha256)
    if split not in {"train", "val", "screen"}:
        raise ValueError(f"unknown closed-book split: {split}")
    rows = [
        json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("closed-book dataset row count mismatch")
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("closed-book dataset contains duplicate source IDs")
    split_clusters: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        row_split = row.get("split")
        split_clusters[row_split].add(row["cluster_id"])
        if (
            not row["prompt"].startswith("Fact: ")
            or row["prompt"] != row["prompt"].rstrip()
        ):
            raise ValueError("closed-book prompt contract failed")
        if row["intermediate"].casefold() in row["prompt"].casefold():
            raise ValueError("closed-book prompt reveals its intermediate")
        if row["completion"] != f" {row['answer']}":
            raise ValueError("closed-book completion contract failed")
    split_names = sorted(split_clusters)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            if split_clusters[left] & split_clusters[right]:
                raise ValueError("closed-book entity clusters cross splits")
    selected = [row for row in rows if row["split"] == split]
    if len(selected) != int(manifest.get("split_rows", {}).get(split, -1)):
        raise ValueError("closed-book selected split row count mismatch")
    return selected, manifest
