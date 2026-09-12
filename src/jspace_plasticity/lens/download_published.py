"""Download and hash-verify a published lens from a frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["source"]
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, str] = {}
    for label in ("artifact", "config", "convergence"):
        filename = source[f"{label}_filename"]
        downloaded[label] = hf_hub_download(
            repo_id=source["repo_id"],
            revision=source["revision"],
            filename=filename,
            local_dir=output_dir,
        )

    artifact = Path(downloaded["artifact"])
    observed_bytes = artifact.stat().st_size
    observed_sha256 = _sha256(artifact)
    if observed_bytes != source["artifact_bytes"]:
        raise ValueError(
            f"published lens size mismatch: {observed_bytes} != "
            f"{source['artifact_bytes']}"
        )
    if observed_sha256 != source["artifact_sha256"]:
        raise ValueError(
            f"published lens SHA-256 mismatch: {observed_sha256} != "
            f"{source['artifact_sha256']}"
        )

    report = {
        "schema_version": 1,
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "downloaded": downloaded,
        "artifact_bytes": observed_bytes,
        "artifact_sha256": observed_sha256,
        "config_sha256": _sha256(Path(downloaded["config"])),
        "convergence_sha256": _sha256(Path(downloaded["convergence"])),
    }
    receipt = output_dir / "provenance.json"
    receipt.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = download(args.manifest, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
