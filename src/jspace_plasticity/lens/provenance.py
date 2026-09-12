"""Pinned provenance for published lenses used by the primary experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PublishedLensSpec:
    model_id: str
    repository_id: str
    revision: str
    artifact_subpath: str
    corpus: str
    sequence_length: int
    requested_prompts: int
    fitted_prompts: int
    dimension_batch_size: int
    dtype: str

    def write_receipt(self, destination: str | Path, artifact: str | Path) -> Path:
        """Record immutable upstream identity plus the downloaded file hash."""

        artifact_path = Path(artifact)
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        payload = {**asdict(self), "artifact_sha256": digest}
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination


QWEN3_8B_NEURONPEDIA = PublishedLensSpec(
    model_id="Qwen/Qwen3-8B",
    repository_id="neuronpedia/jacobian-lens",
    revision="4ad585e83fb62015feefce525bea9ff008023b9e",
    artifact_subpath="qwen3-8b/jlens/Salesforce-wikitext",
    corpus="Salesforce/wikitext:wikitext-103-raw-v1",
    sequence_length=128,
    requested_prompts=1_000,
    fitted_prompts=461,
    dimension_batch_size=128,
    dtype="bfloat16",
)

__all__ = ["PublishedLensSpec", "QWEN3_8B_NEURONPEDIA"]
