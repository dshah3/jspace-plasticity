from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jspace_plasticity.lens.download_published import download


def test_download_verifies_artifact_and_writes_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {
        "lens.pt": b"lens-bytes",
        "config.yaml": b"model: exact\n",
        "convergence.csv": b"n,delta\n1,0.1\n",
    }
    manifest = {
        "source": {
            "repo_id": "owner/repo",
            "revision": "a" * 40,
            "artifact_filename": "lens.pt",
            "artifact_bytes": len(payloads["lens.pt"]),
            "artifact_sha256": hashlib.sha256(payloads["lens.pt"]).hexdigest(),
            "config_filename": "config.yaml",
            "convergence_filename": "convergence.csv",
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_download(*, filename: str, local_dir: Path, **_kwargs) -> str:
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[filename])
        return str(target)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    output = tmp_path / "download"
    result = download(manifest_path, output)
    assert result["artifact_sha256"] == manifest["source"]["artifact_sha256"]
    assert json.loads((output / "provenance.json").read_text())["downloaded"]
