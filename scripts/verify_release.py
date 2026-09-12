"""Verify the top-level relative SHA256SUMS; standard library only."""

import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lines = (root / "SHA256SUMS").read_text().splitlines()
for line in lines:
    expected, relative = line.split("  ", 1)
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError("Unsafe manifest path")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Hash mismatch: {relative}")
print(f"All {len(lines)} file hashes verified.")
