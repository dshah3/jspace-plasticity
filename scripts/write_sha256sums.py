"""Regenerate the top-level SHA256SUMS manifest; standard library only.

Counterpart to scripts/verify_release.py. Run this after any deliberate change
to a tracked file, then re-run verify_release.py to confirm.
"""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".audit-output",
}
SKIP_NAMES = {"SHA256SUMS", ".DS_Store"}


def main() -> None:
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if relative.name in SKIP_NAMES:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        entries.append(f"{digest.hexdigest()}  {relative.as_posix()}")

    (ROOT / "SHA256SUMS").write_text("\n".join(entries) + "\n")
    print(f"Wrote {len(entries)} file hashes to SHA256SUMS.")


if __name__ == "__main__":
    main()
