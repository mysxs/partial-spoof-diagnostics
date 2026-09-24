#!/usr/bin/env python3
"""Verify the SHA-256 records shipped with this GitHub release."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("MANIFEST.sha256.json"))
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    root = manifest.parent
    files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    failures = []
    for name, expected in sorted(files.items()):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(f"MISSING/INVALID {name}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            failures.append(f"MISMATCH {name}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(f"FAIL: {len(failures)} of {len(files)} files")
    print(f"PASS: {len(files)} files match {manifest.name}")


if __name__ == "__main__":
    main()
