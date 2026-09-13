"""Verify source provenance from Git or a gitless SHA-256 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(root: Path, manifest_path: Path, *, expected_commit: str | None = None) -> dict[str, Any]:
    """Verify every tracked file recorded in a gitless deployment manifest."""
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"source snapshot manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    commit = manifest.get("source_git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("source snapshot has no full 40-character Git commit")
    if expected_commit and commit != expected_commit:
        raise ValueError(f"snapshot commit {commit} differs from expected {expected_commit}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("source snapshot contains no tracked files")
    for relative, expected_hash in sorted(files.items()):
        posix = PurePosixPath(relative)
        if posix.is_absolute() or ".." in posix.parts:
            raise ValueError(f"unsafe source snapshot path: {relative}")
        path = root.joinpath(*posix.parts)
        if not path.is_file():
            raise ValueError(f"tracked source file missing: {relative}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"tracked source file changed: {relative}")
    return manifest


def source_state(root: Path) -> tuple[str, bool]:
    """Return ``(commit, dirty)`` from Git, or verify the gitless manifest."""
    root = Path(root).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"], check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        expected = os.environ.get("SOURCE_GIT_COMMIT")
        manifest_path = os.environ.get("SOURCE_SNAPSHOT_MANIFEST")
        if expected and manifest_path:
            manifest = verify_snapshot(root, Path(manifest_path), expected_commit=expected)
            return manifest["source_git_commit"], False
        if expected or manifest_path:
            raise ValueError(
                "set both SOURCE_GIT_COMMIT and SOURCE_SNAPSHOT_MANIFEST"
            )
        return "submitted-zip", False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    commit, dirty = source_state(args.root)
    if dirty:
        raise SystemExit("source tree is dirty")
    print(commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
