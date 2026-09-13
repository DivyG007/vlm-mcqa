"""Gitless deployment provenance (core7.source_snapshot)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vlm_mcqa.core7.source_snapshot import source_state, verify_snapshot


def make_snapshot(root: Path) -> tuple[Path, str]:
    commit = "a" * 40
    source = root / "src" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = root / "source_snapshot.json"
    manifest.write_text(json.dumps({
        "format": 1,
        "source_git_commit": commit,
        "files": {"src/example.py": digest},
    }), encoding="utf-8")
    return manifest, commit


class SourceSnapshotTests(unittest.TestCase):
    def test_gitless_snapshot_is_clean_and_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, commit = make_snapshot(root)
            with patch.dict(os.environ, {
                "SOURCE_GIT_COMMIT": commit,
                "SOURCE_SNAPSHOT_MANIFEST": str(manifest),
            }, clear=False):
                self.assertEqual(source_state(root), (commit, False))

    def test_changed_file_or_wrong_commit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, commit = make_snapshot(root)
            (root / "src" / "example.py").write_text("value = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_snapshot(root, manifest, expected_commit=commit)
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_snapshot(root, manifest, expected_commit="b" * 40)


if __name__ == "__main__":
    unittest.main()
