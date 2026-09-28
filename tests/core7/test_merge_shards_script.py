"""Merging per-method Core-7 shards into one run (scripts/core7/merge_shards.py)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from support import load_script

merge_shards = load_script("scripts/core7/merge_shards.py")


def make_shard(root: Path, method: str, split: str = "discovery") -> Path:
    path = root / method
    path.mkdir()
    run = {
        "status": "complete",
        "methods": [method],
        "split": split,
        "model_key": "internvl35_8b",
        "dataset_hash": "data",
        "image_aware_release_hash": "images",
        "source_git_commit": "a" * 40,
        "profile_name": "report_prelim_32",
        "profile": {"pairs": 32},
        "method_seconds": {method: 1.0},
        "wandb_url": f"https://example.test/{method}",
    }
    (path / "run.json").write_text(json.dumps(run))
    for template in merge_shards.METHOD_FILES[method]:
        name = template.format(split=split)
        if method == "heads" and split == "confirmation" and name.startswith("heads_screen_"):
            continue
        (path / name).write_text("{}\n")
    return path


class ConsolidateShardsTests(unittest.TestCase):
    def test_consolidates_disjoint_complete_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = {method: make_shard(root, method) for method in merge_shards.METHOD_FILES}
            out = root / "combined"
            manifest = merge_shards.consolidate("discovery", out, shards)
            run = json.loads((out / "run.json").read_text())
            self.assertEqual(run["methods"], ["patching", "components", "heads"])
            self.assertEqual(run["method_seconds"], {"patching": 1.0, "components": 1.0, "heads": 1.0})
            self.assertIn("patching_discovery.jsonl", manifest["files"])
            self.assertTrue((out / "heads_screen_discovery.jsonl").is_file())

    def test_rejects_mismatched_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = {method: make_shard(root, method) for method in merge_shards.METHOD_FILES}
            run_path = shards["heads"] / "run.json"
            run = json.loads(run_path.read_text())
            run["dataset_hash"] = "different"
            run_path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, "differs on dataset_hash"):
                merge_shards.consolidate("discovery", root / "combined", shards)


if __name__ == "__main__":
    unittest.main()
