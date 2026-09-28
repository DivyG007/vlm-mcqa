"""Pre-run readiness checks for a full screen (scripts/core7/validate_screening.py)."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from support import HAS_TORCH, load_script, requires_torch

if HAS_TORCH:
    from vlm_mcqa.core7.report import select
    from vlm_mcqa.core7.session import load_profiles

validate_screening = load_script("scripts/core7/validate_screening.py")


class FullScreenValidationTests(unittest.TestCase):
    def test_rejects_mixed_dataset_or_release_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "example"
            directory.mkdir()
            run = {"model_key": "example", "status": "complete", "dataset_hash": "old",
                   "profile_name": "full_core7", "source_git_commit": "abc", "source_tree_dirty": False}
            (directory / "run.json").write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, "dataset_hash"):
                validate_screening.check_run(root, "example", dataset_hash="new", commit="abc",
                                               profile="full_core7")
            run["dataset_hash"] = "new"
            run["image_aware_release_hash"] = "old-images"
            (directory / "run.json").write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, "image-aware release hash"):
                validate_screening.check_run(root, "example", dataset_hash="new", commit="abc",
                                               profile="full_core7", release_hash="new-images")

    @requires_torch
    def test_stored_selection_must_match_screening_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys = {f"model_{i}" for i in range(12)}
            for i, key in enumerate(sorted(keys)):
                directory = root / key
                directory.mkdir()
                (directory / "run.json").write_text(json.dumps({
                    "model_key": key, "family": ("qwen_vl", "internvl", "llava_onevision", "paligemma")[i // 3],
                    "params_b": 2.0 + i, "lm_backbone": "qwen2"}))
                (directory / "behavior_screening_summary.json").write_text(json.dumps({
                    "worst_position_accuracy": 0.94, "letters_accuracy": 0.95,
                    "worst_scheme_position_accuracy": 0.5, "clean_minus_shuffled": 0.5,
                    "pair_yield": {"discovery": {"usable_pairs": 70},
                                   "confirmation": {"usable_pairs": 55}}}))
                (directory / "calibration.json").write_text(json.dumps({
                    "all_passed": True, "gates": {"identity": {"value": 0.0, "limit": 0.25}}}))
            output = root / "_selection"
            select([root / key for key in sorted(keys)], load_profiles()["selection"], output)
            validate_screening.check_selection(root, output / "selection.json", keys)
            recorded = json.loads((output / "selection.json").read_text())
            recorded["core7_models"].reverse()
            (output / "selection.json").write_text(json.dumps(recorded))
            with self.assertRaisesRegex(ValueError, "selection differs"):
                validate_screening.check_selection(root, output / "selection.json", keys)

    def test_dataset_preflight_checks_splits_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            config = root / "configs/core7"
            config.mkdir(parents=True)
            sizes = {"calibration": 1, "screening": 1, "behavior": 1,
                     "discovery": 1, "confirmation": 1}
            (config / "clevr_mcq4_dataset.json").write_text(json.dumps({
                "dataset_version": "test-v2", "split_sizes": sizes}))
            counts = {"calibration": 1, "screening": 1, "behavior": 1,
                      "discovery": 2, "confirmation": 2}
            (data / "image.png").write_bytes(b"image")
            (data / "items.jsonl").write_text("\n".join(json.dumps({"image": "image.png"})
                                                       for _ in range(7)) + "\n")
            (data / "variants.jsonl").write_text("{}\n")
            (data / "contrasts.jsonl").write_text("{}\n")
            (data / "validation_report.json").write_text("{}")
            hashes = {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                      for name in ("items.jsonl", "variants.jsonl", "contrasts.jsonl")}
            manifest = {"dataset_version": "test-v2", "counts": {"items_by_split": counts,
                        "variants_by_split": {"all": 12 * 7 + 4 * 4}}, "sha256": hashes,
                        "dataset_hash": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]}
            (data / "dataset_manifest.json").write_text(json.dumps(manifest))
            with patch.object(validate_screening, "ROOT", root):
                self.assertEqual(validate_screening.check_dataset(data), manifest["dataset_hash"])
                snapshot = data / "dataset_config.json"
                source_config = config / "clevr_mcq4_dataset.json"
                snapshot.write_text(source_config.read_text())
                self.assertEqual(validate_screening.check_dataset(data, source_config),
                                 manifest["dataset_hash"])
                snapshot.write_text(json.dumps({"dataset_version": "changed"}))
                with self.assertRaisesRegex(ValueError, "config snapshot"):
                    validate_screening.check_dataset(data, source_config)
                snapshot.write_text(source_config.read_text())
                (data / "image.png").unlink()
                with self.assertRaisesRegex(ValueError, "image missing"):
                    validate_screening.check_dataset(data)


if __name__ == "__main__":
    unittest.main()
