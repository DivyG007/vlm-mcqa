"""Future static preservation checks. Authored 2026-10-21; NOT EXECUTED.

These checks import no project/model code and require no compute server.
They detect changes relative to the local byte baseline, not runtime fitness.
Run only after execution is separately authorized. A failure requires review,
not blindly regenerating the baseline or reverting pre-existing user changes.
"""

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/ai_handoff/stretch_preservation_baseline_2026-10-21.json"


class StretchPreservationTests(unittest.TestCase):
    def test_original_core7_files_preserved(self):
        baseline = json.loads(BASELINE.read_text(encoding="utf-8-sig"))
        self.assertEqual(baseline["schema_version"], 1)
        self.assertIsNone(baseline["compute_server"])
        self.assertTrue(baseline["files"])
        for record in baseline["files"]:
            with self.subTest(path=record["path"]):
                relative = Path(record["path"])
                self.assertFalse(relative.is_absolute())
                self.assertNotIn("..", relative.parts)
                path = ROOT / relative
                self.assertTrue(path.is_file(), "preserved file removed")
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(actual, record["sha256"], "review baseline change")

    def test_paligemma2_configuration_preserved(self):
        models = json.loads((ROOT / "configs/core7/models.json").read_text())
        spec = models["models"]["paligemma2_10b_mix"]
        self.assertEqual(spec["hf_id"], "google/paligemma2-10b-mix-448")
        self.assertEqual(spec["family"], "paligemma")
        self.assertEqual(spec["lm_backbone"], "gemma2")
        self.assertEqual(spec["attn_implementation"], "eager")
        self.assertEqual(spec["prompt_prefix"], "answer en ")

    def test_frozen_selection_thresholds_preserved(self):
        profiles = json.loads((ROOT / "configs/core7/profiles.json").read_text())
        self.assertEqual(profiles["profiles"]["full_core7"]["pairs"], 64)
        selection = profiles["selection"]
        self.assertEqual(selection["min_worst_position_accuracy"], 0.90)
        self.assertEqual(selection["min_shuffled_gap"], 0.30)
        self.assertEqual(selection["min_usable_discovery_pairs"], 64)
        self.assertEqual(selection["min_usable_confirmation_pairs"], 48)


if __name__ == "__main__":
    unittest.main()
