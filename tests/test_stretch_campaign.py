"""Authored static implementation checks; deliberately NOT EXECUTED."""
import json
from pathlib import Path
import tempfile
import unittest

from vlm_mcqa.stretch.campaign import run_semantic_campaign, validate_sample_plan
from vlm_mcqa.stretch.data import digest


def fixture(status="ready"):
    contrasts = [{"contrast_id": "x-y", "pair_id": "p", "split": "discovery", "layout_id": "layout",
                  "donor_variant_id": "x", "recipient_variant_id": "y"},
                 {"contrast_id": "y-x", "pair_id": "p", "split": "discovery", "layout_id": "layout",
                  "donor_variant_id": "y", "recipient_variant_id": "x"}]
    plan = {"schema": "stretch.fixed-text-sample.v1", "status": status, "split": "discovery", "pair_ids": ["p"],
            "contrast_ids": ["x-y", "y-x"],
            "target_pairs": 1, "min_pairs": 1, "layouts_per_pair": 1,
            "control_complete": True,
            "selected_layouts": {"p": ["layout"]},
            "baseline_identity": {"model_key": "fixture", "dataset_hash": "data"}}
    plan["plan_sha256"] = digest(json.dumps(plan, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode())
    return plan, contrasts, {"model_key": "fixture", "dataset_hash": "data"}


class CampaignTests(unittest.TestCase):
    def test_no_execution_denied_before_session_or_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new"
            with self.assertRaises(PermissionError):
                run_semantic_campaign(
                    object(), [], {}, {}, calibration={}, identity={}, output_dir=output,
                    protected_inputs=[], bootstrap_seed=1, bootstrap_samples=100, execute=False)
            self.assertFalse(output.exists())

    def test_plan_hash_and_scientific_scope_required(self):
        plan, contrasts, recipe = fixture()
        self.assertEqual(validate_sample_plan(plan, contrasts, recipe), contrasts)
        plan["pair_ids"] = ["another-pair"]
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_sample_plan(plan, contrasts, recipe)

    def test_target_shortfall_not_promoted(self):
        plan, contrasts, recipe = fixture("target_shortfall")
        with self.assertRaisesRegex(ValueError, "no automatic shortfall"):
            validate_sample_plan(plan, contrasts, recipe)

    def test_missing_reverse_and_wrong_baseline_rejected(self):
        plan, contrasts, recipe = fixture()
        with self.assertRaises(ValueError):
            validate_sample_plan(plan, contrasts[:1], recipe)
        with self.assertRaises(ValueError):
            validate_sample_plan(plan, contrasts, {"model_key": "other", "dataset_hash": "data"})


if __name__ == "__main__":
    unittest.main()
