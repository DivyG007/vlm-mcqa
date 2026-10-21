"""Authored reliability/stratum tests; no execution during source work."""
import copy
import unittest

from vlm_mcqa.stretch.failure_protocol import evaluate_failure_rules, freeze_failure_rules


def rules():
    return {"stage": 12, "strata_flags": ["truth_aligned_content_with_wrong_final_symbol"],
            "min_scenes": 2, "reliability_min": .75, "seed": 13,
            "bootstrap_draws": 100, "margin_bins": [0, 1, 3],
            "match_fields": ["question_family", "scheme"], "max_control_reuse": 1}


class FailureProtocolTests(unittest.TestCase):
    def test_modified_rules_rejected(self):
        frozen = freeze_failure_rules(rules())
        frozen["stage"] = 99
        with self.assertRaises(ValueError):
            evaluate_failure_rules({"status": "descriptive_not_causal", "rows": []},
                                   {"items": [], "variants": []}, frozen)

    def test_insufficient_reliability_never_nominates_causal_strata(self):
        report = {"status": "descriptive_not_causal", "rows": []}
        data = {"items": [], "variants": []}
        for i in range(20):
            vid, scene, item = f"v{i}", f"s{i}", f"i{i}"
            data["items"].append({"item_id": item, "question_family": "color", "reasoning_depth": 1})
            data["variants"].append({"variant_id": vid, "item_id": item})
            report["rows"].append({"variant_id": vid, "stage": 12, "scene_id": scene,
                                   "scheme": "letters", "final_correct": False,
                                   "content_proxy_correct": True, "confidence_gap": .5,
                                   "unresolved": False,
                                   "descriptive_flags": ["truth_aligned_content_with_wrong_final_symbol"]})
        result = evaluate_failure_rules(report, data, freeze_failure_rules(rules()))
        for phase in result["phases"].values():
            block = phase["strata"]["truth_aligned_content_with_wrong_final_symbol"]
            self.assertEqual(block["status"], "descriptive_only")
            self.assertFalse(block["reliability_gate_passed"])
            self.assertEqual(block["matched_correct_controls"], [])

    def test_duplicate_rows_not_silently_weighted(self):
        row = {"variant_id": "v", "stage": 12, "scene_id": "s", "final_correct": True}
        report = {"status": "descriptive_not_causal", "rows": [row, copy.deepcopy(row)]}
        with self.assertRaises(ValueError):
            evaluate_failure_rules(report, {"items": [], "variants": []}, freeze_failure_rules(rules()))

    def test_too_many_strata_not_searched(self):
        config = rules()
        config["strata_flags"] *= 3
        with self.assertRaises(ValueError):
            freeze_failure_rules(config)


if __name__ == "__main__":
    unittest.main()
