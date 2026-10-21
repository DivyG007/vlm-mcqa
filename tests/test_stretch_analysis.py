"""Authored only; not executed under the October 21 no-execution instruction."""
import json
from pathlib import Path
import tempfile
import unittest

from vlm_mcqa.stretch.analysis_cli import _write_new
from vlm_mcqa.stretch.evidence import exposure_inventory, synthesize_core7
from vlm_mcqa.stretch.failure_analysis import analyze_failures, unique_winner


def fixture():
    variant = {"variant_id": "v", "item_id": "i", "scene_id": "s", "counterfactual_pair_id": "p",
               "option_contents": ["red", "blue", "green", "yellow"], "correct_index": 0,
               "semantic_answer": "red", "labels": ["A", "B", "C", "D"], "correct_symbol": "A",
               "label_scheme": "letters", "split": "behavior"}
    behavior = {"variant_id": "v", "item_id": "i", "scheme": "letters", "position": 0,
                "condition": "clean", "label_logits": [0, 3, 1, 2], "correct": False,
                "predicted_index": 1, "global_top_is_label": True}
    lens = {"variant_id": "v", "item_id": "i", "scheme": "letters", "position": 0,
            "stage": 5, "content_logits": [4, 1, 2, 3], "correct": True}
    return {"variants": [variant], "contrasts": []}, behavior, lens


class FailureAnalysisTests(unittest.TestCase):
    def test_behavior_not_lens_defines_final_outcome(self):
        data, behavior, lens = fixture()
        result = analyze_failures(data, [behavior], [lens], stages=[5], expected_variant_ids=["v"])
        row = result["rows"][0]
        self.assertIs(row["final_correct"], False)
        self.assertIn("truth_aligned_content_with_wrong_final_symbol", row["descriptive_flags"])

    def test_clean_condition_only_and_tie_unresolved(self):
        data, behavior, lens = fixture()
        blank = dict(behavior, condition="blank", correct=True)
        behavior = dict(behavior, label_logits=[3, 3, 0, 0])
        result = analyze_failures(data, [blank, behavior], [lens], stages=[5], expected_variant_ids=["v"])
        self.assertIsNone(result["rows"][0]["final_correct"])
        self.assertIn("ambiguous_or_invalid_behavior_logits", result["rows"][0]["unresolved_reasons"])

    def test_duplicate_records_are_excluded_not_last_wins(self):
        data, behavior, lens = fixture()
        result = analyze_failures(data, [behavior, behavior], [lens, lens], stages=[5], expected_variant_ids=["v"])
        self.assertIsNone(result["rows"][0]["final_correct"])
        self.assertEqual(result["coverage"]["behavior"]["input_rows"], 2)
        self.assertEqual(result["coverage"]["missing_or_duplicate_clean_behavior"], ["v"])

    def test_scene_clustering_weights_scenes_equally(self):
        data, behavior, lens = fixture()
        second = dict(data["variants"][0], variant_id="v2")
        third = dict(data["variants"][0], variant_id="v3", scene_id="s2", counterfactual_pair_id=None)
        data["variants"] += [second, third]
        behaviors = [behavior, dict(behavior, variant_id="v2"),
                     dict(behavior, variant_id="v3", correct=True, predicted_index=0, label_logits=[4, 1, 2, 3])]
        lenses = [lens, dict(lens, variant_id="v2"), dict(lens, variant_id="v3")]
        result = analyze_failures(data, behaviors, lenses, stages=[5], expected_variant_ids=["v", "v2", "v3"])
        rate = result["summaries_by_prespecified_stage"]["5"]["scene_clustered"]["metrics"]["final_correct"]
        self.assertEqual(rate["equal_cluster_mean"], 0.5)
        self.assertEqual(rate["n_clusters"], 2)

    def test_missing_lens_and_invalid_metadata_remain_unresolved(self):
        data, behavior, lens = fixture()
        lens["position"] = 1
        result = analyze_failures(data, [behavior], [lens], stages=[5, 6], expected_variant_ids=["v"])
        self.assertIs(result["rows"][0]["final_correct"], False)
        self.assertIsNone(result["rows"][0]["content_proxy_correct"])
        self.assertEqual(result["coverage"]["missing_or_duplicate_lens_keys"], [["v", 6]])

    def test_missing_content_does_not_imply_grounding_failure(self):
        data, behavior, lens = fixture()
        lens.pop("content_logits")
        row = analyze_failures(data, [behavior], [lens], stages=[5], expected_variant_ids=["v"])["rows"][0]
        self.assertTrue(row["unresolved"])
        self.assertEqual(row["descriptive_flags"], [])

    def test_winners_reject_nonfinite_and_tied_values(self):
        self.assertIsNone(unique_winner([1, 1, 0, 0]))
        self.assertIsNone(unique_winner([float("nan"), 1, 0, 0]))
        self.assertIsNone(unique_winner([1, 0.99, 0, 0], tolerance=0.02))
        with self.assertRaises(ValueError):
            unique_winner([1, 0, 0, 0], tolerance=float("nan"))

    def test_split_and_symbol_mapping_mismatch_are_unresolved(self):
        data, behavior, lens = fixture()
        behavior["split"] = "confirmation"
        row = analyze_failures(data, [behavior], [lens], stages=[5], expected_variant_ids=["v"])["rows"][0]
        self.assertIsNone(row["final_correct"])
        self.assertIn("record_dataset_metadata_mismatch", row["unresolved_reasons"])
        data["variants"][0]["correct_symbol"] = "B"
        row = analyze_failures(data, [behavior], [lens], stages=[5], expected_variant_ids=["v"])["rows"][0]
        self.assertIn("missing_or_invalid_dataset_mapping", row["unresolved_reasons"])


class EvidenceTests(unittest.TestCase):
    def test_missing_corrected_qwen_does_not_fallback_to_old_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "qwen25vl_7b_core7" / "discovery"
            base.mkdir(parents=True)
            (base / "summary.json").write_text(json.dumps({"discovery": {"position_transfer_layer": 24},
                                                           "heads": {"verdict": "obsolete"}}))
            report = synthesize_core7(tmp)
            qwen = report["models"]["qwen25vl_7b_core7"]
            self.assertEqual(qwen["heads"]["status"], "missing_corrected_m3")
            self.assertEqual(qwen["phases"]["discovery"]["causal_aggregate"], {"position_transfer_layer": 24})

    def test_exposure_resolves_both_endpoints_without_untouched_claim(self):
        data, _, _ = fixture()
        data["variants"].append(dict(data["variants"][0], variant_id="other", counterfactual_pair_id="q"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            path.write_text(json.dumps({"donor_variant_id": "v", "recipient_variant_id": "other"}) + "\n")
            report = exposure_inventory(data, {"m1": path}, ["m1", "m2"])
            self.assertEqual(report["known_used_union"], ["p", "q"])
            self.assertFalse(report["declared_source_coverage_complete"])
            self.assertEqual(report["missing_required_sources"], ["m2"])
            self.assertEqual(report["historical_exposure_completeness"], "not_established_by_this_tool")

    def test_output_refuses_existing_or_input_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaises(ValueError):
                _write_new(path / "inside", {}, [path])
            with self.assertRaises(FileExistsError):
                _write_new(path, {}, [])
            raw = path / "raw.jsonl"
            raw.write_text("{}\n")
            with self.assertRaises(ValueError):
                _write_new(path / "new_analysis", {}, [raw])


if __name__ == "__main__":
    unittest.main()
