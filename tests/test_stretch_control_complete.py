import hashlib
from types import SimpleNamespace
import unittest

from vlm_mcqa.stretch.data import CONTRAST_TYPE
from vlm_mcqa.stretch.controls import (bind_calibration_contrast, coverage_is_adequate,
                                       select_control_complete_sample, validate_frozen_controls)
from vlm_mcqa.stretch.campaign import validate_sample_plan


IDENTITY = {"model_key": "fixture", "dataset_hash": "data", "prediction_source_sha256": "a" * 64}
POLICY = {"seed": 0, "per_recipient": 1, "max_donor_reuse": 1,
          "match_family": True, "margin_bin_edges": [0.0, 1.0, 2.0, 4.0, 8.0]}


def fixture(drop_position=False):
    prompt = "What color?\nA. red\nB. blue\nC. green\nD. yellow\nAnswer:"
    layout = hashlib.sha256(prompt.encode()).hexdigest()
    base = {"split": "discovery", "labels": list("ABCD"), "label_scheme": "letters",
            "variant_kind": "position", "question_family": "colour", "prompt": prompt,
            "question": "What color?", "option_contents": ["red", "blue", "green", "yellow"]}
    left = {**base, "variant_id": "left", "item_id": "item-left", "scene_id": "scene-left",
            "image": "left.png", "counterfactual_pair_id": "pair-main", "correct_index": 0,
            "correct_symbol": "A", "semantic_answer": "red"}
    right = {**base, "variant_id": "right", "item_id": "item-right", "scene_id": "scene-right",
             "image": "right.png", "counterfactual_pair_id": "pair-main", "correct_index": 1,
             "correct_symbol": "B", "semantic_answer": "blue"}
    variants = {row["variant_id"]: row for row in (left, right)}
    for recipient, symbol, index in ((left, "B", 1), (right, "A", 0)):
        vid = "position-" + recipient["variant_id"]
        if drop_position and recipient is right:
            continue
        variants[vid] = {**recipient, "variant_id": vid, "correct_symbol": symbol,
                         "correct_index": index, "semantic_answer": recipient["option_contents"][index]}
    for index, (symbol, answer) in enumerate((("A", "red"), ("B", "blue"))):
        vid = f"random-{index}"
        variants[vid] = {**base, "variant_id": vid, "item_id": f"random-item-{index}",
                         "scene_id": f"random-scene-{index}", "image": f"random-{index}.png",
                         "counterfactual_pair_id": f"random-pair-{index}", "correct_index": index,
                         "correct_symbol": symbol, "semantic_answer": answer}
    contrasts = []
    for donor, recipient in ((left, right), (right, left)):
        contrasts.append({"contrast_id": donor["variant_id"] + "->" + recipient["variant_id"],
            "contrast_type": CONTRAST_TYPE, "pair_id": "pair-main", "split": "discovery",
            "donor_variant_id": donor["variant_id"], "recipient_variant_id": recipient["variant_id"],
            "donor_symbol": donor["correct_symbol"], "recipient_symbol": recipient["correct_symbol"],
            "donor_content": donor["semantic_answer"], "recipient_content": recipient["semantic_answer"],
            "prompt_sha256": layout, "layout_id": layout,
            "donor_scene_id": donor["scene_id"], "recipient_scene_id": recipient["scene_id"]})
    baselines = []
    for row in (left, right):
        logits = [0.0] * 4
        logits[row["correct_index"]] = 3.0
        baselines.append({"condition": "clean", "variant_id": row["variant_id"],
            "item_id": row["item_id"], "position": row["correct_index"], "scheme": "letters",
            "split": "discovery", "scene_id": row["scene_id"], "labels": list("ABCD"),
            "correct_symbol": row["correct_symbol"], "model_key": "fixture", "dataset_hash": "data",
            "correct": True, "predicted_index": row["correct_index"], "label_logits": logits})
    clean = {vid: SimpleNamespace(correct=True, readout={"margin": 3.0}) for vid in variants}
    session = SimpleNamespace(variants=variants, clean=lambda vid: clean[vid])
    return session, contrasts, baselines


class ControlCompleteTests(unittest.TestCase):
    def test_calibration_rebinds_to_first_model_eligible_direction(self):
        config = {"calibration_contrast_id": "qwen-only-direction", "other": 1}
        prepared = bind_calibration_contrast(config, {"contrast_ids": ["eligible-a", "eligible-b"]})
        self.assertEqual(prepared["calibration_contrast_id"], "eligible-a")
        self.assertEqual(prepared["requested_calibration_contrast_id"], "qwen-only-direction")
        self.assertEqual(config["calibration_contrast_id"], "qwen-only-direction")

    def test_calibration_rebind_requires_eligible_direction(self):
        with self.assertRaisesRegex(ValueError, "eligible calibration direction"):
            bind_calibration_contrast({"calibration_contrast_id": "old"}, {"contrast_ids": []})

    def test_pilot_uses_phase_minimum_not_full_science_minimum(self):
        coverage = {"unmatched_control_groups": 0, "semantic_pairs": 8,
                    "semantic_evaluated": 32, "controls_requested": 64,
                    "controls_evaluated": 64}
        self.assertTrue(coverage_is_adequate(coverage, selected_count=32, min_pairs=8))
        self.assertFalse(coverage_is_adequate(coverage, selected_count=32, min_pairs=16))

    def test_controls_are_selected_and_hash_bound_before_execution(self):
        session, contrasts, baselines = fixture()
        plan = select_control_complete_sample(session, contrasts, baselines, split="discovery",
            target_pairs=1, min_pairs=1, layouts_per_pair=1, seed=0, excluded_pair_ids=[],
            baseline_identity=IDENTITY, control_policy=POLICY)
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["coverage"]["selected_controls"], 4)
        selected = validate_sample_plan(plan, contrasts, {"model_key": "fixture", "dataset_hash": "data"})
        self.assertEqual(len(validate_frozen_controls(plan, selected, POLICY)), 4)
        plan["matched_controls"][0]["donor_variant_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "immutable control-complete"):
            validate_frozen_controls(plan, selected, POLICY)

    def test_missing_direction_control_excludes_whole_pair_before_run(self):
        session, contrasts, baselines = fixture(drop_position=True)
        plan = select_control_complete_sample(session, contrasts, baselines, split="discovery",
            target_pairs=1, min_pairs=1, layouts_per_pair=1, seed=0, excluded_pair_ids=[],
            baseline_identity=IDENTITY, control_policy=POLICY)
        self.assertEqual(plan["status"], "insufficient")
        self.assertEqual(plan["pair_ids"], [])
        self.assertEqual(plan["coverage"]["control_incomplete_pairs"], 1)
        reasons = plan["exclusions"]["control_reasons"]["pair-main"]
        self.assertTrue(any(row["control_type"] == "position" for row in reasons))


if __name__ == "__main__":
    unittest.main()
