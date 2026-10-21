"""Authored sampling specifications. Deliberately unexecuted October 21."""
import hashlib
import json
import unittest

from vlm_mcqa.stretch.data import CONTRAST_TYPE
from vlm_mcqa.stretch.sampling import select_fixed_text_sample


IDENTITY = {"model_key": "explicit-model", "dataset_hash": "explicit-data",
            "prediction_source_sha256": "a" * 64}


def fixture(pairs=3, layouts=2):
    variants, contrasts, baselines = {}, [], []
    for p in range(pairs):
        pair = f"pair-{p}"
        for layout in range(layouts):
            prompt = f"question {p}; exact layout {layout}"
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            ids = [f"p{p}-l{layout}-i{side}" for side in (0, 1)]
            for side, vid in enumerate(ids):
                variant = {"variant_id": vid, "item_id": f"item-{p}-{side}",
                           "scene_id": f"scene-{p}-{side}", "image": f"image-{p}-{side}.png",
                           "counterfactual_pair_id": pair, "split": "discovery", "prompt": prompt,
                           "question": "question", "labels": list("ABCD"),
                           "label_scheme": "letters", "option_contents": ["red", "blue", "green", "gray"],
                           "correct_index": side, "semantic_answer": ["red", "blue"][side],
                           "correct_symbol": "AB"[side]}
                variants[vid] = variant
                logits = [0, 0, 0, 0]
                logits[side] = 2
                baselines.append({"variant_id": vid, "item_id": variant["item_id"],
                                  "position": side, "scheme": "letters", "condition": "clean",
                                  "correct": True, "predicted_index": side, "label_logits": logits})
            for a, b in (ids, ids[::-1]):
                donor, recipient = variants[a], variants[b]
                contrasts.append({"contrast_id": a + "->" + b, "contrast_type": CONTRAST_TYPE,
                                  "pair_id": pair, "split": "discovery", "donor_variant_id": a,
                                  "recipient_variant_id": b, "donor_symbol": donor["correct_symbol"],
                                  "recipient_symbol": recipient["correct_symbol"],
                                  "donor_content": donor["semantic_answer"],
                                  "recipient_content": recipient["semantic_answer"],
                                  "prompt_sha256": digest, "layout_id": digest,
                                  "donor_scene_id": donor["scene_id"], "recipient_scene_id": recipient["scene_id"]})
    return contrasts, variants, baselines


def select(data, **kwargs):
    args = {"split": "discovery", "target_pairs": 2, "layouts_per_pair": 2,
            "min_pairs": 1, "seed": 19, "excluded_pair_ids": [], "baseline_identity": IDENTITY}
    args.update(kwargs)
    return select_fixed_text_sample(*data, **args)


class SamplingTests(unittest.TestCase):
    def test_ready_bidirectional_exact_layouts_cap_and_hash(self):
        data = fixture()
        plan = select(data)
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(len(plan["contrast_ids"]), 8)
        self.assertEqual(len(plan["pair_ids"]), 2)
        self.assertTrue(all(len(x) == 2 for x in plan["selected_layouts"].values()))
        body = {k: v for k, v in plan.items() if k != "plan_sha256"}
        expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False).encode()).hexdigest()
        self.assertEqual(plan["plan_sha256"], expected)
        shuffled = list(reversed(data[0])), dict(reversed(list(data[1].items()))), list(reversed(data[2]))
        self.assertEqual(plan, select(shuffled))

    def test_orders_precede_outcomes(self):
        data = fixture()
        before = select(data)
        changed = [{**r, "correct": False, "predicted_index": 3, "label_logits": [0, 0, 0, 3]}
                   for r in data[2]]
        after = select((data[0], data[1], changed))
        self.assertEqual(before["candidate_pair_order"], after["candidate_pair_order"])
        self.assertEqual(before["candidate_layout_order"], after["candidate_layout_order"])
        self.assertEqual(after["status"], "insufficient")
        self.assertEqual(after["coverage"]["endpoint_states"]["incorrect"], 12)

    def test_missing_duplicate_unknown_incorrect_and_metadata(self):
        data = fixture(pairs=1, layouts=2)
        baseline = data[2]
        for changed, reason in ((baseline[1:], "missing"),
                                (baseline + [baseline[0]], "duplicate"),
                                ([{**baseline[0], "label_logits": [0, 0, 0, 0]}, *baseline[1:]], "unknown"),
                                ([{**baseline[0], "scheme": "numbers"}, *baseline[1:]], "metadata_mismatch")):
            plan = select((data[0], data[1], changed), target_pairs=1)
            self.assertEqual(plan["status"], "insufficient")
            self.assertEqual(plan["coverage"]["endpoint_states"][reason], 1)
            self.assertEqual(plan["coverage"]["layout_shortfall_pairs"], 1)

    def test_shortfall_is_not_ready_and_exclusions_do_not_attest_exposure(self):
        plan = select(fixture(pairs=1), target_pairs=2)
        self.assertEqual(plan["status"], "target_shortfall")
        self.assertFalse(plan["confirmation_claim"])
        excluded = select(fixture(), excluded_pair_ids=["pair-0", "pair-1", "pair-2"])
        self.assertEqual(excluded["status"], "insufficient")
        self.assertIn("caller_audit_required", excluded["historical_exposure"])

    def test_malformed_direction_and_identity_fail(self):
        data = fixture()
        with self.assertRaises(ValueError):
            select((data[0][1:], data[1], data[2]))
        with self.assertRaises(ValueError):
            select(data, baseline_identity={**IDENTITY, "prediction_source_sha256": "bad"})
        variants = {k: {**v, "dataset_hash": "different"} for k, v in data[1].items()}
        with self.assertRaises(ValueError):
            select((data[0], variants, data[2]))

    def test_scene_and_pair_cross_split_rejected(self):
        data = fixture()
        variant = next(iter(data[1].values()))
        extra = {**variant, "variant_id": "other", "split": "confirmation"}
        with self.assertRaises(ValueError):
            select((data[0], {**data[1], "other": extra}, data[2]))

    def test_full_release_auxiliary_splits_are_accepted(self):
        data = fixture()
        first = next(iter(data[1].values()))
        variants = dict(data[1])
        for split in ("behavior", "calibration", "screening"):
            vid = "unrelated-" + split
            variants[vid] = {**first, "variant_id": vid, "split": split,
                             "scene_id": "scene-" + split,
                             "counterfactual_pair_id": "pair-" + split}
        self.assertEqual(select((data[0], variants, data[2]))["status"], "ready")

    def test_no_implicit_count_thresholds(self):
        for kwargs in ({"target_pairs": True}, {"layouts_per_pair": 1.0}, {"min_pairs": 3}):
            with self.assertRaises(ValueError):
                select(fixture(), **kwargs)


if __name__ == "__main__":
    unittest.main()
