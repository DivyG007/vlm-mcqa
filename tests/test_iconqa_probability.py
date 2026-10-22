"""Focused independent failure cases for natural-pair admission and factorials."""
import copy
import random
import unittest

from vlm_mcqa.clevr_mcq4.iconqa_probability import CHOICES, select_pairs
from vlm_mcqa.clevr_mcq4.variants import build_variants, build_pair_contrasts, validate


def fixture():
    inventory, pools = {}, {s: [] for s in ("train", "val", "test")}
    for split, start in (("train", 0), ("val", 2), ("test", 4)):
        for n in (start, start + 1):
            pid = str(n)
            inventory[pid] = {"pid": pid, "pixel_sha256": f"pixels{n}", "errors": [],
                "data": {"split": split, "ques_type": "choose_txt", "choices": CHOICES.copy(),
                         "answer": n % 2, "question": "marbles?"}}
        pools[split].append({"x": str(start), "y": str(start+1), "options_x": CHOICES,
                             "answer_x": CHOICES[0], "answer_y": CHOICES[1]})
    return inventory, pools


class ProbabilityTests(unittest.TestCase):
    def test_native_splits_retained(self):
        rows, pools = fixture()
        selected = select_pairs(pools, rows, seed=7, train_pairs=1)
        self.assertEqual([s for s, *_ in selected], ["train", "val", "test"])

    def test_cross_task_cross_split_pixel_reuse_rejected(self):
        rows, pools = fixture()
        rows["unrelated"] = {"pid": "unrelated", "pixel_sha256": "pixels2",
                             "data": {"split": "test", "ques_type": "choose_img"}}
        with self.assertRaisesRegex(ValueError, "reused across original splits"):
            select_pairs(pools, rows, seed=7, train_pairs=1)

    def test_same_image_in_two_pairs_rejected(self):
        rows, pools = fixture()
        pools["val"].append(copy.deepcopy(pools["val"][0]))
        with self.assertRaisesRegex(ValueError, "reused image/PID"):
            select_pairs(pools, rows, seed=7, train_pairs=1)

    def test_source_truth_not_candidate_claim_is_checked(self):
        rows, pools = fixture()
        rows["3"]["data"]["answer"] = 0
        with self.assertRaisesRegex(ValueError, "different correct answers"):
            select_pairs(pools, rows, seed=7, train_pairs=1)

    def test_different_question_rejected(self):
        rows, pools = fixture()
        rows["3"]["data"]["question"] = "different?"
        with self.assertRaisesRegex(ValueError, "question mismatch"):
            select_pairs(pools, rows, seed=7, train_pairs=1)

    def test_train_underfill_not_silent(self):
        rows, pools = fixture()
        with self.assertRaisesRegex(ValueError, "insufficient"):
            select_pairs(pools, rows, seed=7, train_pairs=2)

    def test_all_answer_transitions_have_valid_factorials(self):
        for a in CHOICES:
            for b in CHOICES:
                if a == b:
                    continue
                base = {"split": "discovery", "counterfactual_pair_id": "pair", "question": "marbles?",
                        "question_family": "grade1_S2", "program_depth": None, "option_contents": CHOICES}
                x = dict(base, item_id="x", scene_id="px", image="x.png", semantic_answer=a)
                y = dict(base, item_id="y", scene_id="py", image="y.png", semantic_answer=b)
                xv, yv = [build_variants(i, permuted_control=True) for i in (x, y)]
                contrasts = build_pair_contrasts(xv, yv, random.Random(7))
                result = validate([x, y], xv + yv, contrasts)
                self.assertEqual(result["contrasts_by_type"]["content_and_position"], 8)


if __name__ == "__main__":
    unittest.main()
