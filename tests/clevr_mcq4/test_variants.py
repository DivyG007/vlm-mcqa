"""Prompt variants and directed pair contrasts (clevr_mcq4.variants)."""

from __future__ import annotations

import random
import unittest
from collections import Counter

from vlm_mcqa.clevr_mcq4.variants import ALPHABETS, build_pair_contrasts, build_variants, rotate_to


def _item(member: str, answer: str) -> dict:
    return {"item_id": f"it_{member}", "split": "discovery", "scene_id": "s0",
            "counterfactual_pair_id": "pair_s0", "pair_member": member,
            "image": f"images/{member}.png", "question": "What color?",
            "question_family": "count_compare", "program_depth": 3,
            "option_contents": ["red", "blue", "green", "gray"], "semantic_answer": answer}


class VariantTests(unittest.TestCase):
    def test_rotation_places_answer_at_every_position(self) -> None:
        options = ["a", "b", "c", "d"]
        for p in range(4):
            self.assertEqual(rotate_to(options, "c", p)[p], "c")

    def test_variants_and_contrasts_are_consistent(self) -> None:
        xv = build_variants(_item("x", "red"), permuted_control=True)
        yv = build_variants(_item("y", "blue"), permuted_control=True)
        self.assertEqual(len(xv), 12 + 4)
        for v in xv:
            self.assertEqual(v["labels"], list(ALPHABETS[v["label_scheme"]]))
            self.assertEqual(v["option_contents"][v["correct_index"]], "red")
        perm = [v for v in xv if v["variant_kind"] == "distractor_permuted"]
        base = {v["correct_index"]: v for v in xv if v["variant_kind"] == "position" and v["label_scheme"] == "letters"}
        for v in perm:
            self.assertNotEqual(v["option_contents"], base[v["correct_index"]]["option_contents"])
        contrasts = build_pair_contrasts(xv, yv, random.Random(1))
        kinds = Counter(c["contrast_type"] for c in contrasts)
        self.assertEqual(kinds["position"], 16)
        self.assertEqual(kinds["content"], 8)
        self.assertEqual(kinds["content_and_position"], 8)
        for c in contrasts:
            if c["contrast_type"] == "content_and_position":
                self.assertEqual(len({c["donor_symbol"], c["recipient_symbol"], c["content_symbol"]}), 3)


if __name__ == "__main__":
    unittest.main()
