"""Scene normalisation, question families and counterfactual edits (clevr_mcq4.scene, .questions)."""

from __future__ import annotations

import random
import unittest
from collections import Counter

from vlm_mcqa.clevr_mcq4 import questions as Q
from vlm_mcqa.clevr_mcq4.build import find_counterfactual
from vlm_mcqa.clevr_mcq4.scene import apply_edit, minimal_unique_spec, normalize_scene

from .fixtures import random_scene


class SceneAndQuestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(0)
        self.scenes = [normalize_scene(random_scene(i, self.rng)) for i in range(40)]

    def test_relationships_are_antisymmetric(self) -> None:
        for scene in self.scenes:
            rel = scene["relationships"]
            for i, lefts in enumerate(rel["left"]):
                for j in lefts:
                    self.assertIn(i, rel["right"][j])

    def test_minimal_unique_spec_describes_its_object(self) -> None:
        for scene in self.scenes:
            for i in range(len(scene["objects"])):
                spec = minimal_unique_spec(scene, i)
                if spec is not None:
                    self.assertTrue(all(scene["objects"][i][a] == v for a, v in spec.items()))

    def test_every_family_yields_valid_items_with_options(self) -> None:
        seen = Counter()
        for scene in self.scenes:
            for family, propose in Q.PROPOSERS.items():
                for params, _text in propose(scene, self.rng)[:5]:
                    answer = Q.evaluate(family, scene, params)
                    if answer is None:
                        continue
                    options = Q.build_options(Q.FAMILY_ANSWER_TYPE[family], answer, scene, self.rng)
                    if options is None:
                        continue
                    self.assertIn(answer, options)
                    self.assertEqual(len(set(options)), 4)
                    self.assertGreaterEqual(Q.program_depth(Q.program(family, params)), 2)
                    seen[family] += 1
        self.assertEqual(set(seen), set(Q.PROPOSERS), seen)

    def test_counterfactual_changes_answer_within_options(self) -> None:
        found = 0
        for scene in self.scenes:
            for family in ("query_color_relate", "count", "query_relation"):
                for params, text in Q.PROPOSERS[family](scene, self.rng)[:3]:
                    answer = Q.evaluate(family, scene, params)
                    if answer is None:
                        continue
                    options = Q.build_options(Q.FAMILY_ANSWER_TYPE[family], answer, scene, self.rng)
                    if options is None:
                        continue
                    cand = {"question_family": family, "params": params,
                            "semantic_answer": answer, "option_contents": options}
                    cf = find_counterfactual(scene, cand, self.rng)
                    if cf is None:
                        continue
                    found += 1
                    edited = apply_edit(scene, cf["edit"])
                    self.assertEqual(Q.evaluate(family, edited, params), cf["semantic_answer"])
                    self.assertNotEqual(cf["semantic_answer"], answer)
                    self.assertIn(cf["semantic_answer"], options)
                    moved = [a["3d_coords"] != b["3d_coords"] for a, b in zip(scene["objects"], edited["objects"])]
                    self.assertFalse(any(moved))
        self.assertGreater(found, 10)


if __name__ == "__main__":
    unittest.main()
