"""Pair-clustered summary statistics (core7.scoring)."""

from __future__ import annotations

import unittest

from support import HAS_TORCH, requires_torch

if HAS_TORCH:
    from vlm_mcqa.core7.scoring import summarize_clustered


@requires_torch
class ClusteredSummaryTests(unittest.TestCase):
    def test_semantic_pairs_are_weighted_equally(self) -> None:
        # The heavier pair has three directed rows, but it must not receive
        # three times the statistical weight of the other semantic pair.
        cases = {
            "default_seed": ([0.0, 0.0, 0.0, 1.0], ["pair_a", "pair_a", "pair_a", "pair_b"], {}),
            "explicit_seed": ([1.0, 1.0, 1.0, 0.0], ["p1", "p1", "p1", "p2"], {"seed": 7}),
        }
        for name, (values, clusters, kwargs) in cases.items():
            with self.subTest(name):
                summary = summarize_clustered(values, clusters, **kwargs)
                self.assertEqual(summary["mean"], 0.5)
                self.assertEqual(summary["n"], 4)
                self.assertEqual(summary["n_clusters"], 2)
                self.assertEqual(summary["sampling_unit"], "pair")


if __name__ == "__main__":
    unittest.main()
