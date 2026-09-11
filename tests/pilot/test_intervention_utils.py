"""Pilot patching utilities (vlm_mcqa.interventions.core)."""

from __future__ import annotations

import unittest

from vlm_mcqa.interventions.core import (
    bbox_to_grid_indices,
    complement_indices,
    find_subsequence,
    normalized_recovery,
)


class InterventionUtilityTests(unittest.TestCase):
    def test_find_subsequence_returns_all_matches(self) -> None:
        self.assertEqual(find_subsequence([1, 2, 1, 2, 3], [1, 2]), [0, 2])
        self.assertEqual(find_subsequence([1, 2], []), [])

    def test_bbox_maps_to_grid_cells_and_complement(self) -> None:
        selected = bbox_to_grid_indices(
            bbox=[0, 0, 50, 50],
            image_size=100,
            grid_height=2,
            grid_width=2,
            image_token_indices=[10, 11, 12, 13],
        )
        self.assertEqual(selected, [10])
        self.assertEqual(complement_indices(selected, [10, 11, 12, 13]), [11, 12, 13])

    def test_normalized_recovery_is_unclipped(self) -> None:
        self.assertEqual(normalized_recovery(-4.0, 4.0, 0.0), 0.5)
        self.assertEqual(normalized_recovery(-4.0, 4.0, 8.0), 1.5)


if __name__ == "__main__":
    unittest.main()
