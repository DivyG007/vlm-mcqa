"""Pair-balanced case selection for capped methods (core7.sampling, core7.contrasts)."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from support import HAS_TORCH, requires_torch
from vlm_mcqa.core7.sampling import round_robin_across_groups

if HAS_TORCH:
    import vlm_mcqa.core7.contrasts as contrasts


@dataclass
class FakeCase:
    contrast: dict
    recipient: dict
    donor: dict | None = None

    def __post_init__(self) -> None:
        if self.donor is None:
            self.donor = {}

    @property
    def kind(self) -> str:
        return self.contrast["contrast_type"]


class FakeSession:
    def __init__(self) -> None:
        self.contrasts = []
        self.profile = {"require_correct": False}


class RoundRobinSamplingTests(unittest.TestCase):
    def test_cap_covers_every_group_before_repeating(self) -> None:
        buckets = {f"pair_{i}": [f"{i}:{j}" for j in range(4)] for i in range(64)}
        selected = round_robin_across_groups(buckets, 96)
        self.assertEqual(len(selected), 96)
        self.assertEqual(len({value.split(":")[0] for value in selected}), 64)
        keys = sorted(buckets)
        self.assertEqual(selected[:64], [buckets[key][i % 4] for i, key in enumerate(keys)])
        self.assertEqual(selected[64:], [buckets[key][(i + 1) % 4]
                                         for i, key in enumerate(keys[:32])])

    def test_regression_flat_slice_undercovers_groups(self) -> None:
        buckets = {f"pair_{i}": [f"{i}:{j}" for j in range(16)] for i in range(64)}
        old = [value for key in sorted(buckets) for value in buckets[key]][:96]
        corrected = round_robin_across_groups(buckets, 96)
        self.assertLess(len({value.split(":")[0] for value in old}), 10)
        self.assertEqual(len({value.split(":")[0] for value in corrected}), 64)

    def test_exhaustion_and_zero_limit(self) -> None:
        self.assertEqual(round_robin_across_groups({"a": [1], "b": [2]}, 0), [])
        self.assertEqual(round_robin_across_groups({"a": [1], "b": [2]}, 10), [1, 2])
        with self.assertRaises(ValueError):
            round_robin_across_groups({"a": [1]}, -1)


@requires_torch
class CaseSelectionTests(unittest.TestCase):
    def test_select_cases_balances_pairs_through_filter_path(self) -> None:
        session = FakeSession()
        cases = {}
        schemes = ("letters", "numbers", "rare_letters")
        for pair_index in range(64):
            pair_id = f"pair_{pair_index:03d}"
            bucket = []
            for case_index in range(48):
                contrast = {"pair_id": pair_id, "contrast_type": "position",
                            "split": "confirmation", "case": case_index}
                session.contrasts.append(contrast)
                bucket.append(FakeCase(contrast, {"label_scheme": schemes[case_index // 16]}))
            cases[pair_id] = bucket

        original = contrasts._make_case
        contrasts._make_case = lambda _session, contrast: next(
            c for c in cases[contrast["pair_id"]] if c.contrast is contrast)
        try:
            selected, info = contrasts.select_cases(session, "confirmation", ("position",),
                                                    max_pairs=64, limit_per_type=96)
        finally:
            contrasts._make_case = original

        self.assertEqual(len(selected), 96)
        self.assertEqual(info["eligible_pairs"], 64)
        self.assertEqual(info["pairs_used"], 64)
        self.assertEqual(set(info["contrasts_per_pair"].values()), {1, 2})
        self.assertEqual(sum(v == 2 for v in info["contrasts_per_pair"].values()), 32)
        self.assertEqual(set(info["cases_by_scheme"]), set(schemes))


if __name__ == "__main__":
    unittest.main()
