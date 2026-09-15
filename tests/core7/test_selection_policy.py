"""Model-selection policy: letters readiness first, proposal robustness on ties (core7.report.select)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from support import HAS_TORCH, requires_torch

if HAS_TORCH:
    from vlm_mcqa.core7.report import select

THRESHOLDS = {
    "min_worst_position_accuracy": 0.90,
    "min_proposal_robustness_accuracy": 0.45,
    "min_shuffled_gap": 0.30,
    "min_usable_discovery_pairs": 64,
    "min_usable_confirmation_pairs": 48,
    "max_params_b": 10.0,
    "comparable_accuracy_window": 0.03,
}


def screen(root: Path, name: str, family: str, *, worst: float, robust: float,
           params: float = 3.0, backbone: str = "qwen2") -> Path:
    path = root / name
    path.mkdir()
    (path / "run.json").write_text(json.dumps({
        "model_key": name, "family": family, "params_b": params, "lm_backbone": backbone,
    }))
    (path / "behavior_screening_summary.json").write_text(json.dumps({
        "worst_position_accuracy": worst, "letters_accuracy": worst,
        "clean_minus_shuffled": 0.5, "worst_scheme_position_accuracy": robust,
        "pair_yield": {"discovery": {"usable_pairs": 70},
                       "confirmation": {"usable_pairs": 55}},
    }))
    (path / "calibration.json").write_text(json.dumps({
        "all_passed": True, "gates": {"identity": {"value": 0.0, "limit": 0.25}},
    }))
    return path


@requires_torch
class SelectionPolicyTests(unittest.TestCase):
    def _tie_and_boundary_decision(self, root: Path) -> dict:
        low_robust = screen(root, "low_robust", "qwen", worst=0.95, robust=0.44)
        passing = screen(root, "passing", "qwen", worst=0.95, robust=0.45)
        boundary = screen(root, "boundary", "internvl", worst=0.90, robust=1.0)
        return select([low_robust, passing, boundary], THRESHOLDS, root / "selection")

    def test_primary_accuracy_beats_robustness_within_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            high_primary = screen(root, "high_primary", "qwen", worst=0.96, robust=0.40)
            high_robust = screen(root, "high_robust", "qwen", worst=0.95, robust=0.90)
            decision = select([high_primary, high_robust], THRESHOLDS, root / "selection")
            self.assertEqual(decision["family_winners"]["qwen"]["model_key"], "high_primary")
            self.assertFalse(next(r for r in decision["table"] if r["model_key"] == "high_primary")
                             ["proposal_robustness_pass"])

    def test_robustness_breaks_primary_tie_within_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            decision = self._tie_and_boundary_decision(Path(tmp))
            self.assertEqual(decision["family_winners"]["qwen"]["model_key"], "passing")
            self.assertEqual(decision["core7_models"][0], "passing")

    def test_primary_threshold_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            decision = self._tie_and_boundary_decision(Path(tmp))
            row = next(r for r in decision["table"] if r["model_key"] == "boundary")
            self.assertFalse(row["eligible"])
            self.assertTrue(row["proposal_robustness_pass"])
            self.assertIn("worst_position_accuracy", row["failed_gates"])

    def test_robustness_breaks_tie_across_family_winners(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = screen(root, "a", "qwen", worst=0.95, robust=0.50)
            b = screen(root, "b", "internvl", worst=0.95, robust=0.75, backbone="qwen3")
            decision = select([a, b], THRESHOLDS, root / "selection")
            self.assertEqual(decision["winner_ranking"], ["b", "a"])
            self.assertEqual(decision["core7_models"], ["b", "a"])

    def test_exact_tie_prefers_robustness_over_backbone_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = screen(root, "first", "qwen", worst=1.0, robust=0.6)
            robust = screen(root, "robust", "llava", worst=0.96, robust=0.8)
            diverse = screen(root, "diverse", "internvl", worst=0.96, robust=0.6,
                             backbone="qwen3")
            decision = select([first, robust, diverse], THRESHOLDS, root / "selection")
            self.assertEqual(decision["core7_models"], ["first", "robust"])

    def test_near_tie_prefers_backbone_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = screen(root, "first", "qwen", worst=1.0, robust=0.6)
            second_score = screen(root, "second_score", "llava", worst=0.96, robust=0.8)
            diverse = screen(root, "diverse", "internvl", worst=0.95, robust=0.6,
                             backbone="qwen3")
            decision = select([first, second_score, diverse], THRESHOLDS, root / "selection")
            self.assertEqual(decision["core7_models"], ["first", "diverse"])


if __name__ == "__main__":
    unittest.main()
