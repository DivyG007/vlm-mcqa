"""Pilot evaluation and plotting helpers (vlm_mcqa.eval.qwen_shapes, analysis.plot_shapes)."""

from __future__ import annotations

import unittest

from vlm_mcqa.analysis.plot_shapes import _lens_scheme
from vlm_mcqa.eval.qwen_shapes import (
    _aggregate,
    _image_token_id,
    _probe_labels,
    _should_collect_lens,
)


class _Object:
    pass


class ShapesEvaluationTests(unittest.TestCase):
    def test_probe_labels_add_default_alphabet_once(self) -> None:
        self.assertEqual(_probe_labels(["Q", "Z", "R", "X"]), list("ABCDQZRX"))
        self.assertEqual(_probe_labels(["A", "B", "C", "D"]), list("ABCD"))

    def test_lens_collected_for_every_position_scheme(self) -> None:
        for scheme in ("letters", "numbers", "rare_letters", "punctuation"):
            self.assertTrue(
                _should_collect_lens(
                    {"variant_family": "answer_position", "label_scheme": scheme}
                )
            )
        self.assertFalse(
            _should_collect_lens(
                {"variant_family": "cross_image", "label_scheme": "letters"}
            )
        )

    def test_lens_scheme_is_explicit_with_legacy_fallback(self) -> None:
        self.assertEqual(
            _lens_scheme({"label_scheme": "punctuation", "label_logits": {}}),
            "punctuation",
        )
        self.assertEqual(
            _lens_scheme({"label_logits": {label: 0.0 for label in "QZRX"}}),
            "rare_letters",
        )

    def test_aggregate_reports_position_and_condition_metrics(self) -> None:
        records = [
            {
                "variant_family": "answer_position",
                "label_scheme": "letters",
                "correct_index": 0,
                "predicted_label": "A",
                "is_correct": True,
                "global_top_token_is_valid_label": True,
                "valid_label_probability_mass": 0.9,
            },
            {
                "variant_family": "answer_position",
                "label_scheme": "letters",
                "correct_index": 1,
                "predicted_label": "A",
                "is_correct": False,
                "global_top_token_is_valid_label": False,
                "valid_label_probability_mass": 0.5,
            },
        ]

        summary = _aggregate(records)

        self.assertEqual(summary["metrics"]["overall"], {"count": 2, "accuracy": 0.5})
        self.assertEqual(
            summary["metrics"]["correct_position:1"], {"count": 1, "accuracy": 1.0}
        )
        self.assertEqual(summary["prediction_counts"], {"A": 2})
        self.assertEqual(
            summary["vocabulary_metrics"]["letters"]["global_top_token_valid_rate"],
            0.5,
        )

    def test_image_token_id_from_top_level_or_nested_config(self) -> None:
        top_level = _Object()
        top_level.config = _Object()
        top_level.config.image_token_id = 151655
        self.assertEqual(_image_token_id(top_level), 151655)

        nested = _Object()
        nested.config = _Object()
        nested.config.text_config = _Object()
        nested.config.text_config.image_token_id = 151655
        self.assertEqual(_image_token_id(nested), 151655)


if __name__ == "__main__":
    unittest.main()
