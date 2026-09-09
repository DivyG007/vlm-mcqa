"""Pilot W&B metric payload (vlm_mcqa.tracking.wandb_report)."""

from __future__ import annotations

import unittest

from vlm_mcqa.tracking.wandb_report import _metric_payload


class WandbReportTests(unittest.TestCase):
    def test_metric_payload_is_flat(self) -> None:
        summary = {
            "elapsed_seconds": 12.5,
            "record_count": 40,
            "metrics": {"overall": {"count": 40, "accuracy": 0.95}},
            "vocabulary_metrics": {
                "letters": {
                    "count": 40,
                    "global_top_token_valid_rate": 0.9,
                    "mean_valid_label_probability_mass": 0.8,
                }
            },
        }

        payload = _metric_payload(summary)

        self.assertEqual(payload["accuracy/overall"], 0.95)
        self.assertEqual(payload["vocabulary/letters/global_top_valid_rate"], 0.9)
        self.assertEqual(payload["vocabulary/letters/valid_label_mass"], 0.8)
        self.assertEqual(payload["runtime/evaluation_seconds"], 12.5)
        self.assertEqual(payload["runtime/record_count"], 40.0)


if __name__ == "__main__":
    unittest.main()
