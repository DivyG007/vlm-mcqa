"""Per-run aggregation of method outputs (core7.aggregate)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from support import HAS_TORCH, requires_torch

if HAS_TORCH:
    from vlm_mcqa.core7.aggregate import aggregate_run


@requires_torch
class AggregateTests(unittest.TestCase):
    def test_no_content_outcome_has_no_peak_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{"contrast_type": "content_and_position", "group": "final", "site": "resid_post",
                     "layer": layer, "outcome": "position"} for layer in (0, 4)]
            (root / "patching_discovery.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
            aggregate = aggregate_run(root, "discovery")
            self.assertIsNone(aggregate["cp_content_peak_layer"])


if __name__ == "__main__":
    unittest.main()
