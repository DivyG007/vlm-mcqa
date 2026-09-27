"""W&B destination, metric flattening and offline logging (core7.tracking)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from support import requires_wandb
from vlm_mcqa.core7.tracking import WANDB_ENTITY, WANDB_PROJECT, Tracker, destination, flatten


class TrackingTests(unittest.TestCase):
    def test_default_wandb_destination(self) -> None:
        self.assertEqual((WANDB_ENTITY, WANDB_PROJECT), ("vlm-mcqa", "smoke-testing"))

    def test_destination_overridden_by_environment(self) -> None:
        with patch.dict(os.environ, {"CORE7_WANDB_ENTITY": "personal-workspace",
                                     "CORE7_WANDB_PROJECT": "personal-project"}):
            self.assertEqual(destination(), ("personal-workspace", "personal-project"))

    def test_flatten_keeps_numeric_leaves(self) -> None:
        flat = flatten({"a": {"b": 1, "c": [1, 2], "d": True, "e": float("nan")}, "f": "x"})
        self.assertEqual(flat, {"a/b": 1.0, "a/d": 1.0})

    def test_disabled_tracker_is_noop(self) -> None:
        tracker = Tracker.start(enabled=False, out_dir=Path("."), name="x", job_type="t", config={})
        tracker.log({"a": 1})
        tracker.curve("c", {0: 1.0})
        tracker.finish()
        self.assertIsNone(tracker.url)

    @requires_wandb
    def test_offline_run_logs_curves_figures_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "figures").mkdir()
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            fig.savefig(out / "figures" / "m1.png")
            (out / "run.json").write_text("{}")
            with patch.dict(os.environ, {"WANDB_MODE": "offline"}):
                tracker = Tracker.start(enabled=True, out_dir=out, name="unit", job_type="test",
                                        config={"model_key": "tiny"}, tags=["unit"])
                tracker.log({"behavior": {"accuracy": 0.5}})
                tracker.aggregate({"position_curve": {0: 0.1, 1: 0.9}, "position_transfer_layer": 1}, "discovery")
                tracker.figures(out)
                tracker.artifact(out, "unit-artifact")
                tracker.finish()
            info = json.loads((out / "wandb_test.json").read_text())
            self.assertEqual((info["entity"], info["project"]), destination())


if __name__ == "__main__":
    unittest.main()
