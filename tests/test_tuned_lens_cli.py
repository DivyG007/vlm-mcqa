"""CLI glue: cache -> train -> evaluate -> estimate through the real command
functions on a tiny random VLM (model loading is the only substitution)."""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

try:
    import torch
    import transformers
    from test_core7_tiny_models import make_adapter, make_dataset
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch/transformers not installed")
class TunedLensCliTests(unittest.TestCase):
    def test_pilot_pipeline_writes_manifests_and_estimate(self):
        from vlm_mcqa.tuned_lens import cli

        warnings.filterwarnings("ignore")
        transformers.logging.set_verbosity_error()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_dataset(root / "data")
            configs = root / "configs"
            configs.mkdir()
            (configs / "corpus.json").write_text(json.dumps({
                "corpus_splits": ["screening"], "schemes": ["letters", "numbers", "rare_letters"], "seed": 0,
                "val_scenes": 1, "pilot_train_scenes": 2, "pilot_val_scenes": 1, "shard_size": 8,
                "final_gate_limit": 0.5}))
            (configs / "train.json").write_text(json.dumps({
                "lr_scale": 1.0, "momentum": 0.9, "weight_decay": 1e-3, "steps": 8, "pilot_steps": 4,
                "tokens_per_step": 32, "microbatch_tokens": 16, "grad_clip": 1.0, "eval_every": 2,
                "log_every": 1, "seed": 0}))
            (configs / "models.json").write_text(json.dumps({"tiny_qwen_vl": {"models_config": "unused"}}))
            (configs / "wandb.json").write_text(json.dumps({"entity": "unused", "project": "unused"}))
            adapter = make_adapter("qwen_vl")
            out = root / "run"
            base = ["--model", "tiny_qwen_vl", "--data", str(root / "data"), "--out", str(out),
                    "--config-dir", str(configs), "--no-wandb", "--pilot"]
            with mock.patch.object(cli, "_adapter", lambda args, configs: (adapter, adapter.spec)):
                cli.main(["cache", *base])
                cli.main(["train", *base])
                cli.main(["evaluate", *base, "--n-items", "2"])
                cli.main(["estimate", *base, "--full-variants", "100", "--eval-variants", "10"])
            cache = json.loads((out / "cache" / "cache_manifest.json").read_text())
            self.assertTrue(cache["final_gate_passed"] and cache["calibration_all_passed"])
            self.assertEqual(cache["translator_stages"], adapter.num_layers)
            self.assertIn("source_git_commit", cache)
            summary = json.loads((out / "translators" / "train_summary.json").read_text())
            self.assertEqual([s["stage"] for s in summary["stages"]], list(range(adapter.num_layers)))
            self.assertEqual(summary["train_config"]["steps"], 4)  # pilot steps
            comparison = json.loads((out / "eval" / "comparison_behavior.json").read_text())
            self.assertEqual(len(comparison["translator_sha256"]), adapter.num_layers)
            self.assertTrue(comparison["figures"])
            estimate = json.loads((out / "estimate.json").read_text())
            self.assertGreater(estimate["hours"]["with_40pct_margin"], estimate["hours"]["total"])
            with self.assertRaises(SystemExit):  # --n-items is pilot-only
                cli.main(["evaluate", *[a for a in base if a != "--pilot"], "--n-items", "2"])


if __name__ == "__main__":
    unittest.main()
