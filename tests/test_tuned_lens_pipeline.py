"""Tuned Lens pipeline checks on tiny random VLMs (no downloads, CPU only).

Covers the frozen fit-position rule, the leak-proof corpus split, identity at
initialization, resumable caching and training, and exact parity of the
evaluator's Logit Lens rows with Core-7's own ``run_lens`` rows. Says nothing
about real-model results.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path

try:
    import torch
    import transformers
    from test_core7_tiny_models import make_adapter, make_dataset
except ImportError:  # pragma: no cover
    torch = None

CORPUS = {"corpus_splits": ["screening"], "schemes": ["letters", "numbers", "rare_letters"], "seed": 0,
          "val_scenes": 1, "pilot_train_scenes": 2, "pilot_val_scenes": 1, "shard_size": 5}
TRAIN = {"lr_scale": 1.0, "momentum": 0.9, "weight_decay": 1e-3, "steps": 4, "tokens_per_step": 32,
         "microbatch_tokens": 16, "grad_clip": 1.0, "eval_every": 2, "log_every": 1, "seed": 0}


class StubTokenizer:
    def decode(self, ids):
        return {7: ".", 8: "A", 9: "B"}.get(ids[0], "x")


class PositionRuleTests(unittest.TestCase):
    def test_excludes_image_final_labels_predecessors_and_delimiters(self):
        from vlm_mcqa.tuned_lens.positions import fit_positions

        ids = [1, 1, 2, 3, 4, 8, 7, 5, 9, 7, 6, 4, 3]
        groups = {"image": [0, 1], "labels": [5, 8]}
        got = fit_positions(ids, groups, StubTokenizer(), num_labels=2)
        self.assertEqual(got, [2, 3, 10, 11])  # 4 precedes label 5; 6 and 9 are "." delimiters; 12 is final

    def test_refuses_unlocated_labels(self):
        from vlm_mcqa.tuned_lens.positions import fit_positions

        with self.assertRaises(ValueError):
            fit_positions([1, 2, 3], {"image": [], "labels": [1]}, StubTokenizer(), num_labels=4)


@unittest.skipIf(torch is None, "torch/transformers not installed")
class TinyPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")
        transformers.logging.set_verbosity_error()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.data = cls.root / "data"
        make_dataset(cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _session(self, adapter, name):
        from vlm_mcqa.core7.calibration import run_calibration
        from vlm_mcqa.core7.session import Session
        from test_core7_tiny_models import PROFILE

        session = Session(adapter, self.data, self.root / name, PROFILE, seed=0)
        report = run_calibration(session, num_variants=1)
        self.assertTrue(report["all_passed"])
        return session

    def test_corpus_split_is_scene_disjoint_and_excludes_evaluation(self):
        from vlm_mcqa.tuned_lens.corpus import load_corpus

        corpus = load_corpus(self.data, CORPUS)
        train = {v["scene_id"] for v in corpus["train"]}
        val = {v["scene_id"] for v in corpus["val"]}
        self.assertEqual(len(val), 1)
        self.assertFalse(train & val)
        self.assertTrue(all(v["split"] == "screening" for v in corpus["train"] + corpus["val"]))
        self.assertEqual(load_corpus(self.data, CORPUS)["manifest"]["corpus_sha256"], corpus["manifest"]["corpus_sha256"])
        with self.assertRaises(ValueError):
            load_corpus(self.data, {**CORPUS, "corpus_splits": ["screening", "behavior"]})

    def test_zero_translator_is_logit_lens_and_final_stage_is_identity(self):
        from vlm_mcqa.tuned_lens.translator import TunedLens, lens_logits

        adapter = make_adapter("paligemma")
        handle, d = adapter.handle, int(adapter.handle.info["hidden_size"])
        lens = TunedLens(adapter.num_layers, d)
        h = torch.randn(5, d)
        for stage in range(adapter.num_layers + 1):
            torch.testing.assert_close(lens_logits(handle, lens, h, stage), lens_logits(handle, None, h, stage))
        with torch.no_grad():
            lens.translators[0].weight.fill_(0.01)
        self.assertFalse(torch.allclose(lens_logits(handle, lens, h, 0), lens_logits(handle, None, h, 0)))

    def test_cache_train_resume_and_evaluate(self):
        from vlm_mcqa.core7.lens import run_lens
        from vlm_mcqa.core7.aggregate import read_rows
        from vlm_mcqa.tuned_lens.cache import build_cache, load_stage, shard_paths
        from vlm_mcqa.tuned_lens.corpus import load_corpus
        from vlm_mcqa.tuned_lens.evaluate import run_evaluation
        from vlm_mcqa.tuned_lens.train import train_stage
        from vlm_mcqa.tuned_lens.translator import TunedLens, load_lens
        from PIL import Image

        for family in ("qwen_vl", "paligemma"):
            with self.subTest(family=family):
                adapter = make_adapter(family)
                session = self._session(adapter, f"cal_{family}")  # selects the softcap, as the CLI does
                corpus = load_corpus(self.data, CORPUS)
                load = lambda rel: Image.open(self.data / rel).convert("RGB")  # noqa: E731
                cache = self.root / f"cache_{family}"
                first = {s: build_cache(adapter, load, corpus[s], cache / s, shard_size=5, log=lambda _: None)
                         for s in ("train", "val")}
                self.assertEqual(first["train"]["skipped"], 0)
                self.assertLess(first["train"]["max_final_gate"], 1e-3)
                stamp = shard_paths(cache / "train", 0)[0].stat().st_mtime_ns
                again = build_cache(adapter, load, corpus["train"], cache / "train", shard_size=5, log=lambda _: None)
                self.assertEqual(again["tokens"], first["train"]["tokens"])
                self.assertEqual(shard_paths(cache / "train", 0)[0].stat().st_mtime_ns, stamp)  # reused, not rewritten
                L, d = adapter.num_layers, int(adapter.handle.info["hidden_size"])
                final_train, final_val = load_stage(cache / "train", L), load_stage(cache / "val", L)
                self.assertEqual(tuple(final_train.shape), (first["train"]["tokens"], d))

                lens = TunedLens(L, d)
                out = self.root / f"translators_{family}" / "nested"  # train_stage must create it

                class Interrupt(Exception):
                    pass

                class Tracker:  # raises on the 3rd validation log (after the step-2 checkpoint is saved)
                    calls = 0

                    def log(self, metrics):
                        if any(k.endswith("val_kl") for k in metrics):
                            Tracker.calls += 1
                            if Tracker.calls == 3:
                                raise Interrupt

                stage_data = lambda s: ((load_stage(cache / "train", s), final_train),  # noqa: E731
                                        (load_stage(cache / "val", s), final_val))
                with self.assertRaises(Interrupt):
                    train_stage(adapter.handle, lens, 0, *stage_data(0), TRAIN, out, tracker=Tracker(), log=lambda _: None)
                self.assertTrue((out / "translator_stage_00.partial.pt").exists())
                messages = []
                summary = train_stage(adapter.handle, lens, 0, *stage_data(0), TRAIN, out, log=messages.append)
                self.assertTrue(any("resumed at step 2" in m for m in messages))
                self.assertEqual([h["step"] for h in summary["history"]], [0, 2, 4])
                self.assertAlmostEqual(summary["history"][0]["val_kl"], summary["logit_lens_val_kl"], places=5)
                self.assertLessEqual(summary["tuned_val_kl"], summary["logit_lens_val_kl"] + 1e-9)
                self.assertFalse((out / "translator_stage_00.partial.pt").exists())
                for stage in range(1, L):
                    train_stage(adapter.handle, lens, stage, *stage_data(stage), TRAIN, out, log=lambda _: None)
                skipped = []
                train_stage(adapter.handle, lens, 0, *stage_data(0), TRAIN, out, log=skipped.append)
                self.assertTrue(any("already complete" in m for m in skipped))
                loaded, hashes = load_lens(out, L, d)
                self.assertEqual(len(hashes), L)

                # Parity: evaluator's Logit Lens rows == Core-7 run_lens rows; zero lens == Logit Lens.
                run_lens(session, split="behavior", n_items=2)
                reference = {(r["variant_id"], r["stage"]): r for r in read_rows(session.out_dir / "lens_behavior.jsonl")}
                eval_session = self._session(adapter, f"eval_{family}")
                run_evaluation(eval_session, TunedLens(L, d), n_items=2, log=lambda _: None)
                logit_rows = list(read_rows(eval_session.out_dir / "lens_behavior_logit.jsonl"))
                tuned_rows = list(read_rows(eval_session.out_dir / "lens_behavior_tuned.jsonl"))
                self.assertEqual(len(logit_rows), len(reference))
                for row, twin in zip(logit_rows, tuned_rows):
                    ref = reference[(row["variant_id"], row["stage"])]
                    for key in ("margin", "mean_label_logit", "correct_restricted_prob", "label_mass_full"):
                        self.assertAlmostEqual(row[key], ref[key], places=5)
                        self.assertAlmostEqual(row[key], twin[key], places=4)
                comparison = json.loads((eval_session.out_dir / "comparison_behavior.json").read_text())
                self.assertIn("logit/letters/label_readable_stage", comparison["milestones"])


if __name__ == "__main__":
    unittest.main()
