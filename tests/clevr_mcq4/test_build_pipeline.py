"""Plan -> render -> finalize dataset build (clevr_mcq4.build), with a stub renderer."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import unittest
from pathlib import Path

from vlm_mcqa.clevr_mcq4.build import finalize, plan
from support import REPO_ROOT
from vlm_mcqa.clevr_mcq4.variants import read_jsonl

from .fixtures import random_scene


class BuildPipelineTests(unittest.TestCase):
    def test_build_end_to_end_with_stub_renderer(self) -> None:
        rng = random.Random(3)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "images").mkdir()
            scenes = [random_scene(i, rng) for i in range(200)]
            for s in scenes:
                (tmp / "images" / s["image_filename"]).write_bytes(b"png")
            (tmp / "scenes.json").write_text(json.dumps({"scenes": scenes}))
            cfg = json.loads((REPO_ROOT / "configs/core7/clevr_mcq4_dataset.json").read_text())
            cfg["split_sizes"] = {"calibration": 6, "screening": 8, "behavior": 8, "discovery": 5, "confirmation": 5}
            (tmp / "cfg.json").write_text(json.dumps(cfg))
            out = tmp / "out"
            plan(argparse.Namespace(scenes=str(tmp / "scenes.json"), images_dir=str(tmp / "images"),
                                    config=tmp / "cfg.json", out_dir=str(out)))
            jobs = json.loads((out / "render_jobs.json").read_text())
            self.assertEqual(len(jobs), 20)
            for job in jobs:  # stand-in for Blender
                Path(job["output_image"]).write_bytes(b"png")
                scene = dict(job["scene"])
                scene["objects"] = [{**o, "bbox": [0, 0, 10, 10]} for o in scene["objects"]]
                Path(job["output_scene"]).write_text(json.dumps(scene))
            finalize(argparse.Namespace(out_dir=str(out)))
            items = read_jsonl(out / "items.jsonl")
            variants = read_jsonl(out / "variants.jsonl")
            contrasts = read_jsonl(out / "contrasts.jsonl")
            self.assertEqual(len(items), 6 + 8 + 8 + 20)
            scenes_by_split = {}
            for item in items:
                self.assertEqual(scenes_by_split.setdefault(item["scene_id"], item["split"]), item["split"])
            self.assertTrue(all(v["split"] == next(i["split"] for i in items if i["item_id"] == v["item_id"])
                                for v in variants[:50]))
            self.assertIn("random_pair", {c["contrast_type"] for c in contrasts})
            manifest = json.loads((out / "dataset_manifest.json").read_text())
            self.assertEqual(len(manifest["dataset_hash"]), 16)


if __name__ == "__main__":
    unittest.main()
