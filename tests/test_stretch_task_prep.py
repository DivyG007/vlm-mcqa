"""Future verification specification. NOT EXECUTED in static implementation."""
from dataclasses import replace
import tempfile
import hashlib
import json
from pathlib import Path
import unittest

from vlm_mcqa.stretch.task_prep import (
    Rect, TokenCell, ViewTransform, image_option_plan, map_region,
    matched_token_controls, spatial_factorial, validate_split_separation,
)
from vlm_mcqa.stretch.task_cli import preparation_envelope, write_new_manifest


SOURCE = {"dataset_id": "toy-v1", "source_sha256": "a" * 64,
          "scene_id": "scene1", "pair_id": "pair1", "split": "discovery"}


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.views = [ViewTransform("tile", Rect(10, 20, 30, 40), Rect(0, 0, 40, 40))]
        self.cells = [TokenCell(100 + i, "tile", Rect(i * 10, 0, (i + 1) * 10, 20))
                      for i in range(4)]

    def test_crop_resize_and_clip(self):
        self.assertEqual(self.views[0].map_region(Rect(0, 0, 20, 30)), Rect(0, 0, 20, 20))
        self.assertIsNone(self.views[0].map_region(Rect(0, 0, 5, 5)))
        self.assertEqual(map_region(Rect(10, 20, 20, 30), self.views, self.cells,
                                    rule="center"), (100, 101))

    def test_overlap_and_nonoverlap_controls(self):
        region = Rect(10, 20, 20, 30)
        self.assertEqual(map_region(region, self.views, self.cells, rule="overlap", min_fraction=1),
                         (100, 101))
        controls = matched_token_controls(self.cells, region, self.views, count=2, size=1, seed="x")
        self.assertEqual(set(controls), {(102,), (103,)})
        with self.assertRaises(ValueError):
            matched_token_controls(self.cells, region, self.views, count=3, size=1, seed="x")

    def test_mapping_fails_closed(self):
        for fraction in (True, float("nan"), float("inf"), "0.5", None):
            with self.assertRaises(ValueError):
                map_region(Rect(10, 20, 20, 30), self.views, self.cells,
                           rule="overlap", min_fraction=fraction)
        for count, size in ((True, 1), (1, False), (1.5, 1), (1, 1.5)):
            with self.assertRaises(ValueError):
                matched_token_controls(self.cells, Rect(10, 20, 20, 30), self.views,
                                       count=count, size=size, seed="x")
        for bad in [self.cells + [self.cells[0]], [replace(self.cells[0], view_id="unknown")],
                    [replace(self.cells[0], footprint=Rect(-1, 0, 5, 5))]]:
            with self.assertRaises(ValueError):
                map_region(Rect(10, 20, 20, 30), self.views, bad, rule="center")

    def test_multiple_tiles_keep_explicit_sequence_indices(self):
        views = [self.views[0], replace(self.views[0], view_id="other")]
        cells = self.cells + [TokenCell(500, "other", Rect(0, 0, 10, 20))]
        self.assertEqual(map_region(Rect(10, 20, 20, 30), views, cells, rule="center"),
                         (100, 101, 500))


class ManifestTests(unittest.TestCase):
    def plan(self, **kwargs):
        args = dict(query_asset="query.png", option_assets={c: c + ".png" for c in "abcd"},
                    target_option="b", panel_order=list("dcba"), labels_by_position=list("1234"))
        args.update(kwargs)
        return image_option_plan(SOURCE, **args)

    def test_independent_permutations_and_no_visible_semantics(self):
        record = self.plan()
        self.assertEqual(record["annotations"]["correct_symbol"], "3")
        self.assertEqual(record, self.plan())
        self.assertNotIn("semantic_id", record["visible"]["options"][0])
        changed = self.plan(labels_by_position=list("4321"))
        self.assertEqual(changed["annotations"]["correct_symbol"], "2")
        self.assertNotEqual(record["derivative_id"], changed["derivative_id"])
        with self.assertRaises(ValueError):
            self.plan(panel_order=list("aabc"))

    def test_split_leakage_rejected(self):
        a = self.plan()
        b = {**a, "source": {**SOURCE, "split": "confirmation"}}
        with self.assertRaises(ValueError):
            validate_split_separation([a, b])

    def test_envelope_preserves_exact_hash_scope(self):
        task = self.plan()
        original = dict(task)
        envelope = preparation_envelope(task, path_base=Path("specs"), source_path=Path("data.json"))
        self.assertEqual(task, original)
        body = {k: v for k, v in envelope["task"].items() if k != "derivative_id"}
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        expected = "image-options-v1-" + hashlib.sha256(encoded).hexdigest()[:20]
        self.assertEqual(envelope["task"]["derivative_id"], expected)
        self.assertNotIn("path_base", envelope["task"])
        self.assertFalse(envelope["preparation"]["source_hash_verified"])

    def test_spatial_factorial_and_crossfactor_confounds(self):
        cells = [{"condition_id": f"{q}{o}", "question_level": q, "option_level": o,
                  "question": f"ask region {q}", "question_region": f"r{q}",
                  "option_region": f"r{o}", "image": "same.png",
                  "option_ids": [f"{o}-{i}" for i in range(4)], "labels": list("ABCD"),
                  "correct_option_id": f"{o}-0", **{k: SOURCE[k] for k in ("scene_id", "pair_id", "split")}}
                 for q in (0, 1) for o in (0, 1)]
        self.assertEqual(len(spatial_factorial(SOURCE, cells)["conditions"]), 4)
        for bad in (True, 0.0):
            invalid = [{**cells[0], "question_level": bad}, *cells[1:]]
            with self.assertRaises(ValueError):
                spatial_factorial(SOURCE, invalid)
        with self.assertRaises(ValueError):
            spatial_factorial(SOURCE, cells[:3])
        cells[1] = {**cells[1], "question": "confounded"}
        with self.assertRaises(ValueError):
            spatial_factorial(SOURCE, cells)

    def test_writer_preserves_existing_and_overlapping_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            input_dir = base / "inputs"
            input_dir.mkdir()
            source = input_dir / "input.json"
            source.write_text("{}", encoding="utf-8")
            output = base / "derivative"
            write_new_manifest(output, self.plan(), inputs=[source])
            with self.assertRaises(ValueError):
                write_new_manifest(output, self.plan(), inputs=[source])
            with self.assertRaises(ValueError):
                write_new_manifest(source / "child", self.plan(), inputs=[source])
            with self.assertRaises(ValueError):
                write_new_manifest(input_dir / "new-output", self.plan(), inputs=[source])
            self.assertEqual(source.read_text(encoding="utf-8"), "{}")


if __name__ == "__main__":
    unittest.main()
