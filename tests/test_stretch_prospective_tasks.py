import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from vlm_mcqa.stretch.prospective_tasks import (
    build_image_option_parent, build_spatial_parent, select_spatial_cohort,
    validate_cohort_separation, write_audit_sheet,
)
from vlm_mcqa.stretch.task_runtime import _load_tasks


def _objects():
    colors = ["red", "blue", "green", "yellow", "purple"]
    rows = []
    for index, color in enumerate(colors):
        x, y = 5 + (index % 3) * 60, 5 + (index // 3) * 80
        rows.append({"id": f"o{index}", "size": "small", "material": "rubber",
                     "shape": f"shape{index}", "description": f"the small rubber shape{index}",
                     "color": color, "bbox": [x, y, x + 35, y + 45]})
    return rows


class ProspectiveTaskTests(unittest.TestCase):
    def _scene(self, root, split="discovery", identity="s0", render=False):
        objects = _objects()
        metadata = root / f"{identity}.json"
        metadata.write_text(json.dumps({"objects": objects}), encoding="utf-8")
        image = root / f"{identity}.png"
        if render:
            from PIL import Image, ImageDraw
            canvas = Image.new("RGB", (190, 170), "white")
            draw = ImageDraw.Draw(canvas)
            palette = ["red", "blue", "green", "yellow", "purple"]
            for obj, color in zip(objects, palette):
                draw.rectangle(obj["bbox"], fill=color)
            canvas.save(image)
        else:
            image.write_bytes(b"fixture-image-not-decoded-by-spatial-preparation")
        return {"source": {"dataset_id": "fixture", "source_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                           "scene_id": identity, "pair_id": "p-" + identity, "split": split},
                "source_metadata_path": str(metadata), "objects": objects,
                "image": image.name, "image_path": str(image),
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}

    def test_spatial_build_and_underfill_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = self._scene(root)
            first = build_spatial_parent(scene, phase="discovery", seed=0)
            second = build_spatial_parent(scene, phase="discovery", seed=0)
            self.assertEqual(first, second)
            self.assertEqual([options[:2] for options in first["option_sets"]],
                             [first["option_sets"][0][:2]] * 2)
            cohort = select_spatial_cohort([scene], phase="discovery", target_units=2, min_units=1, seed=0)
            self.assertEqual(cohort["status"], "reduced_power")
            self.assertEqual(cohort["effective_target_units"], 1)
            manifest = root / "spatial-tasks.json"
            manifest.write_text(json.dumps(cohort["tasks"]), encoding="utf-8")
            loaded, _ = _load_tasks({"name": "spatial", "split": "discovery", "tasks_path": str(manifest)},
                                    {"seed": 0, "target_units": 1})
            self.assertEqual(len(loaded), 4)

    def test_spatial_allows_nonquery_description_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = self._scene(root)
            scene["objects"][3]["description"] = scene["objects"][4]["description"]
            metadata = Path(scene["source_metadata_path"])
            metadata.write_text(json.dumps({"objects": scene["objects"]}), encoding="utf-8")
            scene["source"]["source_sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
            built = build_spatial_parent(scene, phase="discovery", seed=0)
            selected = set(built["query_objects"])
            self.assertFalse(selected & {scene["objects"][3]["id"], scene["objects"][4]["id"]})

    def test_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            discovery = select_spatial_cohort([self._scene(root, "discovery", "same")],
                                               phase="discovery", target_units=1, min_units=1, seed=0)
            confirmation_scene = self._scene(root, "confirmation", "other")
            confirmation_scene["source"]["scene_id"] = "same"
            confirmation = select_spatial_cohort([confirmation_scene], phase="confirmation",
                                                  target_units=1, min_units=1, seed=0)
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_cohort_separation(discovery, confirmation)

    def test_image_option_builder_makes_orthogonal_layouts_and_audit(self):
        try:
            import PIL  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("Pillow unavailable in local unit-test interpreter")
        font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        if not font.is_file():
            self.skipTest("DejaVu fixture font unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = self._scene(root, render=True)
            assets = root / "assets"
            assets.mkdir()
            tasks = build_image_option_parent(
                scene, phase="discovery", seed=0, output=assets / "parent", assets_root=assets,
                font_path=font, font_sha256=hashlib.sha256(font.read_bytes()).hexdigest())
            self.assertEqual(len(tasks), 4)
            visual = [row for row in tasks if row["condition"] == "visual"]
            self.assertEqual(len(visual), 2)
            self.assertNotEqual(visual[0]["correct_index"], visual[1]["correct_index"])
            manifest = root / "image-option-tasks.json"
            manifest.write_text(json.dumps(tasks), encoding="utf-8")
            loaded, _ = _load_tasks({"name": "image_options", "split": "discovery", "tasks_path": str(manifest)},
                                    {"seed": 0, "target_units": 1})
            self.assertEqual(len(loaded), 4)
            audit = write_audit_sheet(tasks, assets_root=assets, output=assets / "audit.png")
            self.assertEqual(audit["visual_tasks"], 2)
            self.assertTrue((assets / "audit.png").is_file())


if __name__ == "__main__":
    unittest.main()
