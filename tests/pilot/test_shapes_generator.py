"""Synthetic Shapes dataset generator (vlm_mcqa.data.generate_shapes)."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from PIL import Image

from vlm_mcqa.data.generate_shapes import generate_dataset


class ShapesGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.output = Path(self.tempdir.name) / "dataset"
        self.manifest = generate_dataset(self.output, num_pairs=3, seed=7, image_size=224)
        self.records = [json.loads(line) for line in self.manifest.read_text().splitlines()]

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_record_counts_and_image_size(self) -> None:
        self.assertEqual(len(self.records), 3 * 40)
        self.assertEqual(len(list((self.output / "images").glob("*.png"))), 6)
        with Image.open(self.output / self.records[0]["image"]) as image:
            self.assertEqual(image.size, (224, 224))

    def test_cross_image_pairs_keep_text_and_change_answer(self) -> None:
        groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for record in self.records:
            if record["variant_family"] == "cross_image":
                groups[(record["pair_id"], record["label_scheme"])].append(record)

        self.assertEqual(len(groups), 3 * 4)
        for pair in groups.values():
            self.assertEqual(len(pair), 2)
            first, second = sorted(pair, key=lambda row: row["pair_member"])
            self.assertEqual(first["prompt"], second["prompt"])
            self.assertEqual(first["option_contents"], second["option_contents"])
            self.assertNotEqual(first["correct_content"], second["correct_content"])
            self.assertNotEqual(first["correct_label"], second["correct_label"])
            self.assertEqual(
                sorted(obj["color"] for obj in first["objects"]),
                sorted(obj["color"] for obj in second["objects"]),
            )

    def test_position_variants_cover_every_answer_position(self) -> None:
        groups: dict[tuple[str, str, str], set[int]] = defaultdict(set)
        for record in self.records:
            if record["variant_family"] == "answer_position":
                key = (record["pair_id"], record["pair_member"], record["label_scheme"])
                groups[key].add(record["correct_index"])
        self.assertEqual(len(groups), 3 * 2 * 4)
        self.assertTrue(all(positions == {0, 1, 2, 3} for positions in groups.values()))


if __name__ == "__main__":
    unittest.main()
