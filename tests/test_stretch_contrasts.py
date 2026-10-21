"""Static-authored tests; not executed in the October 21 implementation task."""
import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from vlm_mcqa.stretch.data import (build_fixed_text_contrasts, load_dataset,
                                  validate_fixed_text_contrast, validate_bidirectional,
                                  validate_output_path, write_new_json)


def example_dataset():
    items, variants = [], []
    for member, answer in (("x", "red"), ("y", "blue")):
        item = {"item_id": member, "scene_id": "scene-" + member, "split": "discovery",
                "counterfactual_pair_id": "pair-1", "image": member + ".png", "semantic_answer": answer}
        items.append(item)
        for layout, options in enumerate((["red", "blue", "green", "yellow"],
                                           ["green", "yellow", "red", "blue"])):
            labels = ["A", "B", "C", "D"]
            prompt = "Which colour?\n" + "\n".join(f"{l}. {o}" for l, o in zip(labels, options))
            index = options.index(answer)
            variants.append({**item, "variant_id": f"{member}-{layout}", "variant_kind": "position",
                             "question": "Which colour?", "labels": labels, "option_contents": options,
                             "label_scheme": "letters", "correct_index": index, "correct_symbol": labels[index],
                             "prompt": prompt, "question_family": "colour"})
    return {"items": items, "variants": variants, "contrasts": [], "manifest": {"dataset_hash": "test"}}


class FixedTextTests(unittest.TestCase):
    def test_bidirectional_exact_text_and_label_readout(self):
        dataset = example_dataset()
        rows = build_fixed_text_contrasts(dataset)
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["readout"] for r in rows}, {"donor_label_minus_recipient_label"})
        self.assertEqual({r["eligibility"] for r in rows}, {"pending_baseline_inference"})
        validate_bidirectional(rows)
        with self.assertRaises(ValueError):
            validate_bidirectional(rows[:-1])

    def test_mapping_and_split_corruption_rejected(self):
        dataset = example_dataset()
        rows = build_fixed_text_contrasts(dataset)
        variants = {v["variant_id"]: copy.deepcopy(v) for v in dataset["variants"]}
        variants[rows[0]["recipient_variant_id"]]["option_contents"] = ["blue", "red", "green", "yellow"]
        with self.assertRaises(ValueError):
            validate_fixed_text_contrast(rows[0], variants)
        dataset["items"][1]["split"] = "confirmation"
        for v in dataset["variants"]:
            if v["item_id"] == "y":
                v["split"] = "confirmation"
        with self.assertRaises(ValueError):
            build_fixed_text_contrasts(dataset)

    def test_duplicate_variants_and_tar_members_rejected(self):
        dataset = example_dataset()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for key in ("items", "variants"):
                rows = dataset[key] + (dataset[key][:1] if key == "variants" else [])
                (root / (key + ".jsonl")).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset(root)
            archive = root / "duplicate.tar"
            with tarfile.open(archive, "w") as tar:
                for name in ("d/items.jsonl", "d/items.jsonl", "d/variants.jsonl"):
                    content = b"{}\n"
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    tar.addfile(info, io.BytesIO(content))
            with self.assertRaises(ValueError):
                load_dataset(archive)

    def test_archive_read_without_extraction_and_exclusive_output(self):
        dataset = example_dataset()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "release.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for key in ("items", "variants"):
                    content = "\n".join(json.dumps(r) for r in dataset[key]).encode()
                    info = tarfile.TarInfo("release/" + key + ".jsonl")
                    info.size = len(content)
                    tar.addfile(info, io.BytesIO(content))
            loaded = load_dataset(archive)
            self.assertEqual(loaded["variants"], dataset["variants"])
            self.assertFalse((root / "release").exists())
            output = root / "derivative.json"
            write_new_json(output, {"n": 4}, inputs=[archive])
            with self.assertRaises(FileExistsError):
                write_new_json(output, {}, inputs=[archive])
            with self.assertRaises(ValueError):
                validate_output_path(root / "nested.json", [root])


if __name__ == "__main__":
    unittest.main()
