"""Protocol invariants; authored only, execution deferred by user instruction."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from vlm_mcqa.stretch.protocol import compile_recipe, load_frozen_heads
from vlm_mcqa.stretch.runtime import validate_runtime_context
from vlm_mcqa.stretch.runtime import REQUIRED_GATES
from vlm_mcqa.stretch.runtime import validate_head_control, head_control_conditions, evaluate_head_control
from vlm_mcqa.stretch.data import digest


def frozen_source():
    return {"model_key": "fixture", "dataset_hash": "dataset", "freeze_hash": "freeze",
            "source_sha256": "head-source", "head_layers": [1, 2, 3, 4],
            "head_candidates": [[layer, head] for layer in (1, 2, 3, 4) for head in range(4)]}


def recipe(**kwargs):
    return compile_recipe(frozen_source(), num_heads=32, num_layers=6, image_layer=0,
                          image_site="resid_post", seed=7, **kwargs)


def control_fixture(kind):
    recipient = {"variant_id": "recipient", "item_id": "item-r", "counterfactual_pair_id": "pair-r",
                 "scene_id": "scene-r", "split": "discovery", "image": "r.png", "semantic_answer": "red",
                 "labels": ["A", "B", "C", "D"], "option_contents": ["red", "blue", "green", "yellow"],
                 "correct_index": 0, "correct_symbol": "A", "label_scheme": "letters", "question": "Colour?"}
    donor = copy.deepcopy(recipient)
    donor.update({"variant_id": "donor", "correct_index": 1, "correct_symbol": "B"})
    if kind == "position":
        donor["option_contents"] = ["yellow", "red", "blue", "green"]
    else:
        donor.update({"item_id": "item-d", "counterfactual_pair_id": "pair-d", "scene_id": "scene-d",
                      "image": "d.png", "semantic_answer": "blue"})
    variants = {v["variant_id"]: v for v in (donor, recipient)}
    items = {v["item_id"]: {k: v[k] for k in
             ("item_id", "counterfactual_pair_id", "scene_id", "split", "image", "semantic_answer")}
             for v in (donor, recipient)}
    contrast = {"contrast_id": "control-1", "contrast_type": kind, "split": "discovery", "pair_id": "pair-r",
                "donor_variant_id": "donor", "recipient_variant_id": "recipient", "donor_symbol": "B",
                "recipient_symbol": "A", "donor_content": donor["semantic_answer"], "recipient_content": "red"}
    return contrast, variants, items


class RecipeTests(unittest.TestCase):
    def test_control_metadata_and_semantics(self):
        for kind in ("position", "random_pair"):
            contrast, variants, items = control_fixture(kind)
            metadata = validate_head_control(contrast, variants, items)
            self.assertEqual(metadata["role"], "control")
            self.assertEqual(metadata["sample_unit"], "pair-r")
            self.assertEqual(metadata["donor_pair_id"], "pair-r" if kind == "position" else "pair-d")
        contrast, variants, items = control_fixture("position")
        variants["donor"]["image"] = "changed.png"
        with self.assertRaises(ValueError):
            validate_head_control(contrast, variants, items)
        contrast, variants, items = control_fixture("random_pair")
        variants["donor"]["counterfactual_pair_id"] = "pair-r"
        items["item-d"]["counterfactual_pair_id"] = "pair-r"
        with self.assertRaises(ValueError):
            validate_head_control(contrast, variants, items)

    def test_head_control_subset_never_restores_images(self):
        conditions = head_control_conditions(recipe())
        self.assertEqual(len(conditions), 27)
        self.assertFalse({c["operation"] for c in conditions} & {"image_restore", "self_image", "restore_clamp"})
        with self.assertRaises(PermissionError):
            evaluate_head_control(object(), {}, {}, calibration={}, identity={}, execute=False)

    def test_frozen_order_random_counts_and_multi_layer_reference(self):
        compiled = recipe()
        self.assertEqual(compiled["sets"]["top4"], frozen_source()["head_candidates"][:4])
        self.assertEqual(compiled["condition_count"], 52)
        for name, heads in compiled["sets"].items():
            if name.startswith("random4"):
                self.assertEqual(len(heads), 4)
                self.assertEqual(Counter(l for l, _ in heads), Counter({1: 4}))
                self.assertFalse(set(map(tuple, heads)) & set(map(tuple, compiled["sets"]["top4"])))
        self.assertEqual(len(compiled["sets"]["all_heads"]), 128)
        self.assertEqual({l for l, _ in compiled["sets"]["all_heads"]}, {1, 2, 3, 4})
        self.assertEqual(recipe(), compiled)

    def test_expanded_controls_explicit_and_sender_temporal_order(self):
        self.assertIn("random16_00", recipe(expanded_controls=True)["sets"])
        self.assertNotIn("random16_00", recipe()["sets"])
        with self.assertRaises(ValueError):
            compile_recipe(frozen_source(), num_heads=32, num_layers=6, image_layer=1,
                           image_site="resid_post", seed=1)
        compile_recipe(frozen_source(), num_heads=32, num_layers=6, image_layer=1,
                       image_site="resid_pre", seed=1)

    def test_localization_only_and_old_qwen_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "freeze.json"
            path.write_text(json.dumps({"freeze_hash": "old", "head_layers": [1]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_frozen_heads(path, model_key="fixture", dataset_hash="data")
            with self.assertRaises(ValueError):
                load_frozen_heads(path, model_key="qwen25vl_7b", dataset_hash="data")

    def test_corrected_qwen_compact_strings_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "m3.json"
            path.write_text(json.dumps({"freeze_hash": "4f991401f5cb7239", "head_layers": [1, 2],
                                        "method": "M3 pair-balanced head localization", "contrast_type": "position",
                                        "label_scheme": "letters", "semantic_pairs_per_split": 64,
                                        "directed_contrasts_per_split": 96,
                                        "frozen_candidates": [f"L{l}H{h}" for l in (1, 2) for h in range(8)]}),
                            encoding="utf-8")
            frozen = load_frozen_heads(path, model_key="qwen25vl_7b", dataset_hash="f5573e25f3aa2bf2")
            self.assertEqual(frozen["head_candidates"][:2], [[1, 0], [1, 1]])
            self.assertEqual(frozen["model_data_binding"], "known_corrected_qwen_compact_release")
            with self.assertRaises(ValueError):
                load_frozen_heads(path, model_key="llava_ov_7b_si", dataset_hash="f5573e25f3aa2bf2")
            with self.assertRaises(ValueError):
                load_frozen_heads(path, model_key="qwen25vl_7b", dataset_hash="wrong")

    def test_runtime_denied_before_session_or_torch_access(self):
        with self.assertRaises(PermissionError):
            validate_runtime_context(object(), {}, {}, {}, execute=False)

    def test_runtime_provenance_and_real_gate_evidence_required(self):
        compiled = recipe()
        spec = SimpleNamespace(key="fixture", hf_id="org/checkpoint", revision="a" * 40)
        description = {"model_key": "fixture", "hf_id": spec.hf_id}
        session = SimpleNamespace(
            manifest={"dataset_hash": "dataset"}, release_manifest={"image_aware_release_hash": "images"},
            adapter=SimpleNamespace(spec=spec, handle=SimpleNamespace(num_heads=32), num_layers=6,
                                    describe=lambda: description))
        calibration = {"gates": {name: {"passed": True} for name in REQUIRED_GATES},
                       "all_passed": True, "adapter": description, "variants": ["v1"]}
        identity = {key: compiled[key] for key in
                    ("model_key", "dataset_hash", "freeze_hash", "head_source_sha256", "recipe_sha256")}
        identity.update({"source_identity": "source-manifest", "checkpoint_identity":
                         {"hf_id": spec.hf_id, "revision": spec.revision},
                         "image_aware_release_hash": "images", "calibration_sha256":
                         digest(json.dumps(calibration, sort_keys=True, separators=(",", ":")).encode()),
                         "eligibility_decision": "approved_for_this_stretch_protocol"})
        identity["calibration_binding"] = {key: identity[key] for key in
                                           ("source_identity", "checkpoint_identity", "image_aware_release_hash")}
        validate_runtime_context(session, compiled, calibration, identity, execute=True)
        session.release_manifest["image_aware_release_hash"] = "different-images"
        with self.assertRaises(ValueError):
            validate_runtime_context(session, compiled, calibration, identity, execute=True)
        session.release_manifest["image_aware_release_hash"] = "images"
        spec.revision = "main"
        with self.assertRaises(ValueError):
            validate_runtime_context(session, compiled, calibration, identity, execute=True)


if __name__ == "__main__":
    unittest.main()
