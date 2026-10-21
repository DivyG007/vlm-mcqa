"""Future negative provenance fixtures; authored and deliberately unexecuted."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from vlm_mcqa.campaigns.preparation import prepare_inventory
from vlm_mcqa.campaigns.verification import file_hash, verify_inventory
from vlm_mcqa.campaigns.retry import (
    _confirmation_sample_sensitivity_compatible,
    _placement_retry_compatible,
    _retry_compatible,
    _stretch_signature,
    copy_stretch_confirmation_preparation,
    copy_stretch_discovery,
    scientific_config,
)


class ProvenanceTests(unittest.TestCase):
    def test_exact_inventory_detects_mutation_and_missing_required_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = root / "weights.safetensors"
            payload.write_bytes(b"fixture")
            manifest = root / "inventory.json"
            manifest.write_text(json.dumps({"files": {payload.name: file_hash(payload)}}))
            verify_inventory(root, manifest, required={payload.name})
            with self.assertRaises(ValueError):
                verify_inventory(root, manifest, required={"missing.json"})
            payload.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_inventory(root, manifest)

    def test_unsafe_inventory_path_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "inventory.json"
            for relative in ("../outside", "C:/outside", "a\\b"):
                path.write_text(json.dumps({"files": {relative: "a" * 64}}))
                with self.assertRaises(ValueError):
                    verify_inventory(root, path)

    def test_checkpoint_preparation_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}")
            manifest = root / "manifest.json"
            prepare_inventory(checkpoint, manifest, kind="checkpoint", hf_id="fixture/model", revision="a" * 40)
            with self.assertRaises(ValueError):
                prepare_inventory(checkpoint, manifest, kind="checkpoint", hf_id="fixture/model", revision="a" * 40)

    def test_retry_configuration_ignores_only_attempt_fields(self):
        a = {"output_dir": "first", "retry_from": "old", "seed": 1, "model": {"key": "m"}}
        b = {"output_dir": "second", "seed": 1, "model": {"key": "m"}}
        self.assertEqual(scientific_config(a), scientific_config(b))
        self.assertNotEqual(scientific_config(a), scientific_config({**b, "seed": 2}))

    def test_stretch_retry_allows_only_cache_capacity_change_and_copies_frozen_discovery(self):
        base = {"kind": "stretch", "output_dir": "old", "seed": 0,
                "deployment": {"device_map": {"layer": "cuda:0"}, "prepared_cache": 8},
                "study": {"name": "semantic_heads"}}
        retry = {**base, "output_dir": "new", "retry_from": "old",
                 "deployment": {**base["deployment"], "prepared_cache": 2}}
        self.assertTrue(_retry_compatible(base, retry))
        self.assertFalse(_retry_compatible(base, {**retry, "seed": 1}))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous, output = root / "previous", root / "output"
            previous.mkdir(); output.mkdir()
            (previous / "discovery").mkdir()
            (previous / "discovery/raw.jsonl").write_text("{}\n")
            (previous / "discovery_result.json").write_text('{"status":"complete"}')
            (previous / "study_freeze.json").write_text('{"freeze_sha256":"x"}')
            record = {"reusable": [{"phase": "stretch_discovery", "source": str(previous),
                                     "files": {"discovery/raw.jsonl": {},
                                               "discovery_result.json": {}, "study_freeze.json": {}}}]}
            self.assertTrue(copy_stretch_discovery(record, output))
            self.assertEqual((output / "discovery/raw.jsonl").read_text(), "{}\n")

    def test_stretch_retry_signature_allows_only_path_and_cache_changes(self):
        def fixture(source, checkpoint, cache):
            recipe = f"{source}/configs/recipe.json"
            config = {
                "schema": "vlm-mcqa-campaign/v1", "kind": "stretch", "seed": 0,
                "source": {"root": source},
                "model": {"key": "internvl", "checkpoint_dir": checkpoint},
                "deployment": {"prepared_cache": cache, "dtype": "float16",
                               "device_map": {"layer.0": "cuda:0"}},
                "session_profile": {"calibration_batch_size": 2},
                "resolved_documents": [{"path": recipe, "sha256": "a" * 64}],
                "study": {"name": "e20", "config": {"recipe": recipe,
                          "layers": [0, 12, 24]}},
            }
            proof = {
                "model_spec": {"hf_id": "org/model", "revision": "b" * 40,
                               "family": "internvl", "dtype": "bfloat16"},
                "dataset_hash": "dataset", "image_aware_release_hash": "release",
                "protocol_decision_sha256": "c" * 64,
                "study_input_file_sha256": {recipe: "a" * 64},
            }
            return config, proof

        old_config, old_proof = fixture("/snapshots/old", "/ssd/checkpoint", 8)
        new_config, new_proof = fixture("/snapshots/new", "/scratch/checkpoint", 2)
        self.assertEqual(_stretch_signature(old_config, old_proof),
                         _stretch_signature(new_config, new_proof))
        changed = copy.deepcopy(new_config)
        changed["study"]["config"]["layers"] = [0, 16, 32]
        self.assertNotEqual(_stretch_signature(old_config, old_proof),
                            _stretch_signature(changed, new_proof))

    def test_stretch_retry_copies_only_completed_confirmation_preparation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous, output = root / "previous", root / "output"
            preparation = previous / "confirmation_preparation"
            preparation.mkdir(parents=True); output.mkdir()
            (preparation / "baseline.jsonl").write_text("{}\n")
            (preparation / "sample_plan.json").write_text('{"status":"ready"}')
            record = {"reusable": [{"phase": "stretch_confirmation_preparation",
                                     "source": str(previous),
                                     "files": {"confirmation_preparation/baseline.jsonl": {},
                                               "confirmation_preparation/sample_plan.json": {}}}]}
            self.assertTrue(copy_stretch_confirmation_preparation(record, output))
            self.assertEqual((output / "confirmation_preparation/baseline.jsonl").read_text(), "{}\n")

    def test_stretch_retry_allows_only_same_module_gpu_remapping(self):
        old = {"deployment": {"device_map": {"layer.0": "cuda:0", "layer.1": "cuda:1"}}}
        moved = {"deployment": {"device_map": {"layer.0": "cuda:1", "layer.1": "cuda:1"}}}
        self.assertTrue(_placement_retry_compatible(old, moved))
        self.assertFalse(_placement_retry_compatible(old, {
            "deployment": {"device_map": {"layer.0": "cpu", "layer.1": "cuda:1"}}}))
        self.assertFalse(_placement_retry_compatible(old, {
            "deployment": {"device_map": {"layer.0": "cuda:1"}}}))
        self.assertTrue(_placement_retry_compatible(old, {
            "deployment": {"device_map": {"layer.0": "cuda:0", "layer.1": "cuda:2"}}}))

    def test_e20_confirmation_sensitivity_requires_exact_observed_shortfall(self):
        with tempfile.TemporaryDirectory() as temp:
            previous = Path(temp) / "primary"
            previous.mkdir()
            (previous / "confirmation_result.json").write_text(json.dumps({
                "scientific_verdict": "insufficient_baseline_eligible_sample",
                "coverage": {"coverage": {"control_complete_pairs": 14,
                                            "selected_pairs": 14}},
            }))
            (previous / "completion.json").write_text(json.dumps({"status": "insufficient"}))
            recipe = "/snapshots/source/configs/recipe.json"
            old = {
                "schema": "vlm-mcqa-campaign/v1", "kind": "stretch", "seed": 0,
                "source": {"root": "/snapshots/source"},
                "model": {"key": "llava", "checkpoint_dir": "/scratch/checkpoint"},
                "deployment": {"dtype": "float16", "device_map": {"layer": "cuda:0"}},
                "session_profile": {},
                "resolved_documents": [{"path": "/launch/old/effective_protocol.json",
                                          "sha256": "d" * 64}],
                "study": {"name": "semantic_heads", "config": {"recipe": recipe}},
            }
            proof = {
                "model_spec": {"hf_id": "org/model", "revision": "b" * 40},
                "dataset_hash": "dataset", "image_aware_release_hash": "release",
                "protocol_decision_sha256": "c" * 64,
                "study_input_file_sha256": {recipe: "a" * 64},
            }
            new = copy.deepcopy(old)
            new["resolved_documents"][0]["path"] = "/launch/new/effective_protocol.json"
            new["study"]["config"]["confirmation_sample_sensitivity"] = {
                "target_pairs": 14, "min_pairs": 14,
                "primary_attempt": str(previous),
                "label": "reduced_heldout_sample_sensitivity",
                "reason": "use every frozen control-complete pair",
            }
            self.assertTrue(_confirmation_sample_sensitivity_compatible(
                previous, old, new, proof, proof))
            new["study"]["config"]["confirmation_sample_sensitivity"]["target_pairs"] = 13
            self.assertFalse(_confirmation_sample_sensitivity_compatible(
                previous, old, new, proof, proof))


if __name__ == "__main__":
    unittest.main()
