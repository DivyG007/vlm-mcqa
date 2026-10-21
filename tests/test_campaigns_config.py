"""Authored only: no tests or campaign programs executed during implementation."""
import copy
import unittest

from vlm_mcqa.campaigns.config import plan_campaign, validate_config
from vlm_mcqa.campaigns.runner import run_campaign, calibrate_campaign, technical_pilot_campaign
from vlm_mcqa.campaigns.session import _align_qwen_rotary_inputs, create_session


def fixture():
    return {"schema": "vlm-mcqa-campaign/v1", "kind": "core7_preselected", "seed": 0,
            "output_dir": "new", "source": {"root": "source", "manifest": "source.json"},
            "model": {"key": "paligemma2_10b_mix", "models_config": "models.json", "checkpoint_dir": "weights",
                      "checkpoint_manifest": "checkpoint.json"},
            "data": {"directory": "data", "dataset_hash": "dataset", "image_aware_release_hash": "images"},
            "deployment": {"dtype": "bfloat16", "device_map": {"": "cuda:7"}, "offload_policy": "forbid", "batch_size": 1},
            "profile": "full_core7", "profiles_config": "profiles.json",
            "protocol": {"selection_mode": "preselected_protocol_extension", "decision_path": "decision.json", "decision_sha256": "a" * 64}}


class ConfigurationTests(unittest.TestCase):
    def test_qwen_rotary_boundary_moves_only_positions(self):
        class Tensor:
            def __init__(self, device, shape, dtype):
                self.device, self.shape, self.dtype = device, shape, dtype

            def to(self, device):
                return Tensor(device, self.shape, self.dtype)

        class Rotary:
            inv_freq = Tensor("cuda:1", (4,), "float32")

        hidden = Tensor("cuda:1", (1, 3, 8), "float16")
        positions = Tensor("cpu", (3, 1, 3), "int64")
        args, kwargs = _align_qwen_rotary_inputs(Rotary(), (hidden, positions), {})
        self.assertIs(args[0], hidden)
        self.assertEqual(args[1].device, "cuda:1")
        self.assertEqual(args[1].shape, positions.shape)
        self.assertEqual(args[1].dtype, positions.dtype)
        self.assertEqual(kwargs, {})

    def test_plan_is_host_independent_and_contains_freeze(self):
        config = fixture()
        before = copy.deepcopy(config)
        plan = plan_campaign(config)
        self.assertEqual(config, before)
        self.assertIsNone(plan["compute_server"])
        self.assertIn("freeze", plan["stages"])
        self.assertFalse(plan["runtime_validated"])

    def test_pali_route_cannot_repeat_llava_or_shrink_profile(self):
        for key, value in (("model", {**fixture()["model"], "key": "llava_ov_7b_si"}), ("profile", "smoke")):
            config = fixture()
            config[key] = value
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_reduced_pali_route_is_explicit_and_cannot_masquerade_as_full(self):
        config = fixture()
        config.update(kind="core7_correct_only_reduced", profile="pali_correct_only_reduced")
        config["protocol"]["selection_mode"] = "preselected_correct_only_sensitivity"
        validate_config(config)
        config["profile"] = "full_core7"
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_no_automatic_placement_or_offload(self):
        for override in ({"device_map": "auto"}, {"offload_policy": "allow"}, {"device_map": "cpu"},
                         {"device_map": {"": "disk"}}, {"device_map": {"": "auto"}}):
            config = fixture()
            config["deployment"].update(override)
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_runtime_denies_before_validation_or_model_import(self):
        with self.assertRaises(PermissionError):
            run_campaign({}, execute=False)
        with self.assertRaises(PermissionError):
            create_session({}, "unused", execute=False)
        with self.assertRaises(PermissionError):
            calibrate_campaign({}, execute=False)
        with self.assertRaises(PermissionError):
            technical_pilot_campaign({}, execute=False)


if __name__ == "__main__":
    unittest.main()
