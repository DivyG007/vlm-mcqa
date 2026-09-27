"""Fail-closed launch gates that do not load a model (core7.cli)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from vlm_mcqa.core7.cli import _enforce_device_placement


class _Adapter:
    def __init__(self, device_map: dict[str, str] | None) -> None:
        self.device_map = device_map

    def describe(self) -> dict:
        return {"hf_device_map": self.device_map}


class DevicePlacementGateTests(unittest.TestCase):
    def test_gate_is_opt_in(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            _enforce_device_placement(_Adapter(None))

    def test_two_cuda_devices_pass(self) -> None:
        with patch.dict(os.environ, {"CORE7_REQUIRE_MULTI_GPU": "1"}, clear=True):
            _enforce_device_placement(_Adapter({"vision": "0", "layer.0": "cuda:1"}))

    def test_single_device_or_cpu_offload_fails(self) -> None:
        with patch.dict(os.environ, {"CORE7_REQUIRE_MULTI_GPU": "1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "multi-GPU required"):
                _enforce_device_placement(_Adapter({"": "0"}))
            with self.assertRaisesRegex(RuntimeError, "offload"):
                _enforce_device_placement(_Adapter({"vision": "0", "layer.0": "cpu"}))


if __name__ == "__main__":
    unittest.main()
