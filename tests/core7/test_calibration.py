"""Scale-aware tolerance for cross-kernel calibration gates (core7.calibration)."""

from __future__ import annotations

import unittest

from support import HAS_TORCH, requires_torch

if HAS_TORCH:
    import torch

    from vlm_mcqa.core7.calibration import _kernel_tolerance, _logit_error


@requires_torch
class KernelToleranceTests(unittest.TestCase):
    def test_nonfinite_logits_fail_closed(self) -> None:
        self.assertEqual(_logit_error(torch.tensor([float("nan")]), torch.zeros(1)), float("inf"))
        self.assertEqual(_logit_error(torch.tensor([float("inf")]), torch.zeros(1)), float("inf"))

    def test_float32_tolerance_is_fixed(self) -> None:
        self.assertEqual(_kernel_tolerance(2e-3, "float32", torch.tensor([100.0])), 2e-3)

    def test_bfloat16_small_logits_use_base_tolerance(self) -> None:
        # 8 * 2^-8 * 8 = 0.25: below |logit| 8 the base tolerance applies
        self.assertEqual(_kernel_tolerance(0.25, "bfloat16", torch.tensor([-5.0, 8.0])), 0.25)

    def test_bfloat16_tolerance_scales_with_peak_logit(self) -> None:
        # eight units of bf16 roundoff (2^-8) at the largest |logit|
        self.assertEqual(_kernel_tolerance(0.25, "bfloat16", torch.tensor([3.0, -32.0])), 1.0)
        self.assertAlmostEqual(_kernel_tolerance(0.25, "bfloat16", torch.tensor([54.75])), 1.7109375)

    def test_bfloat16_rounding_passes_and_real_mismatch_fails(self) -> None:
        ref = torch.linspace(-55, 55, 1000).to(torch.bfloat16)
        rounded = (ref.float() + 1.0).to(torch.bfloat16)  # the largest noise measured (InternVL3.5-2B)
        limit = _kernel_tolerance(0.25, "bfloat16", ref)
        self.assertLessEqual(float((rounded.float() - ref.float()).abs().max()), limit)
        self.assertGreater(2.0, limit)  # a 2-logit batching bug would still fail


if __name__ == "__main__":
    unittest.main()
