"""Source-authored integration specifications. Not executed during implementation."""
import unittest

from vlm_mcqa.stretch.statistics import cluster_estimate
from vlm_mcqa.stretch.studies import calibrate_study, run_study, study_spec


class StudyDispatchTests(unittest.TestCase):
    def test_no_execution_before_implementation_import(self):
        for name in ("semantic_heads", "route", "failures", "spatial", "image_options", "external"):
            for function in (run_study, calibrate_study):
                with self.assertRaises(PermissionError):
                    function(None, {"name": name}, {"execute": False})

    def test_unknown_study(self):
        with self.assertRaises(ValueError):
            study_spec("invented")

    def test_pair_weight_not_direction_weight(self):
        rows = [{"pair": "a", "x": 0.0}] * 7 + [{"pair": "b", "x": 1.0}]
        result = cluster_estimate(rows, cluster="pair", value="x", seed=9, draws=100)
        self.assertEqual(result["mean"], .5)
        self.assertEqual(result["n_clusters"], 2)

    def test_nonfinite_not_zero(self):
        rows = [{"scene": "a", "x": None}, {"scene": "b", "x": 1.0}]
        result = cluster_estimate(rows, cluster="scene", value="x", seed=9, draws=100)
        self.assertEqual(result["mean"], 1.0)
        self.assertIsNone(result["ci95"])
        self.assertEqual(result["excluded"]["missing_or_nonfinite_value"], 1)


if __name__ == "__main__":
    unittest.main()
