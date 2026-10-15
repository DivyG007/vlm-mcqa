"""Audit regressions: pair weighting, valid negative results and tamper rejection."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('llava_audit', ROOT / 'scripts/local/audit_llava_completed.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
FIXTURE = ROOT / 'results/pkgpu_complete/llava_audit_20261019/20261016_124107'


class ResultAuditTests(unittest.TestCase):
    def test_pair_mean_does_not_weight_pairs_by_directed_contrast_count(self):
        rows = [dict(pair_id='a', normalized_recovery=1), dict(pair_id='a', normalized_recovery=1),
                dict(pair_id='b', normalized_recovery=0)]
        self.assertEqual(audit.pair_means(rows), .5)

    @unittest.skipUnless(FIXTURE.exists(), 'requires ignored raw result mirror')
    def test_negative_order_replication_is_valid_completed_execution(self):
        result = audit.audit(FIXTURE)
        self.assertTrue(result['passed'])
        self.assertTrue(result['checks']['execution_complete'])
        self.assertFalse(result['checks']['all_hypotheses_replicated'])
        self.assertFalse(result['checks']['content_vs_position_order']['replicated'])

    @unittest.skipUnless(FIXTURE.exists(), 'requires ignored raw result mirror')
    def test_raw_checksum_mismatch_rejected_before_accepting_results(self):
        with patch.object(audit, 'sha', return_value='corrupt'):
            with self.assertRaises(AssertionError):
                audit.audit(FIXTURE)


if __name__ == '__main__':
    unittest.main()
