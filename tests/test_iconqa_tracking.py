"""Focused W&B wiring checks without network or GPU work."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from vlm_mcqa.core7.tracking import Tracker


class IconqaTrackingTest(unittest.TestCase):
    def test_stable_identity_and_resume_reach_sdk_and_receipt(self):
        run = MagicMock(id='stable123', name='attempt', url='https://wandb.ai/test/run')
        run.name = 'attempt'
        run.settings.mode = 'online'
        sdk = types.SimpleNamespace(init=MagicMock(return_value=run))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'wandb': sdk}):
            tracker = Tracker.start(enabled=True, out_dir=Path(tmp), name='attempt',
                                    job_type='iconqa-core7', config={'batch_size': 1},
                                    run_id='stable123', resume='allow')
            kwargs = sdk.init.call_args.kwargs
            self.assertEqual((kwargs['id'], kwargs['resume']), ('stable123', 'allow'))
            receipt = json.loads((Path(tmp)/'wandb_iconqa-core7.json').read_text())
            self.assertEqual(receipt['mode'], 'online')
            self.assertEqual(receipt['id'], 'stable123')
            tracker.log({'accuracy': 1.0}, 'discovery/behavior')
            run.log.assert_called_with({'discovery/behavior/accuracy': 1.0})
            tracker.finish('failed')
            run.finish.assert_called_once_with(exit_code=1)

    def test_existing_callers_do_not_get_resume_options(self):
        run = MagicMock(id='new', name='attempt', url='https://wandb.ai/test/run')
        run.name = 'attempt'
        run.settings.mode = 'offline'
        sdk = types.SimpleNamespace(init=MagicMock(return_value=run))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'wandb': sdk}):
            Tracker.start(enabled=True, out_dir=Path(tmp), name='attempt',
                          job_type='test', config={})
            self.assertNotIn('id', sdk.init.call_args.kwargs)
            self.assertNotIn('resume', sdk.init.call_args.kwargs)


if __name__ == '__main__':
    unittest.main()
