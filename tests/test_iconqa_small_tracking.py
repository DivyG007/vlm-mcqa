"""Tracking faults must preserve a local W&B run and scientific outputs."""
import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/ada'))
from iconqa_qwen_small import start_tracker,LocalFirstTracker


class SmallTrackingTest(unittest.TestCase):
    def test_online_failure_uses_same_identity_offline(self):
        offline=SimpleNamespace(url=None)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,{'wandb':SimpleNamespace(finish=MagicMock())}), \
                patch('vlm_mcqa.core7.tracking.Tracker.start',side_effect=[RuntimeError('network unavailable'),offline]) as start:
            tracker=start_tracker(Path(tmp),{'batch_size':4},False)
            calls=start.call_args_list
            self.assertEqual([c.kwargs['mode'] for c in calls],['online','offline'])
            self.assertEqual(calls[0].kwargs['run_id'],calls[1].kwargs['run_id'])
            state=json.loads((Path(tmp)/'tracking_state.json').read_text())
            self.assertTrue(state['pending_sync']);self.assertIsNone(tracker.url)

    def test_metric_publication_failure_is_recorded_locally(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend=SimpleNamespace(log=MagicMock(side_effect=RuntimeError('publication failure')))
            tracker=LocalFirstTracker(backend,Path(tmp));tracker.log({'metric':1})
            error=json.loads((Path(tmp)/'tracking_errors.jsonl').read_text())
            self.assertEqual(error['operation'],'log')


if __name__=='__main__':unittest.main()
