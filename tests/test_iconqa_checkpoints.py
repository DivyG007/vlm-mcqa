import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('checkpoint',Path(__file__).parents[1]/'scripts/ada/iconqa_checkpoints.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).parents[1]/'results/test_tmp';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.temp.name)
        self.file=self.root/'rows.jsonl';self.rng=random.Random(7)
        self.blocks=module.Blocks(self.root/'markers',{'source':'a'},self.rng)
    def tearDown(self): self.temp.cleanup()
    def action(self):
        self.file.write_text('valid\n');return self.rng.random()
    def files(self):return [self.file]
    def test_resume_skips_and_restores_rng(self):
        result=self.blocks.run('method',self.action,self.files);expected=self.rng.random()
        self.rng.seed(999)
        self.assertEqual(self.blocks.run('method',lambda:1/0,self.files),result)
        self.assertEqual(self.rng.random(),expected)
    def test_modified_output_rejected(self):
        self.blocks.run('method',self.action,self.files);self.file.write_text('changed')
        with self.assertRaisesRegex(ValueError,'artifact changed'):self.blocks.run('method',self.action,self.files)
    def test_changed_identity_rejected(self):
        self.blocks.run('method',self.action,self.files)
        other=module.Blocks(self.root/'markers',{'source':'b'},self.rng)
        with self.assertRaisesRegex(ValueError,'identity mismatch'):other.run('method',self.action,self.files)
    def test_partial_preserved_before_rerun(self):
        self.file.write_text('partial')
        self.blocks.run('method',self.action,self.files)
        self.assertEqual(next((self.root/'markers').glob('partial_*/rows.jsonl')).read_text(),'partial')
    def test_failed_action_never_marked_complete(self):
        with self.assertRaises(ZeroDivisionError):self.blocks.run('method',lambda:1/0,self.files)
        self.assertFalse((self.root/'markers/method.json').exists())
