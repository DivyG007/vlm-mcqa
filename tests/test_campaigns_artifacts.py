"""Future artifact audit fixtures; NOT EXECUTED."""
import json
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest

from vlm_mcqa.campaigns.audit import audit_attempt, inventory
from vlm_mcqa.campaigns.runner import capture_logs, write_new


class ArtifactTests(unittest.TestCase):
    def test_python_progress_and_failure_trace_are_durable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    with capture_logs(root):
                        print("fixture progress")
                        raise RuntimeError("fixture failure")
            self.assertIn("fixture progress", (root / "stdout.log").read_text())
            self.assertIn("fixture failure", (root / "stderr.log").read_text())

    def test_preserved_failure_prevents_complete_even_if_other_status_says_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_new(root / "failure.json", {"error": "post-run audit failed"})
            with self.assertRaises(ValueError):
                audit_attempt(root)

    def test_exclusive_writer_and_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_new(root / "attempt.json", {"config": {"kind": "stretch", "study": {"name": "external"}}})
            (root / "stdout.log").write_text("")
            (root / "stderr.log").write_text("")
            write_new(root / "completion.json", {"status": "complete"})
            for split in ("discovery", "confirmation"):
                (root / split).mkdir()
                (root / split / "raw.jsonl").write_text('{"operation":"clean","variant_id":"v"}\n')
                write_new(root / f"{split}_result.json", {"status": "complete", "study": "external",
                    "coverage": {"rows": 1, "target_units": 1, "observed_units": 1},
                    "scientific_verdict": "negative", "artifacts": {"raw": str(root / split / "raw.jsonl")}})
                write_new(root / f"{split}_extension_calibration.json", {"all_passed": True,
                    "checks": {"fixture_check": {"passed": True, "measured": 0, "threshold": 0}}})
            write_new(root / "artifact_manifest.json", {"files": inventory(root)})
            self.assertTrue(audit_attempt(root)["technical_complete"])
            with self.assertRaises(FileExistsError):
                write_new(root / "completion.json", {})
            (root / "new_unrecorded.json").write_text("{}")
            with self.assertRaises(ValueError):
                audit_attempt(root)

    def test_failed_completion_is_not_audited_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_new(root / "attempt.json", {"config": {"kind": "stretch"}})
            write_new(root / "completion.json", {"status": "insufficient"})
            with self.assertRaises(ValueError):
                audit_attempt(root)


if __name__ == "__main__":
    unittest.main()
