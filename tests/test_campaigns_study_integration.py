"""Authored phase/freeze integration fixtures. Not executed during source work.

These exercise the real campaign lifecycle and lazy registry with controlled
study implementations. Numerical operation tests live beside each evaluator;
this fixture specifically guards phase, preparation and freeze handoff regressions.
"""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from vlm_mcqa.campaigns import runner
from vlm_mcqa.stretch.studies import STUDIES


class LifecycleTests(unittest.TestCase):
    def test_verified_retry_confirmation_retains_discovery_source_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "discovery").mkdir()
            frozen_identity = {"model_key": "fixture", "dataset_hash": "dataset",
                               "image_aware_release_hash": "images", "source_identity": "old-source",
                               "checkpoint_identity": {"hf_id": "fixture/model", "revision": "a" * 40}}
            (output / "discovery/protocol_freeze.json").write_text(json.dumps({
                "schema": "stretch.protocol-freeze.v1", "study": "semantic_heads",
                "identity": frozen_identity, "protocol": {"selected": True}}))
            (output / "discovery_result.json").write_text(json.dumps({"status": "complete"}))
            (output / "study_freeze.json").write_text(json.dumps({"decisions": {}}))
            adapter = SimpleNamespace(spec=SimpleNamespace(key="fixture"))
            class Session:
                def __init__(self, adapter, data, out, profile, *, seed):
                    self.adapter, self.data_dir, self.out_dir = adapter, Path(data), Path(out)
                    self.profile, self.seed = profile, seed
                    self.out_dir.mkdir(exist_ok=True)
            session = Session(adapter, output / "inputs", output / "retry_session", {}, seed=7)
            config = {"seed": 7, "retry_from": "prior", "model": {"key": "fixture"},
                      "study": {"name": "semantic_heads", "phases": ["discovery", "confirmation"],
                                "config": {"protocol": {"prespecified": True}}}}
            proof = {"dataset_hash": "dataset", "image_aware_release_hash": "images",
                     "source": {"manifest_sha256": "new-source"},
                     "model_spec": {"hf_id": "fixture/model", "revision": "a" * 40},
                     "protocol_decision_sha256": "decision"}
            captured = {}
            def prepare(current, selected, context):
                captured.update(context["identity"])
                return dict(selected), {"status": "prepared"}
            def evaluate(current, selected, context):
                target = Path(context["output_dir"])
                target.mkdir()
                return {"status": "complete", "scientific_verdict": "negative_fixture"}
            retry = {"verified": True}
            with patch("vlm_mcqa.campaigns.retry.plan_retry", return_value=retry), \
                 patch("vlm_mcqa.campaigns.retry.copy_stretch_discovery", return_value=True), \
                 patch("vlm_mcqa.campaigns.session.create_session", return_value=(session, proof)), \
                 patch.dict("sys.modules", {"vlm_mcqa.core7.session": SimpleNamespace(Session=Session)}), \
                 patch.object(runner, "_ordinary", return_value={"all_passed": True}), \
                 patch.object(runner, "verify_inputs", return_value=proof), \
                 patch("vlm_mcqa.stretch.studies.prepare_study", side_effect=prepare), \
                 patch("vlm_mcqa.stretch.studies.calibrate_study", return_value={"all_passed": True}), \
                 patch("vlm_mcqa.stretch.studies.run_study", side_effect=evaluate):
                result = runner.run_stretch(config, proof, output, execute=True)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(captured["source_identity"], "old-source")
            self.assertEqual(captured["retry_execution_source_identity"], "new-source")

    def test_technical_pilot_runs_one_bounded_phase_and_denies_scientific_use(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "pilot"
            protocol = {"sampling": {"discovery": {"target_pairs": 8, "min_pairs": 8}},
                        "claim_boundary": "calibration_and_technical_pilots_are_not_scientific_evidence"}
            config = {"kind": "stretch", "output_dir": str(output), "model": {"key": "fixture"},
                      "study": {"name": "semantic_heads", "phases": ["discovery", "confirmation"],
                                "config": {"protocol": protocol}}}
            proof = {"dataset_hash": "data", "image_aware_release_hash": "images",
                     "source": {"manifest_sha256": "source"}, "protocol_decision_sha256": "decision",
                     "model_spec": {"hf_id": "fixture/model", "revision": "a" * 40}}
            session = SimpleNamespace(adapter=SimpleNamespace(spec=SimpleNamespace(key="fixture")))
            prepared = {"protocol": protocol, "phase": "discovery", "split": "discovery"}
            measured = {"status": "complete", "coverage": {"semantic_pairs": 8},
                        "scientific_verdict": "must_not_escape_pilot"}
            with patch.object(runner, "validate_config"), patch.object(runner, "verify_inputs", return_value=proof), \
                 patch.object(runner, "_protected", return_value=[]), \
                 patch("vlm_mcqa.campaigns.session.create_session", return_value=(session, proof)), \
                 patch.object(runner, "_ordinary", return_value={"all_passed": True}), \
                 patch("vlm_mcqa.stretch.studies.prepare_study", return_value=(prepared, {"status": "ready"})), \
                 patch("vlm_mcqa.stretch.studies.calibrate_study", return_value={"all_passed": True}), \
                 patch("vlm_mcqa.stretch.studies.run_study", return_value=measured):
                result = runner.technical_pilot_campaign(config, execute=True)
            self.assertEqual(result["status"], "technical_complete")
            self.assertEqual(result["coverage"]["semantic_pairs"], 8)
            self.assertFalse(result["scientific_run_performed"])
            self.assertFalse(result["confirmation_eligible"])
            self.assertFalse((output / "result.json").exists())

    def test_calibration_only_carries_study_decision_and_never_runs_science(self):
        from vlm_mcqa.stretch.task_runtime import execution_identity
        from vlm_mcqa.stretch import external
        with tempfile.TemporaryDirectory() as temp:
            config = {"kind": "stretch", "output_dir": str(Path(temp) / "attempt"),
                      "model": {"key": "fixture"},
                      "study": {"name": "external", "config": {"protocol": {"fixture": True}}}}
            proof = {"dataset_hash": "data", "image_aware_release_hash": "images",
                     "source": {"manifest_sha256": "source"}, "protocol_decision_sha256": "decision",
                     "model_spec": {"hf_id": "fixture/model", "revision": "a" * 40}}
            session = SimpleNamespace(adapter=SimpleNamespace(spec=SimpleNamespace(
                key="fixture", hf_id="fixture/model", revision="a" * 40)))
            def measured_calibration(current, selected, context):
                self.assertEqual(context["identity"]["protocol_decision_sha256"], "decision")
                self.assertEqual(context["identity"]["eligibility_decision"], "approved_for_this_stretch_protocol")
                self.assertTrue(execution_identity(current, context))
                return {"all_passed": True, "checks": {"fixture": {"passed": True, "measured": 0, "threshold": 0}}}
            with patch.object(runner, "validate_config"), patch.object(runner, "verify_inputs", return_value=proof), \
                 patch.object(runner, "_protected", return_value=[]), \
                 patch("vlm_mcqa.campaigns.session.create_session", return_value=(session, proof)), \
                 patch.object(runner, "_ordinary", return_value={"all_passed": True}), \
                 patch.object(runner, "audit_attempt", return_value={"technical_complete": True}), \
                 patch.object(external, "calibrate_study", side_effect=measured_calibration), \
                 patch.object(external, "run_study", side_effect=AssertionError("science must not run")):
                report = runner.calibrate_campaign(config, execute=True)
            self.assertEqual(report["status"], "calibrated")
            self.assertFalse(report["scientific_run_performed"])
            self.assertFalse(report["full_campaign_complete"])

    def test_each_study_gets_its_actual_discovery_freeze(self):
        for study in STUDIES:
            with self.subTest(study=study), tempfile.TemporaryDirectory() as temp:
                output = Path(temp)
                adapter = SimpleNamespace(spec=SimpleNamespace(key="fixture"))
                class Session:
                    def __init__(self, adapter, data, out, profile, *, seed):
                        self.adapter, self.data_dir, self.out_dir = adapter, Path(data), Path(out)
                        self.profile, self.seed = profile, seed
                        self.out_dir.mkdir()
                session = Session(adapter, output / "inputs", output / "first_session", {}, seed=7)
                config = {"seed": 7, "model": {"key": "fixture"}, "study": {
                    "name": study, "phases": ["discovery", "confirmation"], "config": {
                        "protocol": {"prespecified": True}, "phase_settings": {
                            "discovery": {"unit_ids": ["discovery-unit"]},
                            "confirmation": {"unit_ids": ["confirmation-unit"]}}}}}
                proof = {"dataset_hash": "dataset", "image_aware_release_hash": "images",
                         "source": {"manifest_sha256": "source"},
                         "model_spec": {"hf_id": "fixture/model", "revision": "a" * 40},
                         "protocol_decision_sha256": "decision"}
                history = []
                core = {"all_passed": True}
                module = importlib.import_module("vlm_mcqa.stretch." + STUDIES[study]["module"])
                def prepare(current, selected, context):
                    return dict(selected), {"status": "prepared"}
                def calibrate(current, selected, context):
                    history.append(("calibrate", selected["phase"]))
                    return {"all_passed": True, "checks": {
                        "analytic_fixture": {"passed": True, "measured": 0, "threshold": 0}}}
                def evaluate(current, selected, context):
                    phase = selected["phase"]
                    history.append(("run", phase))
                    self.assertEqual(selected["split"], phase)
                    self.assertEqual(selected["unit_ids"], [phase + "-unit"])
                    target = Path(context["output_dir"])
                    target.mkdir()
                    if phase == "discovery":
                        name = "protocol_freeze.json" if study in {"semantic_heads", "route", "failures"} else "freeze.json"
                        freeze = {"fixture": study, "selected": "discovery-only"}
                        if study in {"spatial", "image_options"}:
                            freeze["confirmation_protocol"] = {"prespecified": True, "selected_layer": 8}
                        (target / name).write_text(json.dumps(freeze))
                    else:
                        key = "frozen_protocol" if study in {"semantic_heads", "route", "failures"} else "freeze_path"
                        frozen = json.loads(Path(selected[key]).read_text())
                        self.assertEqual(frozen["fixture"], study)
                        self.assertEqual(frozen["selected"], "discovery-only")
                        if study in {"spatial", "image_options"}:
                            self.assertEqual(selected["protocol"], {"prespecified": True, "selected_layer": 8})
                    return {"schema": "fixture/v1", "study": study, "status": "complete",
                            "coverage": {"rows": 1}, "scientific_verdict": "negative_fixture"}
                with patch("vlm_mcqa.campaigns.session.create_session", return_value=(session, proof)), \
                     patch.dict("sys.modules", {"vlm_mcqa.core7.session": SimpleNamespace(Session=Session)}), \
                     patch.object(runner, "_ordinary", return_value=core), \
                     patch.object(runner, "verify_inputs", return_value=proof), \
                     patch.object(module, "calibrate_study", side_effect=calibrate), \
                     patch.object(module, "run_study", side_effect=evaluate), \
                     patch.object(module, "prepare_study", side_effect=prepare, create=True):
                    result = runner.run_stretch(config, proof, output, execute=True)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["scientific_verdict"], "negative_fixture")
                self.assertEqual(history, [("calibrate", "discovery"), ("run", "discovery"),
                                           ("calibrate", "confirmation"), ("run", "confirmation")])


if __name__ == "__main__":
    unittest.main()
