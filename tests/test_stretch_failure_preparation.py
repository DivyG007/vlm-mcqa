"""Focused prospective E22 preparation fixtures; authored but NOT executed."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from vlm_mcqa.stretch import failure_runtime
from vlm_mcqa.stretch.extension_calibration import sha
from vlm_mcqa.stretch.failure_protocol import freeze_failure_rules


FLAG = "truth_aligned_content_with_wrong_final_symbol"


def frozen_rules():
    return freeze_failure_rules({"stage": 3, "strata_flags": [FLAG], "min_scenes": 2,
        "reliability_min": .8, "seed": 5, "bootstrap_draws": 100,
        "margin_bins": [0.0], "match_fields": ["question_family", "scheme"], "max_control_reuse": 1})


def descriptive_fixture(phase="confirmation"):
    rows = []
    for name, correct in (("failure-a", False), ("failure-b", False), ("correct-a", True), ("correct-b", True)):
        rows.append({"variant_id": name, "item_id": name, "scene_id": name, "split": phase,
                     "stage": 3, "final_correct": correct, "content_proxy_correct": True,
                     "question_family": "colour", "scheme": "letters", "confidence_gap": 2.0,
                     "descriptive_flags": [] if correct else [FLAG], "unresolved": False})
    return {"status": "descriptive_not_causal", "rows": rows}, {"items": [{"item_id": row["item_id"]} for row in rows]}


class SelectionTests(unittest.TestCase):
    def test_preparation_denied_before_source_or_session_access(self):
        with self.assertRaises(PermissionError):
            failure_runtime.prepare_study(object(), {}, {"execute": False})

    def test_confirmation_uses_one_real_phase_without_internal_resplit(self):
        report, dataset = descriptive_fixture()
        cases, selection = failure_runtime.select_phase_cases(report, dataset, frozen_rules(), phase="confirmation")
        self.assertEqual(selection["status"], "ready")
        self.assertEqual(selection["candidate_stage_rows"], 4)
        self.assertEqual(len(cases), 4)
        self.assertEqual({case["source_phase"] for case in cases}, {"confirmation"})
        self.assertEqual(len({case["recipient_variant_id"] for case in cases if case["role"] == "matched_correct"}), 2)

    def test_no_matching_frozen_flag_produces_insufficiency(self):
        report, dataset = descriptive_fixture()
        for row in report["rows"]:
            row["descriptive_flags"] = []
        cases, selection = failure_runtime.select_phase_cases(report, dataset, frozen_rules(), phase="confirmation")
        self.assertEqual(cases, [])
        self.assertEqual(selection["status"], "insufficient")

    def test_unresolved_content_does_not_become_guessed_failure_stratum(self):
        report, dataset = descriptive_fixture()
        for row in report["rows"]:
            if not row["final_correct"]:
                row["unresolved"] = True
        cases, selection = failure_runtime.select_phase_cases(report, dataset, frozen_rules(), phase="confirmation")
        self.assertEqual(cases, [])
        self.assertEqual(selection["unresolved_rows"], 2)
        self.assertEqual(selection["status"], "insufficient")

    def test_duplicate_or_wrong_phase_baselines_rejected(self):
        report, dataset = descriptive_fixture()
        with self.assertRaises(ValueError):
            failure_runtime.select_phase_cases(report, dataset, frozen_rules(), phase="discovery")
        report["rows"].append(copy.deepcopy(report["rows"][0]))
        with self.assertRaises(ValueError):
            failure_runtime.select_phase_cases(report, dataset, frozen_rules(), phase="confirmation")

    def test_existing_e19_report_can_supply_complete_frozen_rules(self):
        rules = frozen_rules()
        resolved = failure_runtime._prospective_rules({"strata_rules_path": {"schema": "stretch.failure-strata.v1", "rules": rules},
            "strata_rules_sha256": rules["rules_sha256"], "allowed_strata": [FLAG],
            "min_stratum_scenes": 2, "tie_tolerance": 0.0})
        self.assertEqual(resolved, rules)


class PreparationTests(unittest.TestCase):
    def test_actual_preparation_saves_baselines_lens_and_bound_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs, session_output = root / "inputs", root / "session"
            inputs.mkdir()
            session_output.mkdir()
            rules = frozen_rules()
            rules_path, scope_path = inputs / "rules.json", inputs / "scope.json"
            rules_path.write_text(json.dumps(rules), encoding="utf-8")
            variants = {}
            for name, correct in (("failure-a", False), ("failure-b", False), ("correct-a", True), ("correct-b", True)):
                variants[name] = {"variant_id": name, "item_id": name, "scene_id": name,
                    "counterfactual_pair_id": None, "split": "discovery", "label_scheme": "letters",
                    "labels": list("ABCD"), "correct_index": 0, "correct_symbol": "A",
                    "option_contents": ["red", "blue", "green", "yellow"], "semantic_answer": "red",
                    "question_family": "colour", "_correct": correct}
            scope_path.write_text(json.dumps({"schema": "stretch.failure-candidate-scope.v1",
                "phase": "discovery", "variant_ids": list(variants)}), encoding="utf-8")
            body = {"strata_rules_path": str(rules_path), "strata_rules_sha256": rules["rules_sha256"],
                    "allowed_strata": [FLAG], "min_stratum_scenes": 2, "tie_tolerance": 0.0,
                    "discovery_allowed_splits": ["discovery"]}
            class FourLogits:
                def float(self):
                    return self
                def cpu(self):
                    return self
                def __getitem__(self, index):
                    return [4.0, 0.0, 0.0, 0.0][index]
            clean_calls = []
            def clean(vid):
                clean_calls.append(vid)
                correct = variants[vid]["_correct"]
                return SimpleNamespace(sites={"resid_post:2": object()}, readout={
                    "correct": correct, "predicted_index": 0 if correct else 1,
                    "label_logits": [2., 0., -1., -2.] if correct else [0., 2., -1., -2.],
                    "global_top_is_label": True})
            session = SimpleNamespace(data_dir=inputs, out_dir=session_output, variants=variants,
                items={vid: {"item_id": vid, "program_depth": 1} for vid in variants}, clean=clean,
                prepared=lambda vid: SimpleNamespace(content_ids=dict(zip(variants[vid]["option_contents"], range(4)))),
                adapter=SimpleNamespace(num_layers=4, handle=SimpleNamespace(readout=lambda hidden: FourLogits())))
            config = {"name": "failures", "phase": "discovery", "candidate_scope_path": str(scope_path)}
            context = {"execute": True, "identity": {"model_key": "fixture", "source_identity": "source", "split": "discovery"},
                       "preparation_dir": root / "preparation"}
            with patch.object(failure_runtime, "core_guard"), patch.object(failure_runtime, "_body", return_value=(body, {})):
                prepared, report = failure_runtime.prepare_study(session, config, context)
            self.assertEqual(report["status"], "ready")
            self.assertEqual(set(clean_calls), set(variants))
            manifest = json.loads(Path(prepared["cases"]).read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["cases"]), 4)
            self.assertEqual(manifest["generation"]["identity_sha256"], sha(context["identity"]))
            failure_runtime.validate_prepared_cases(prepared, context, body)
            cases, excluded = failure_runtime.candidate_units(prepared, body)
            self.assertEqual(len(cases), 4)
            self.assertEqual(excluded, [])
            with self.assertRaises(ValueError):
                failure_runtime.validate_prepared_cases(prepared, {**context, "identity": {"model_key": "different"}}, body)
            with (root / "preparation/baseline.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaises(ValueError):
                failure_runtime.validate_prepared_cases(prepared, context, body)
            # Even consistently rehashed evidence cannot bypass a failed gate.
            selection_path = root / "preparation/selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["status"] = "insufficient"
            selection["strata"][FLAG]["reliability_gate_passed"] = False
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            for name in ("baseline.jsonl", "selection.json"):
                manifest["generation"]["artifacts"][name] = hashlib.sha256((root / "preparation" / name).read_bytes()).hexdigest()
            manifest["cases_sha256"] = sha({k: v for k, v in manifest.items() if k != "cases_sha256"})
            Path(prepared["cases"]).write_text(json.dumps(manifest), encoding="utf-8")
            prepared["_prepared_failure_cases_sha256"] = manifest["cases_sha256"]
            with self.assertRaisesRegex(ValueError, "insufficient prospective"):
                failure_runtime.validate_prepared_cases(prepared, context, body)

    def test_external_asserted_confirmation_roles_are_rejected(self):
        supplied = {"schema": "stretch.failure-cases.v1", "phase": "confirmation",
                    "rules_sha256": "a" * 64, "cases": []}
        with self.assertRaises(ValueError):
            failure_runtime.candidate_units({"phase": "confirmation", "cases": supplied},
                                            {"strata_rules_sha256": "a" * 64})

    def test_wrong_answer_calibration_requires_actually_wrong_recipient(self):
        session = SimpleNamespace(variants={"r": {"split": "discovery"}, "d": {"split": "discovery"}},
                                  clean=lambda vid: SimpleNamespace(correct=True))
        body = {"sites": {"image": {"site": "resid_post", "layer": 0},
                          "options": {"site": "resid_post", "layer": 1},
                          "final": {"site": "resid_post", "layer": 2}},
                "allowed_strata": [FLAG], "donor_classes": ["same_image_remap"],
                "image_repair_enabled": False}
        config = {"calibration_case": {"recipient_variant_id": "r", "donor_variant_id": "d",
                                       "role": "failure", "stratum": FLAG, "donor_class": "same_image_remap"},
                  "calibration_variant_ids": ["r"], "tolerance": .01}
        with patch.object(failure_runtime, "core_guard"), patch.object(failure_runtime, "_body", return_value=(body, {})), \
             patch.object(failure_runtime, "primitive_checks", side_effect=AssertionError("not reached")):
            with self.assertRaisesRegex(ValueError, "actually incorrect"):
                failure_runtime.calibrate_study(session, config, {"execute": True})


if __name__ == "__main__":
    unittest.main()
