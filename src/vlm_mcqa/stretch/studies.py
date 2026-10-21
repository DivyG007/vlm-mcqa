"""Lazy built-in study dispatch. Importing/planning never loads a model.

Model and processor execution requires an explicitly authorized caller and is
denied before importing the study implementation. No user-supplied callbacks.
"""
from __future__ import annotations

import importlib
from typing import Any, Mapping


STUDIES = {
    "semantic_heads": {"module": "controls", "unit": "semantic_pair",
                       "capabilities": ["decoder_replay", "head_patch", "image_capture"]},
    "route": {"module": "trajectory", "unit": "semantic_pair",
              "capabilities": ["decoder_replay", "head_patch", "modified_trajectory_capture"]},
    "failures": {"module": "failure_runtime", "unit": "scene",
                 "capabilities": ["decoder_replay", "head_patch", "truth_scoring"]},
    "spatial": {"module": "task_runtime", "unit": "scene",
                "capabilities": ["processor_footprints", "selected_head_attention", "edge_intervention"]},
    "image_options": {"module": "task_runtime", "unit": "visual_problem",
                      "capabilities": ["processor_footprints", "visual_option_groups", "region_patch"]},
    "external": {"module": "external", "unit": "source_image_or_template",
                 "capabilities": ["behavior", "frozen_head_transfer"]},
}


def study_spec(name: str) -> dict[str, Any]:
    if name not in STUDIES:
        raise ValueError(f"unknown study {name!r}; choose one of {sorted(STUDIES)}")
    spec = STUDIES[name]
    return {"name": name, "module": spec["module"], "unit": spec["unit"],
            "capabilities": list(spec["capabilities"])}


def _implementation(config: Mapping[str, Any], context: Mapping[str, Any]):
    if context.get("execute") is not True:
        raise PermissionError("study execution requires explicit execute=True")
    name = config.get("name")
    if "study" in config and config["study"] != name:
        raise ValueError("conflicting study selectors")
    specification = study_spec(name)
    if not context.get("identity") or not context.get("output_dir"):
        raise ValueError("study requires current execution identity and fresh output directory")
    if context.get("core_calibration", {}).get("all_passed") is not True:
        raise ValueError("ordinary Core-7 calibration must pass first")
    return importlib.import_module(f".{specification['module']}", __package__)


def calibrate_study(session: Any, config: Mapping[str, Any],
                    context: Mapping[str, Any]) -> dict[str, Any]:
    implementation = _implementation(config, context)
    report = implementation.calibrate_study(session, dict(config), dict(context))
    if not isinstance(report, dict) or type(report.get("all_passed")) is not bool:
        raise ValueError("study calibration must return an explicit all_passed result")
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise ValueError("study calibration must include measured named checks")
    if any(not isinstance(check, dict) or type(check.get("passed")) is not bool
           or "measured" not in check or "threshold" not in check for check in checks.values()):
        raise ValueError("calibration checks require measured, threshold and passed fields")
    if report["all_passed"] != all(check["passed"] for check in checks.values()):
        raise ValueError("calibration overall result disagrees with individual checks")
    return report


def prepare_study(session: Any, config: Mapping[str, Any],
                  context: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize declared baseline/sample preparation through a built-in path.

    Semantic samples and failure cases are generated from current baselines under
    frozen prospective rules. Other studies check declared task inputs locally.
    """
    implementation = _implementation(config, context)
    if config["name"] in {"semantic_heads", "failures"}:
        return implementation.prepare_study(session, dict(config), dict(context))
    return dict(config), {"study": config["name"], "status": "declared_inputs",
                          "preparation": "validated_by_study_calibration_and_evaluator"}


def run_study(session: Any, config: Mapping[str, Any],
              context: Mapping[str, Any]) -> dict[str, Any]:
    implementation = _implementation(config, context)
    calibration = context.get("extension_calibration", {})
    if calibration.get("all_passed") is not True:
        raise ValueError("study-specific measured calibration required")
    report = implementation.run_study(session, dict(config), dict(context))
    required = {"schema", "study", "status", "coverage", "scientific_verdict"}
    if not isinstance(report, dict) or not required.issubset(report):
        raise ValueError("study result missing its schema/coverage/scientific status")
    if report["study"] != config["name"]:
        raise ValueError("study result selector mismatch")
    if report["status"] not in {"complete", "insufficient", "invalid", "failed"} or not isinstance(report["coverage"], dict):
        raise ValueError("study result requires a known technical status and coverage object")
    if not isinstance(report["scientific_verdict"], (str, dict)) or not report["scientific_verdict"]:
        raise ValueError("study result requires an explicit scientific verdict or limitation")
    return report
