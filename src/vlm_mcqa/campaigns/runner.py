"""Explicit execution lifecycle; imports perform no inference or filesystem writes."""
from __future__ import annotations

import copy
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

from .audit import audit_attempt, inventory
from .config import canonical_hash, plan_campaign, validate_config
from .verification import read_object, verify_inputs


def write_new(path: Path, value: Any) -> None:
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()


@contextmanager
def capture_logs(output: Path):
    """Preserve Python stdout/stderr while retaining console progress.

    Called only after the public execution gate and fresh-output checks. Native
    process-level diagnostics may additionally be captured by the shell runbook.
    """
    class Tee:
        def __init__(self, console, stream):
            self.console, self.stream = console, stream
        def write(self, value):
            self.console.write(value)
            self.stream.write(value)
            self.stream.flush()
            return len(value)
        def flush(self):
            self.console.flush()
            self.stream.flush()
        def isatty(self):
            return False
        def __getattr__(self, name):
            return getattr(self.console, name)
    with (output / "stdout.log").open("x", encoding="utf-8") as stdout, \
         (output / "stderr.log").open("x", encoding="utf-8") as stderr:
        with redirect_stdout(Tee(sys.stdout, stdout)), redirect_stderr(Tee(sys.stderr, stderr)):
            try:
                yield
            except BaseException:
                traceback.print_exc()
                raise


def _strict_result(value: Any) -> dict[str, Any]:
    undefined = []
    def convert(item, path):
        if type(item) is float and not math.isfinite(item):
            undefined.append("/".join(path))
            return None
        if isinstance(item, dict):
            return {key: convert(child, [*path, str(key)]) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [convert(child, [*path, str(index)]) for index, child in enumerate(item)]
        return item
    result = convert(value, [])
    if undefined:
        result["undefined_numeric_diagnostics"] = undefined
        result["undefined_numeric_policy"] = "null_in_new_report; native_method_artifacts_retained; no_zero_imputation"
    return result


def _protected(config: Mapping[str, Any], proof: Mapping[str, Any] | None = None) -> list[Path]:
    paths = [Path(config["source"]["manifest"]).parent, Path(config["model"]["checkpoint_dir"]),
             Path(config["data"]["directory"]), Path(config["protocol"]["decision_path"]).parent]
    paths.extend(Path(row["path"]).parent for row in config.get("resolved_documents", []))
    paths.extend(Path(value).parent for value in (proof or {}).get("immutable_evidence_paths", []))
    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                path_keys = {"protocol", "recipe", "contrasts", "sample_plan", "cases", "strata_manifest",
                             "frozen_protocol", "calibration_case", "head_source", "task_assets_root", "source_root"}
                if key == "protected_inputs" and isinstance(item, list):
                    for declared in item:
                        p = Path(declared)
                        paths.append(p if p.is_dir() else p.parent)
                elif (key.endswith(("_path", "_dir")) or key in path_keys) and isinstance(item, str):
                    p = Path(item)
                    paths.append(p if p.is_dir() else p.parent)
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    collect(config.get("study", {}))
    return [p.resolve() for p in paths]


def _ordinary(session):
    from ..core7.calibration import run_calibration
    calibration = run_calibration(session)
    if calibration.get("all_passed") is not True:
        raise RuntimeError("ordinary calibration failed; bypass unsupported")
    return calibration


def calibrate_campaign(config: Mapping[str, Any], *, execute: bool = False) -> dict[str, Any]:
    """Fresh technical attempt; never run science or claim full campaign completion."""
    if execute is not True:
        raise PermissionError("calibration requires explicit execution authorization")
    validate_config(config)
    proof = verify_inputs(config)
    output = Path(config["output_dir"]).resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("calibration requires a fresh attempt in an existing parent")
    if any(output == p or p in output.parents or output in p.parents for p in _protected(config, proof)):
        raise ValueError("calibration output overlaps immutable input")
    output.mkdir(exist_ok=False)
    write_new(output / "attempt.json", {"schema": "campaign-calibration-attempt/v1", "config": dict(config),
                                        "verified_inputs": proof, "mode": "calibration_only"})
    with capture_logs(output):
        return _calibration_steps(config, proof, output)


def _calibration_steps(config: Mapping[str, Any], proof: Mapping[str, Any], output: Path) -> dict[str, Any]:
    try:
        from .session import create_session
        profile = (read_object(config["profiles_config"])["profiles"][config["profile"]]
                   if config["kind"] in {"core7_preselected", "core7_correct_only_reduced"} else None)
        session, _ = create_session(config, output / "session", execute=True, profile=profile, verified=proof)
        core = _ordinary(session)
        extension = None
        if config["kind"] == "stretch":
            from ..stretch.studies import prepare_study, calibrate_study
            study = copy.deepcopy(config["study"]["config"])
            study.update(copy.deepcopy(study.get("phase_settings", {}).get("discovery", {})))
            study.pop("phase_settings", None)
            study.update(name=config["study"]["name"], phase="discovery", model_key=config["model"]["key"])
            study.setdefault("split", "discovery")
            identity = {"model_key": config["model"]["key"], "dataset_hash": proof["dataset_hash"],
                        "image_aware_release_hash": proof["image_aware_release_hash"],
                        "source_identity": proof["source"]["manifest_sha256"],
                        "checkpoint_identity": {"hf_id": proof["model_spec"]["hf_id"], "revision": proof["model_spec"]["revision"]},
                        "eligibility_decision": "approved_for_this_stretch_protocol",
                        "protocol_decision_sha256": proof["protocol_decision_sha256"],
                        "split": "discovery", "confirmation_protocol_frozen": False,
                        "calibration_sha256": canonical_hash(core)}
            identity["calibration_binding"] = {key: identity[key] for key in
                                               ("source_identity", "checkpoint_identity", "image_aware_release_hash")}
            context = {"execute": True, "identity": identity, "core_calibration": core,
                       "output_dir": output / "extension", "preparation_dir": output / "preparation"}
            study, preparation = prepare_study(session, study, context)
            write_new(output / "preparation_report.json", preparation)
            if preparation.get("status") in {"target_shortfall", "insufficient"}:
                result = {"schema": "campaign-calibration-result/v1", "status": "insufficient",
                          "ordinary_all_passed": core["all_passed"], "study_all_passed": None,
                          "reason": "insufficient_baseline_cases_for_extension_calibration",
                          "scientific_run_performed": False, "full_campaign_complete": False}
                write_new(output / "calibration_result.json", result)
                write_new(output / "artifact_manifest.json", {"schema": "campaign-artifacts/v1", "files": inventory(output)})
                return result
            extension = calibrate_study(session, study, context)
            write_new(output / "extension_calibration.json", extension)
            if extension.get("all_passed") is not True:
                raise RuntimeError("study calibration failed")
        result = {"schema": "campaign-calibration-result/v1", "status": "calibrated",
                  "ordinary_all_passed": core["all_passed"], "study_all_passed": extension["all_passed"] if extension else None,
                  "scientific_run_performed": False, "full_campaign_complete": False}
        write_new(output / "calibration_result.json", result)
        write_new(output / "artifact_manifest.json", {"schema": "campaign-artifacts/v1", "files": inventory(output)})
        write_new(output / "audit.json", audit_attempt(output))
        return result
    except BaseException as error:
        write_new(output / "failure.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        if not (output / "artifact_manifest.json").exists():
            write_new(output / "artifact_manifest.json", {"schema": "campaign-artifacts/v1", "files": inventory(output)})
        raise


def technical_pilot_campaign(config: Mapping[str, Any], *, execute: bool = False) -> dict[str, Any]:
    """Run one bounded discovery-shaped timing pilot without creating science.

    The pilot uses the real intervention, controls, preparation, and calibration
    paths.  Its protocol must independently declare an 8--16 pair discovery
    sample.  Any implementation freeze written by the study is retained only as
    a diagnostic artifact and is explicitly ineligible for confirmation.
    """
    if execute is not True:
        raise PermissionError("technical pilot requires explicit execution authorization")
    validate_config(config)
    if config["kind"] != "stretch":
        raise ValueError("technical pilot is available only for stretch campaigns")
    proof = verify_inputs(config)
    output = Path(config["output_dir"]).resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("technical pilot requires a fresh attempt in an existing parent")
    if any(output == p or p in output.parents or output in p.parents for p in _protected(config, proof)):
        raise ValueError("technical pilot output overlaps immutable input")

    base = copy.deepcopy(config["study"]["config"])
    base.update(copy.deepcopy(base.get("phase_settings", {}).get("discovery", {})))
    base.pop("phase_settings", None)
    body = base.get("protocol")
    if not isinstance(body, Mapping):
        raise ValueError("technical pilot requires a resolved protocol document")
    sampling = body.get("sampling", {}).get("discovery", {})
    target = sampling.get("target_pairs")
    minimum = sampling.get("min_pairs")
    if type(target) is not int or type(minimum) is not int or not (8 <= minimum <= target <= 16):
        raise ValueError("technical pilot protocol must declare 8--16 discovery pairs")
    if body.get("claim_boundary") != "calibration_and_technical_pilots_are_not_scientific_evidence":
        raise ValueError("technical pilot protocol must deny scientific use")

    output.mkdir(exist_ok=False)
    write_new(output / "attempt.json", {
        "schema": "campaign-technical-pilot-attempt/v1", "config": dict(config),
        "verified_inputs": proof, "mode": "technical_pilot", "phase": "discovery",
        "target_pairs": target, "scientific_run_performed": False,
    })
    try:
        with capture_logs(output):
            from .session import create_session
            from ..stretch.studies import prepare_study, calibrate_study, run_study
            session, _ = create_session(config, output / "session", execute=True, verified=proof)
            core = _ordinary(session)
            study = base
            study.update(name=config["study"]["name"], phase="discovery",
                         split="discovery", model_key=config["model"]["key"])
            identity = {
                "model_key": config["model"]["key"], "dataset_hash": proof["dataset_hash"],
                "image_aware_release_hash": proof["image_aware_release_hash"],
                "source_identity": proof["source"]["manifest_sha256"],
                "checkpoint_identity": {"hf_id": proof["model_spec"]["hf_id"],
                                        "revision": proof["model_spec"]["revision"]},
                "eligibility_decision": "approved_for_this_stretch_protocol",
                "protocol_decision_sha256": proof["protocol_decision_sha256"],
                "split": "discovery", "confirmation_protocol_frozen": False,
                "calibration_sha256": canonical_hash(core),
            }
            identity["calibration_binding"] = {key: identity[key] for key in
                                               ("source_identity", "checkpoint_identity",
                                                "image_aware_release_hash")}
            context = {"execute": True, "identity": identity, "core_calibration": core,
                       "output_dir": output / "extension", "preparation_dir": output / "preparation"}
            study, preparation = prepare_study(session, study, context)
            write_new(output / "preparation_report.json", preparation)
            if preparation.get("status") != "ready":
                result = {"schema": "campaign-technical-pilot-result/v1", "status": "insufficient",
                          "reason": "pilot_sample_not_ready", "preparation": preparation,
                          "scientific_run_performed": False, "confirmation_eligible": False}
            else:
                extension = calibrate_study(session, study, context)
                write_new(output / "extension_calibration.json", extension)
                if extension.get("all_passed") is not True:
                    raise RuntimeError("technical pilot extension calibration failed")
                context.update(output_dir=output / "pilot", extension_calibration=extension)
                measured = run_study(session, study, context)
                write_new(output / "measured_study_result.json", measured)
                result = {"schema": "campaign-technical-pilot-result/v1",
                          "status": "technical_complete" if measured.get("status") == "complete" else measured.get("status", "invalid"),
                          "phase": "discovery-shaped-pilot", "target_pairs": target,
                          "coverage": measured.get("coverage", {}),
                          "measured_status": measured.get("status"),
                          "scientific_run_performed": False, "confirmation_eligible": False,
                          "claim_boundary": "timing_and_feasibility_only; excluded_from_scientific_claims_and_confirmation_freeze"}
            write_new(output / "pilot_result.json", result)
        write_new(output / "completion.json", {"status": result["status"],
                                                "scientific_run_performed": False,
                                                "confirmation_eligible": False})
        write_new(output / "artifact_manifest.json", {"schema": "campaign-artifacts/v1",
                                                        "files": inventory(output)})
        return result
    except BaseException as error:
        write_new(output / "failure.json", {"status": "failed", "error_type": type(error).__name__,
                                            "error": str(error), "scientific_run_performed": False})
        if not (output / "artifact_manifest.json").exists():
            write_new(output / "artifact_manifest.json", {"schema": "campaign-artifacts/v1",
                                                            "files": inventory(output)})
        raise


def run_stretch(config: Mapping[str, Any], proof: Mapping[str, Any], output: Path, *, execute: bool) -> dict:
    from .session import create_session
    from ..core7.session import Session
    from ..stretch.studies import prepare_study, calibrate_study, run_study

    retry = None
    if config.get("retry_from"):
        from .retry import (copy_stretch_confirmation_preparation,
                            copy_stretch_discovery, plan_retry)
        retry = plan_retry(config["retry_from"], config, proof)
        write_new(output / "retry_link.json", retry)
        reused_discovery = copy_stretch_discovery(retry, output)
        reused_confirmation_preparation = copy_stretch_confirmation_preparation(retry, output)
    else:
        reused_discovery = False
        reused_confirmation_preparation = False
    session, _ = create_session(config, output / ("retry_session" if reused_discovery else "discovery_session"),
                                execute=execute, verified=proof)
    base = copy.deepcopy(config["study"]["config"])
    base["name"] = config["study"]["name"]
    identity = {"model_key": config["model"]["key"], "dataset_hash": proof["dataset_hash"],
                "image_aware_release_hash": proof["image_aware_release_hash"],
                "source_identity": proof["source"]["manifest_sha256"],
                "checkpoint_identity": {"hf_id": proof["model_spec"]["hf_id"], "revision": proof["model_spec"]["revision"]},
                "eligibility_decision": "approved_for_this_stretch_protocol",
                "protocol_decision_sha256": proof["protocol_decision_sha256"]}
    results = {"discovery": read_object(output / "discovery_result.json")} if reused_discovery else {}
    frozen = read_object(output / "study_freeze.json") if reused_discovery else None
    confirmation_sensitivity = base.get("confirmation_sample_sensitivity")
    if confirmation_sensitivity is not None:
        if not reused_discovery or base["name"] != "semantic_heads":
            raise ValueError("confirmation sample sensitivity requires reused semantic-head discovery")
        if (confirmation_sensitivity.get("label") != "reduced_heldout_sample_sensitivity"
                or confirmation_sensitivity.get("target_pairs")
                != confirmation_sensitivity.get("min_pairs")):
            raise ValueError("invalid confirmation sample sensitivity")
    discovery_calibration_contrast_id = None
    if (reused_discovery and base["name"] == "semantic_heads"
            and "calibration_contrast_id" in base):
        discovery_preparation = read_object(output / "discovery_preparation_report.json")
        discovery_calibration_contrast_id = discovery_preparation.get("calibration_contrast_id")
        if not isinstance(discovery_calibration_contrast_id, str) or not discovery_calibration_contrast_id:
            raise ValueError("reused semantic-head discovery omitted its technical calibration direction")
    retry_confirmation_identity = None
    if reused_discovery and base["name"] in {"semantic_heads", "route", "failures"}:
        implementation_freeze = read_object(output / "discovery/protocol_freeze.json")
        frozen_identity = implementation_freeze.get("identity")
        if not isinstance(frozen_identity, dict):
            raise ValueError("reused discovery protocol freeze omitted its identity binding")
        retry_confirmation_identity = copy.deepcopy(identity)
        for key in ("model_key", "dataset_hash", "checkpoint_identity", "image_aware_release_hash"):
            if frozen_identity.get(key) != identity[key]:
                raise ValueError(f"reused discovery {key} differs from the verified retry")
        if not frozen_identity.get("source_identity"):
            raise ValueError("reused discovery protocol freeze omitted source_identity")
        # A verified retry may execute newer orchestration code while retaining
        # the exact scientific discovery freeze.  Confirmation must therefore
        # bind to that discovery identity; the current execution source remains
        # explicit for provenance instead of silently replacing the freeze.
        retry_confirmation_identity["retry_execution_source_identity"] = identity["source_identity"]
        retry_confirmation_identity["source_identity"] = frozen_identity["source_identity"]
    for split in config["study"]["phases"]:
        if split == "discovery" and reused_discovery:
            continue
        if split == "confirmation":
            if verify_inputs(config) != proof:
                raise ValueError("inputs changed between discovery and confirmation")
            session = Session(session.adapter, session.data_dir, output / "confirmation_session",
                              session.profile, seed=config["seed"])
        phase_config = copy.deepcopy(base)
        # Optional per-phase settings belong in the declared config; confirmation
        # receives immutable discovery decisions rather than choosing them again.
        phase_config.update(copy.deepcopy(base.get("phase_settings", {}).get(split, {})))
        phase_config.pop("phase_settings", None)
        phase_config.update(phase=split, model_key=identity["model_key"])
        if base["name"] != "external":
            phase_config["split"] = split
        else:
            phase_config.setdefault("split", split)
        if frozen:
            # Each implementation owns its exact validated freeze schema.
            # Preserve the actual discovery file; do not substitute our outer
            # campaign provenance envelope for the scientific protocol freeze.
            if base["name"] in {"semantic_heads", "route", "failures"}:
                phase_config["frozen_protocol"] = str(output / "discovery/protocol_freeze.json")
                phase_config.pop("protocol", None)
                if split == "confirmation" and confirmation_sensitivity is not None:
                    primary_freeze = read_object(output / "discovery/protocol_freeze.json")
                    sensitivity_freeze = copy.deepcopy(primary_freeze)
                    sensitivity_body = sensitivity_freeze["protocol"]
                    original_sampling = copy.deepcopy(sensitivity_body["sampling"]["confirmation"])
                    target = confirmation_sensitivity["target_pairs"]
                    if not (target < original_sampling["min_pairs"]
                            <= original_sampling["target_pairs"]):
                        raise ValueError("confirmation sensitivity does not reduce the frozen shortfall")
                    sensitivity_body["sampling"]["confirmation"].update(
                        target_pairs=target, min_pairs=target)
                    sensitivity_body["evidence_status"] = "reduced_heldout_sample_sensitivity"
                    sensitivity_body["primary_confirmation_sampling"] = original_sampling
                    sensitivity_body["confirmation_sensitivity_reason"] = confirmation_sensitivity["reason"]
                    sensitivity_freeze["primary_protocol_sha256"] = primary_freeze["protocol_sha256"]
                    sensitivity_freeze["protocol_sha256"] = canonical_hash(sensitivity_body)
                    sensitivity_freeze["confirmation_sample_sensitivity"] = copy.deepcopy(
                        confirmation_sensitivity)
                    sensitivity_path = output / "confirmation_sensitivity_protocol.json"
                    write_new(sensitivity_path, sensitivity_freeze)
                    phase_config["frozen_protocol"] = str(sensitivity_path)
            else:
                phase_config["freeze_path"] = str(output / "discovery/freeze.json")
                if base["name"] in {"spatial", "image_options"}:
                    confirmation_protocol = frozen["decisions"].get("confirmation_protocol")
                    if not isinstance(confirmation_protocol, dict) or not confirmation_protocol:
                        raise ValueError("task discovery freeze omitted its confirmation protocol")
                    phase_config["protocol"] = copy.deepcopy(confirmation_protocol)
        if (split == "confirmation" and base["name"] == "semantic_heads"
                and "calibration_contrast_id" in base):
            if not discovery_calibration_contrast_id:
                raise ValueError("semantic-head confirmation requires the discovery calibration direction")
            phase_config["calibration_contrast_id"] = discovery_calibration_contrast_id
            if reused_confirmation_preparation:
                phase_config["sample_plan"] = str(output / "confirmation_preparation/sample_plan.json")
        core = _ordinary(session)
        phase_base_identity = retry_confirmation_identity if split == "confirmation" and retry_confirmation_identity else identity
        phase_identity = {**phase_base_identity, "split": split, "confirmation_protocol_frozen": split == "confirmation",
                          "calibration_sha256": canonical_hash(core)}
        phase_identity["calibration_binding"] = {key: phase_identity[key] for key in
                                                ("source_identity", "checkpoint_identity", "image_aware_release_hash")}
        calibration_dir = output / f"{split}_calibration"
        context = {"execute": execute, "output_dir": calibration_dir, "identity": phase_identity,
                   "core_calibration": core, "preparation_dir": output / f"{split}_preparation",
                   "protocol_decision": copy.deepcopy(proof.get("protocol_decision"))}
        phase_config, preparation = prepare_study(session, phase_config, context)
        if split == "discovery" and base["name"] == "semantic_heads":
            discovery_calibration_contrast_id = phase_config.get("calibration_contrast_id")
        write_new(output / f"{split}_preparation_report.json", preparation)
        if preparation.get("status") in {"target_shortfall", "insufficient"}:
            result = {"schema": "stretch-preparation-insufficiency/v1", "study": base["name"],
                      "status": "insufficient", "coverage": preparation,
                      "scientific_verdict": "insufficient_baseline_eligible_sample"}
            write_new(output / f"{split}_result.json", result)
            results[split] = result
            return {"schema": "stretch-campaign-result/v1", "study": base["name"], "status": "insufficient",
                    "phases": results, "scientific_verdict": result["scientific_verdict"]}
        extension = calibrate_study(session, phase_config, context)
        if extension.get("all_passed") is not True:
            write_new(output / f"{split}_extension_calibration.json", extension)
            raise RuntimeError(f"{split} study calibration failed")
        write_new(output / f"{split}_extension_calibration.json", extension)
        context.update(output_dir=output / split, extension_calibration=extension)
        result = run_study(session, phase_config, context)
        write_new(output / f"{split}_result.json", result)
        results[split] = result
        if result.get("status") != "complete":
            return {"schema": "stretch-campaign-result/v1", "study": base["name"],
                    "status": result.get("status", "invalid"), "phases": results,
                    "scientific_verdict": result.get("scientific_verdict", "not_established")}
        if split == "discovery":
            implementation_freeze = output / "discovery" / ("protocol_freeze.json" if base["name"] in
                                                           {"semantic_heads", "route", "failures"} else "freeze.json")
            if not implementation_freeze.is_file():
                raise ValueError("complete discovery did not publish its required implementation freeze")
            frozen = {"schema": "study-discovery-freeze/v1", "study": base["name"],
                      "model_key": identity["model_key"], "dataset_hash": identity["dataset_hash"],
                      "source_identity": identity["source_identity"], "checkpoint_identity": identity["checkpoint_identity"],
                      "protocol_decision_sha256": proof["protocol_decision_sha256"],
                      "study_config": base, "decisions": read_object(implementation_freeze),
                      "discovery_result_sha256": canonical_hash(result)}
            frozen["freeze_sha256"] = canonical_hash(frozen)
            write_new(output / "study_freeze.json", frozen)
    return {"schema": "stretch-campaign-result/v1", "study": base["name"], "status": "complete",
            "phases": results, "scientific_verdict": results["confirmation"]["scientific_verdict"]}


def run_campaign(config: Mapping[str, Any], *, execute: bool = False) -> dict[str, Any]:
    if execute is not True:
        raise PermissionError("campaign execution requires explicit execute=True after user authorization")
    validate_config(config)
    proof = verify_inputs(config)
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError("campaign attempt already exists; retries require a fresh directory")
    for source in _protected(config, proof):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError(f"output overlaps immutable input: {source}")
    if not output.parent.is_dir():
        raise ValueError("explicit existing output parent required")
    plan = plan_campaign(config)
    header = {"schema": "campaign-attempt/v1", "config": copy.deepcopy(dict(config)), "plan": plan,
              "verified_inputs": proof, "status": "started", "execution_authorization": "explicit_call"}
    # Strict serialization happens before directory creation.
    json.dumps(header, allow_nan=False)
    output.mkdir(exist_ok=False)
    write_new(output / "attempt.json", header)
    try:
        with capture_logs(output):
            if config["kind"] in {"core7_preselected", "core7_correct_only_reduced"}:
                from .core7 import run_pali
                result = run_pali(config, proof, output, execute=execute)
            else:
                result = run_stretch(config, proof, output, execute=execute)
        result = _strict_result(result)
        write_new(output / "result.json", result)
        write_new(output / "completion.json", {"status": result["status"], "config_sha256": plan["config_sha256"],
                                                "scientific_verdict": result.get("scientific_verdict", "not_established")})
        write_new(output / "artifact_manifest.json", {"schema": "campaign-artifacts/v1", "files": inventory(output)})
        if result["status"] == "complete":
            audit = audit_attempt(output)
            write_new(output / "audit.json", audit)
        return result
    except BaseException as error:
        write_new(output / "failure.json", {"status": "failed", "error_type": type(error).__name__,
                                            "error": str(error), "config_sha256": plan["config_sha256"]})
        # Evidence is preserved even if the attempt never reaches completion.
        if not (output / "artifact_manifest.json").exists():
            write_new(output / "artifact_manifest.json", {"schema": "campaign-artifacts/v1", "files": inventory(output)})
        raise
