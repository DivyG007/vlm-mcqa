"""Explicit restart planning and verified whole-phase reuse into fresh attempts."""
from __future__ import annotations

import copy
from pathlib import Path
import shutil

from .audit import audit_core7_phase
from .config import canonical_hash
from .verification import file_hash, read_object


def scientific_config(config):
    return {k: v for k, v in config.items() if k not in {"output_dir", "retry_from"}}


def _retry_compatible(previous, current):
    """Allow only cache-capacity deployment changes for a stretch restart."""
    old, new = scientific_config(previous), scientific_config(current)
    if old.get("kind") != "stretch" or new.get("kind") != "stretch":
        return old == new
    old = dict(old)
    new = dict(new)
    old_deployment = dict(old.pop("deployment"))
    new_deployment = dict(new.pop("deployment"))
    for key in ("prepared_cache", "clean_cache"):
        old_deployment.pop(key, None)
        new_deployment.pop(key, None)
    return old == new and old_deployment == new_deployment


def _stretch_signature(config, proof):
    """Scientific identity independent of commit-addressed and node-local paths."""
    source_root = str(Path(config["source"]["root"]).resolve())
    checkpoint_root = str(Path(config["model"]["checkpoint_dir"]).resolve())
    study_hashes = proof.get("study_input_file_sha256", {})

    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()
                    if key not in {"protected_inputs"}}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, str):
            resolved = str(Path(value).resolve()) if value.startswith("/") else value
            if resolved in study_hashes:
                return {"file_sha256": study_hashes[resolved]}
            if resolved == source_root or resolved.startswith(source_root + "/"):
                return "$SOURCE/" + resolved[len(source_root):].lstrip("/")
            if resolved == checkpoint_root or resolved.startswith(checkpoint_root + "/"):
                return "$CHECKPOINT/" + resolved[len(checkpoint_root):].lstrip("/")
        return value

    deployment = copy.deepcopy(config["deployment"])
    for key in ("prepared_cache", "clean_cache", "device_map"):
        deployment.pop(key, None)
    # Resolved documents are content-addressed inputs.  Their launch-directory
    # paths change for every immutable retry snapshot, while the verified hash
    # is the scientific identity that must remain fixed.
    documents = [{"sha256": document["sha256"]}
                 for document in config.get("resolved_documents", [])]
    return {
        "schema": config["schema"], "kind": config["kind"], "seed": config["seed"],
        "model": {"key": config["model"]["key"], "spec": proof["model_spec"]},
        "data": {"dataset_hash": proof["dataset_hash"],
                 "image_aware_release_hash": proof["image_aware_release_hash"]},
        "deployment": normalize(deployment),
        "session_profile": normalize(config.get("session_profile", {})),
        "protocol_decision_sha256": proof["protocol_decision_sha256"],
        "resolved_documents": documents,
        "study": normalize(config["study"]),
    }


def _placement_retry_compatible(previous, current):
    """Permit GPU-only remapping of the same modules after a measured OOM."""
    old = previous.get("deployment", {}).get("device_map")
    new = current.get("deployment", {}).get("device_map")
    if not isinstance(old, dict) or not isinstance(new, dict) or set(old) != set(new):
        return False
    # A measured memory failure may require spreading the exact same modules
    # over an additional GPU. Scientific identity is unchanged because no
    # module, input, intervention, or phase setting may be added or removed.
    allowed = {f"cuda:{index}" for index in range(4)}
    return bool(old) and set(old.values()) <= allowed and set(new.values()) <= allowed


def _confirmation_sample_sensitivity_compatible(previous_root, old_config, new_config,
                                                old_proof, new_proof):
    """Permit one explicit E20 retry on every available held-out pair.

    The primary attempt must already have stopped before confirmation
    interventions solely because its deterministic control-complete cohort was
    smaller than the frozen minimum.  The sensitivity may only set both the
    target and minimum to that exact observed cohort size; every other
    scientific input remains unchanged.
    """
    sensitivity = new_config.get("study", {}).get("config", {}).get(
        "confirmation_sample_sensitivity")
    if not isinstance(sensitivity, dict):
        return False
    required = {"target_pairs", "min_pairs", "primary_attempt", "label", "reason"}
    if set(sensitivity) != required or sensitivity.get("label") != "reduced_heldout_sample_sensitivity":
        return False
    target = sensitivity.get("target_pairs")
    if type(target) is not int or target < 1 or sensitivity.get("min_pairs") != target:
        return False
    old_without = copy.deepcopy(old_config)
    new_without = copy.deepcopy(new_config)
    new_without["study"]["config"].pop("confirmation_sample_sensitivity", None)
    if (_stretch_signature(old_without, old_proof)
            != _stretch_signature(new_without, new_proof)):
        return False
    previous_root = Path(previous_root).resolve()
    if str(previous_root) != str(Path(sensitivity["primary_attempt"]).resolve()):
        return False
    result_path = previous_root / "confirmation_result.json"
    completion_path = previous_root / "completion.json"
    if not result_path.is_file() or not completion_path.is_file():
        return False
    result = read_object(result_path)
    completion = read_object(completion_path)
    coverage = result.get("coverage", {}).get("coverage", {})
    return (
        result.get("scientific_verdict") == "insufficient_baseline_eligible_sample"
        and completion.get("status") == "insufficient"
        and coverage.get("control_complete_pairs") == target
        and coverage.get("selected_pairs") == target
    )


def plan_retry(previous, config, current_proof=None):
    previous = Path(previous).resolve()
    old = read_object(previous / "attempt.json")
    if config["kind"] == "stretch" and current_proof is not None:
        same_science = (_stretch_signature(old["config"], old["verified_inputs"])
                        == _stretch_signature(config, current_proof))
        declared_sensitivity = _confirmation_sample_sensitivity_compatible(
            previous, old["config"], config, old["verified_inputs"], current_proof)
        compatible = ((same_science or declared_sensitivity)
                      and _placement_retry_compatible(old["config"], config))
    else:
        compatible = _retry_compatible(old["config"], config)
    if not compatible:
        raise ValueError("retry changes scientific/input/deployment configuration")
    manifest = read_object(previous / "artifact_manifest.json")["files"]
    reusable = []
    if config["kind"] in {"core7_preselected", "core7_correct_only_reduced"} and (previous / "discovery/run.json").is_file():
        run = read_object(previous / "discovery/run.json")
        if run.get("status") == "complete":
            audit_core7_phase(previous / "discovery", "discovery")
            files = {k: v for k, v in manifest.items() if k.startswith("discovery/")}
            if not files:
                raise ValueError("prior discovery missing artifact inventory")
            actual = {p.relative_to(previous).as_posix() for p in (previous / "discovery").rglob("*") if p.is_file()}
            if actual != set(files) or any(file_hash(previous / name) != row["sha256"] for name, row in files.items()):
                raise ValueError("prior discovery files changed or unrecorded")
            reusable.append({"phase": "discovery", "source": str(previous / "discovery"), "files": files})
    elif config["kind"] == "stretch" and (previous / "discovery_result.json").is_file():
        result = read_object(previous / "discovery_result.json")
        freeze = previous / "study_freeze.json"
        if result.get("status") == "complete" and freeze.is_file():
            prefixes = ("discovery/", "discovery_calibration/", "discovery_preparation/",
                        "discovery_session/")
            names = {"discovery_result.json", "discovery_preparation_report.json",
                     "discovery_extension_calibration.json", "study_freeze.json"}
            files = {name: row for name, row in manifest.items()
                     if name in names or name.startswith(prefixes)}
            if not files or "study_freeze.json" not in files or "discovery_result.json" not in files:
                raise ValueError("prior stretch discovery missing artifact inventory")
            if any(not (previous / name).is_file() or file_hash(previous / name) != row["sha256"]
                   for name, row in files.items()):
                raise ValueError("prior stretch discovery artifacts changed or are missing")
            reusable.append({"phase": "stretch_discovery", "source": str(previous), "files": files})
        confirmation_plan = previous / "confirmation_preparation/sample_plan.json"
        confirmation_report = previous / "confirmation_preparation_report.json"
        if confirmation_plan.is_file() and confirmation_report.is_file():
            report = read_object(confirmation_report)
            plan = read_object(confirmation_plan)
            if report.get("status") == "ready" and plan.get("status") == "ready":
                prefix = "confirmation_preparation/"
                files = {name: row for name, row in manifest.items() if name.startswith(prefix)}
                required = {"confirmation_preparation/baseline.jsonl",
                            "confirmation_preparation/sample_plan.json",
                            "confirmation_preparation/preparation.json"}
                if not required <= set(files):
                    raise ValueError("prior confirmation preparation missing artifact inventory")
                if any(not (previous / name).is_file() or file_hash(previous / name) != row["sha256"]
                       for name, row in files.items()):
                    raise ValueError("prior confirmation preparation artifacts changed or are missing")
                reusable.append({"phase": "stretch_confirmation_preparation",
                                 "source": str(previous), "files": files})
    record = {"schema": "campaign-restart/v1", "previous_attempt": str(previous),
              "previous_header_sha256": file_hash(previous / "attempt.json"), "reusable": reusable,
              "policy": "new attempt; audited discovery and completed confirmation preparation may be reused; interventions restart",
              "deployment_placement": {
                  "previous_sha256": canonical_hash(old["config"].get("deployment", {}).get("device_map")),
                  "current_sha256": canonical_hash(config.get("deployment", {}).get("device_map")),
                  "policy": "same modules; GPU-only placement may change after measured OOM",
              }}
    if config.get("study", {}).get("config", {}).get("confirmation_sample_sensitivity"):
        record["confirmation_sample_sensitivity"] = copy.deepcopy(
            config["study"]["config"]["confirmation_sample_sensitivity"])
    record["restart_sha256"] = canonical_hash(record)
    return record


def copy_discovery(record, destination):
    if not record["reusable"]:
        return False
    source = Path(record["reusable"][0]["source"])
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination)
    return True


def copy_stretch_discovery(record, destination):
    reusable = [row for row in record["reusable"] if row["phase"] == "stretch_discovery"]
    if not reusable:
        return False
    source = Path(reusable[0]["source"])
    destination = Path(destination)
    for name in reusable[0]["files"]:
        target = destination / name
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    return True


def copy_stretch_confirmation_preparation(record, destination):
    """Reuse an audited completed baseline/sample plan, never interventions."""
    reusable = [row for row in record.get("reusable", [])
                if row["phase"] == "stretch_confirmation_preparation"]
    if not reusable:
        return False
    source = Path(reusable[0]["source"])
    destination = Path(destination)
    for name in reusable[0]["files"]:
        target = destination / name
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    return True
