"""Strict campaign configuration and side-effect-free stage planning.

Reading configuration is explicit. Planning never loads models, probes devices,
contacts services, creates outputs or infers placement.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from . import SCHEMA

STUDIES = {"semantic_heads", "route", "failures", "spatial", "image_options", "external"}


def campaign_spec() -> dict[str, Any]:
    return {"schema": SCHEMA, "kinds": ["core7_preselected", "core7_correct_only_reduced", "stretch"],
            "studies": sorted(STUDIES), "compute_server": None,
            "checkpoint_policy": "verified_local_materialized_native_safetensors_only",
            "execution_default": "denied", "offload_policy": "forbid"}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _required(value: Mapping[str, Any], key: str, kind: type) -> Any:
    item = value.get(key)
    if type(item) is not kind or not item:
        raise ValueError(f"required nonempty {kind.__name__}: {key}")
    return item


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != SCHEMA or config.get("kind") not in {"core7_preselected", "core7_correct_only_reduced", "stretch"}:
        raise ValueError("unsupported campaign schema/kind")
    for key in ("source", "model", "data", "deployment", "protocol"):
        _required(config, key, dict)
    _required(config, "output_dir", str)
    for key in ("root", "manifest"):
        _required(config["source"], key, str)
    for key in ("key", "models_config", "checkpoint_dir", "checkpoint_manifest"):
        _required(config["model"], key, str)
    for key in ("directory", "dataset_hash", "image_aware_release_hash"):
        _required(config["data"], key, str)
    deployment = config["deployment"]
    if deployment.get("dtype") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("explicit supported dtype required")
    placement = deployment.get("device_map")
    if not ((isinstance(placement, str) and placement) or (isinstance(placement, dict) and placement)):
        raise ValueError("explicit device_map required; no automatic device selection")
    if placement == "auto":
        raise ValueError("provide an explicit placement, not auto allocation")
    devices = placement.values() if isinstance(placement, dict) else [placement]
    if isinstance(placement, dict) and any(not isinstance(key, str) for key in placement):
        raise ValueError("device-map module keys must be strings")
    if any(not ((type(device) is int and device >= 0)
                or (isinstance(device, str) and re.fullmatch(r"cuda:\d+", device))) for device in devices):
        raise ValueError("explicit logical CUDA indices/maps required; auto/cpu/disk/meta/placeholders forbidden")
    if deployment.get("offload_policy") != "forbid":
        raise ValueError("initial implementation supports explicit accelerator placement without offload")
    if type(deployment.get("batch_size")) is not int or deployment["batch_size"] < 1:
        raise ValueError("positive batch_size required")
    if type(config.get("seed")) is not int:
        raise ValueError("explicit integer seed required")
    protocol = config["protocol"]
    for key in ("decision_path", "decision_sha256"):
        _required(protocol, key, str)
    if not re.fullmatch(r"[0-9a-f]{64}", protocol["decision_sha256"]):
        raise ValueError("protocol decision SHA256 required")
    if config["kind"] in {"core7_preselected", "core7_correct_only_reduced"}:
        if config["model"]["key"] != "paligemma2_10b_mix":
            raise ValueError("this requested Core-7 campaign route is PaliGemma2-10B only; LLaVA is complete")
        expected_mode = ("preselected_protocol_extension" if config["kind"] == "core7_preselected"
                         else "preselected_correct_only_sensitivity")
        if protocol.get("selection_mode") != expected_mode:
            raise ValueError("Pali campaign must explicitly declare its full or correct-only status")
        expected_profile = "full_core7" if config["kind"] == "core7_preselected" else "pali_correct_only_reduced"
        if config.get("profile") != expected_profile:
            raise ValueError("Pali campaign profile does not match its declared scientific route")
        _required(config, "profiles_config", str)
    else:
        study = _required(config, "study", dict)
        if study.get("name") not in STUDIES:
            raise ValueError("unimplemented study name")
        _required(study, "config", dict)
        if set(study.get("phases", [])) != {"discovery", "confirmation"} or study["phases"] != ["discovery", "confirmation"]:
            raise ValueError("explicit discovery then confirmation phases required")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    # Resolve only declared path fields. Arbitrary strings and model identifiers
    # must not accidentally become paths. Study paths use explicit *_path keys.
    fields = [(config, "output_dir"), (config.get("source", {}), "root"),
              (config.get("source", {}), "manifest"), (config.get("model", {}), "models_config"),
              (config.get("model", {}), "checkpoint_dir"), (config.get("model", {}), "checkpoint_manifest"),
              (config.get("data", {}), "directory"), (config.get("protocol", {}), "decision_path")]
    if "profiles_config" in config:
        fields.append((config, "profiles_config"))
    if "retry_from" in config:
        fields.append((config, "retry_from"))
    for owner, key in fields:
        if key in owner:
            value = Path(owner[key])
            owner[key] = str((path.parent / value).resolve())
    documents = []
    def study_paths(value: Any, base: Path) -> Any:
        if isinstance(value, dict):
            path_keys = {"protocol", "recipe", "contrasts", "sample_plan", "cases", "strata_manifest",
                         "frozen_protocol", "calibration_case", "head_source", "task_assets_root", "source_root"}
            result = {}
            for key, item in value.items():
                if key == "protected_inputs" and isinstance(item, list):
                    result[key] = [str((base / Path(p)).resolve()) for p in item]
                elif key == "protocol" and isinstance(item, str) and (base / item).is_file():
                    document_path = (base / item).resolve()
                    payload = document_path.read_bytes()
                    documents.append({"path": str(document_path), "sha256": hashlib.sha256(payload).hexdigest()})
                    result[key] = study_paths(json.loads(payload), document_path.parent)
                elif (key.endswith(("_path", "_dir")) or key in path_keys) and isinstance(item, str):
                    target = (base / Path(item)).resolve()
                    result[key] = str(target)
                    if target.is_file():
                        documents.append({"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
                else:
                    result[key] = study_paths(item, base)
            return result
        if isinstance(value, list):
            return [study_paths(v, base) for v in value]
        return value
    if "study" in config:
        config["study"]["config"] = study_paths(config["study"].get("config", {}), path.parent)
    config["resolved_documents"] = documents
    validate_config(config)
    return config


def plan_campaign(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    stages = ["verify_inputs", "protocol_decision", "create_session", "ordinary_calibration"]
    if config["kind"] in {"core7_preselected", "core7_correct_only_reduced"}:
        stages += ["discovery_method_eligibility", "discovery", "freeze", "confirmation_calibration",
                   "confirmation_method_eligibility", "confirmation", "confirm", "audit"]
    else:
        stages += ["discovery_study_preparation", "discovery_study_calibration", "discovery", "freeze",
                   "confirmation_study_preparation", "confirmation_study_calibration", "confirmation", "analysis", "audit"]
    return {"schema": "campaign-plan/v1", "config_sha256": canonical_hash(config), "stages": stages,
            "compute_server": None, "execution_authorized": False, "runtime_validated": False}
